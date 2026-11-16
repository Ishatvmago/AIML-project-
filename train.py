"""
Trains a flight delay classifier and logs the run to MLflow.

Target: delayed = 1 if the flight arrived 15+ minutes late (US DOT definition).

Algorithms: logreg (Logistic Regression), dt (Decision Tree), rf (Random Forest),
            et (Extra Trees), gb (Gradient Boosting), mlp (neural network)

Every run is logged to MLflow. The model used by the API (model/model.joblib) is only
replaced when --save or --register is given, so experiments never overwrite it by accident.

Examples:
    python train.py --algorithm logreg
    python train.py --algorithm dt --max_depth 10
    python train.py --algorithm et --n_estimators 300 --max_depth 20
    python train.py --algorithm mlp --hidden 64,32
    python train.py --algorithm gb --cv 5                    # also 5-fold cross-validation
    python train.py --algorithm rf --n_estimators 100 --max_depth 6
    python train.py --algorithm gb --max_iter 200 --learning_rate 0.1
    python train.py --algorithm gb --feature_set base        # without the date/holiday features
    python train.py --algorithm gb --register                # also register in the MLflow Model Registry
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from features import HOLIDAY_WINDOW, add_date_features, holiday_window_names

DATA_PATH = os.path.join("data", "flights.csv")
MODEL_PATH = os.path.join("model", "model.joblib")
META_PATH = os.path.join("model", "metadata.json")

REGISTERED_MODEL = "flight-delay-model"
CATEGORICAL = ["airline", "origin", "dest"]
FEATURE_SETS = {
    # base: schedule features only (first experiments)
    "base": ["month", "day_of_week", "dep_hour", "distance"],
    # date: adds the day of the month and distance to the nearest US public holiday
    "date": ["month", "day_of_month", "day_of_week", "days_to_holiday", "dep_hour", "distance"],
}
TARGET = "delayed"
TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",                                            # Decision Tree / RF / Extra Trees
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",   # Gradient Boosting trees
    "sklearn.neural_network._stochastic_optimizers.AdamOptimizer",        # neural network optimiser
]


def parse_args():
    p = argparse.ArgumentParser(description="Train a flight delay model")
    p.add_argument("--algorithm", choices=["logreg", "dt", "rf", "et", "gb", "mlp"], default="rf")
    p.add_argument("--n_estimators", type=int, default=100, help="rf/et: number of trees")
    p.add_argument("--max_depth", type=int, default=None, help="dt/rf/et/gb: max tree depth")
    p.add_argument("--max_iter", type=int, default=200, help="gb: boosting rounds / logreg: iterations")
    p.add_argument("--learning_rate", type=float, default=0.1, help="gb: learning rate")
    p.add_argument("--max_leaf_nodes", type=int, default=31, help="gb: leaves per tree")
    p.add_argument("--l2", type=float, default=0.0, help="gb: L2 regularisation")
    p.add_argument("--C", type=float, default=1.0, help="logreg: inverse regularisation strength")
    p.add_argument("--hidden", default="64,32", help="mlp: neurons per hidden layer, e.g. 64,32")
    p.add_argument("--alpha", type=float, default=1e-4, help="mlp: L2 regularisation")
    p.add_argument("--cv", type=int, default=0, help="also run k-fold cross-validation on the training set")
    p.add_argument("--save", action="store_true",
                   help="replace model/model.joblib (the model the API serves) with this model")
    p.add_argument("--feature_set", choices=list(FEATURE_SETS), default="date")
    p.add_argument("--register", action="store_true",
                   help="save this model for the API and register it in the MLflow Model Registry "
                        "with the alias 'production'")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_params(args):
    """Only the hyperparameters the chosen algorithm really uses (logged to MLflow)."""
    used = {
        "logreg": ["C", "max_iter"],
        "dt": ["max_depth"],
        "rf": ["n_estimators", "max_depth"],
        "et": ["n_estimators", "max_depth"],
        "gb": ["max_iter", "learning_rate", "max_depth", "max_leaf_nodes", "l2"],
        "mlp": ["hidden", "alpha", "max_iter"],
    }[args.algorithm]
    common = ["algorithm", "feature_set", "test_size", "seed"]
    return {k: getattr(args, k) for k in common + used}


def build_model(args, NUMERIC):
    if args.algorithm in ("logreg", "mlp"):
        # linear models and neural networks need numbers on the same scale
        prep = ColumnTransformer([
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ])
        if args.algorithm == "logreg":
            clf = LogisticRegression(C=args.C, max_iter=args.max_iter, class_weight="balanced")
        else:
            # early stopping holds back 10% of the training data and stops when it stops improving;
            # class balance is handled with sample weights in fit_params()
            clf = MLPClassifier(hidden_layer_sizes=tuple(int(n) for n in args.hidden.split(",")),
                                alpha=args.alpha, max_iter=args.max_iter, early_stopping=True,
                                random_state=args.seed)
    elif args.algorithm in ("dt", "et"):
        prep = ColumnTransformer([
            ("num", "passthrough", NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ])
        if args.algorithm == "dt":
            clf = DecisionTreeClassifier(max_depth=args.max_depth, min_samples_leaf=20,
                                         class_weight="balanced", random_state=args.seed)
        else:
            clf = ExtraTreesClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                       min_samples_leaf=5, class_weight="balanced",
                                       n_jobs=-1, random_state=args.seed)
    elif args.algorithm == "rf":
        prep = ColumnTransformer([
            ("num", "passthrough", NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ])
        clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                     min_samples_leaf=5, class_weight="balanced",
                                     n_jobs=-1, random_state=args.seed)
    else:
        prep = ColumnTransformer([
            ("num", "passthrough", NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ])
        clf = HistGradientBoostingClassifier(max_iter=args.max_iter, max_depth=args.max_depth,
                                             learning_rate=args.learning_rate,
                                             max_leaf_nodes=args.max_leaf_nodes,
                                             l2_regularization=args.l2,
                                             class_weight="balanced", random_state=args.seed)
    return Pipeline([("prep", prep), ("clf", clf)])


def fit_params(args, y):
    """MLPClassifier has no class_weight option, so the rarer delayed flights get larger sample weights."""
    if args.algorithm == "mlp":
        return {"clf__sample_weight": compute_sample_weight("balanced", y)}
    return {}


def build_metadata(df, args, metrics, NUMERIC):
    """Facts the API needs: allowed values, typical values and training stats for drift checks."""
    holiday = holiday_window_names(df)
    return {
        "algorithm": args.algorithm,
        "features": NUMERIC + CATEGORICAL,
        "allowed": {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL},
        "typical": {
            **{c: int(df[c].median()) for c in NUMERIC},
            **{c: df[c].mode()[0] for c in CATEGORICAL},
        },
        "train_stats": {c: {"mean": float(df[c].mean()), "std": float(df[c].std())} for c in NUMERIC},
        "delay_rate": float(df[TARGET].mean()),
        "delay_rate_by": {
            "airline": df.groupby("airline")[TARGET].mean().round(3).to_dict(),
            "origin": df.groupby("origin")[TARGET].mean().round(3).to_dict(),
            "dep_hour": {str(k): v for k, v in df.groupby("dep_hour")[TARGET].mean().round(3).items()},
            "month": {str(k): v for k, v in df.groupby("month")[TARGET].mean().round(3).items()},
            # delay rate within HOLIDAY_WINDOW days of each holiday (very different per holiday)
            "holiday": df.groupby(holiday)[TARGET].mean().round(3).to_dict(),
            "not_near_holiday": round(float(df.loc[holiday.isna(), TARGET].mean()), 3),
        },
        "metrics": metrics,
        "params": run_params(args),
        "insights": build_insights(df),
    }


def build_insights(df):
    """
    Real statistics from the training data, shown in the dashboard and charts.
    rate      = share of flights that arrived 15+ minutes late
    avg_delay = average minutes late (early arrivals count as 0)
    """
    late = df["arr_delay_min"].clip(lower=0)

    def by(col):
        g = df.assign(late=late).groupby(col)
        out = pd.DataFrame({"rate": g[TARGET].mean(), "avg_delay": g["late"].mean(), "flights": g.size()})
        return {str(k): {"rate": round(float(r.rate), 4), "avg_delay": round(float(r.avg_delay), 1),
                         "flights": int(r.flights)} for k, r in out.iterrows()}

    buckets = pd.cut(df["arr_delay_min"], bins=[-10_000, 14, 30, 60, 120, 100_000],
                     labels=["On time (<15 min)", "15-30 min", "30-60 min", "1-2 hours", "2+ hours"])
    routes = (df.assign(late=late).groupby(["origin", "dest"])
                .agg(volume=(TARGET, "size"), rate=(TARGET, "mean"), distance=("distance", "median"))
                .sort_values("volume", ascending=False).head(40).reset_index())
    return {
        "rows": int(len(df)),
        "avg_delay": round(float(late.mean()), 1),
        "by_hour": by("dep_hour"),
        "by_day": by("day_of_week"),
        "by_month": by("month"),
        "by_days_to_holiday": by("days_to_holiday"),
        "by_airline": by("airline"),
        "by_origin": by("origin"),
        "delay_buckets": {str(k): round(float(v), 4)
                          for k, v in buckets.value_counts(normalize=True, sort=False).items()},
        "top_routes": [{"source": r.origin, "target": r.dest, "volume": int(r.volume),
                        "rate": round(float(r.rate), 4), "distance": int(r.distance)}
                       for r in routes.itertuples()],
        "route_distance": {f"{o}-{d}": int(v) for (o, d), v in
                           df.groupby(["origin", "dest"])["distance"].median().items()},
        # airlines that really flew each route (at least 5 sampled flights), so the API only
        # suggests alternatives that exist on that route
        "route_airlines": {f"{o}-{d}": sorted(g[g >= 5].index.get_level_values(-1).tolist()) for (o, d), g in
                           df.groupby(["origin", "dest"])["airline"].value_counts().groupby(level=[0, 1])},
    }


def main():
    args = parse_args()
    df = add_date_features(pd.read_csv(DATA_PATH))
    NUMERIC = FEATURE_SETS[args.feature_set]
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("flight-delay")

    with mlflow.start_run(run_name=args.algorithm):
        model = build_model(args, NUMERIC)
        model.fit(X_train, y_train, **fit_params(args, y_train))

        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, prob),
        }
        metrics = {k: round(float(v), 4) for k, v in metrics.items()}

        mlflow.log_params(run_params(args))
        mlflow.log_param("rows", len(df))
        mlflow.log_param("n_features", len(NUMERIC) + len(CATEGORICAL))
        mlflow.log_metrics(metrics)
        if args.algorithm == "gb":
            # early stopping (on by default for >10,000 rows) holds back 10% of the training data
            # and stops adding trees when the validation score stops improving
            mlflow.log_metric("trees_built", model[-1].n_iter_)
        if args.algorithm == "mlp":
            mlflow.log_metric("epochs", model[-1].n_iter_)

        if args.cv > 1:
            # k-fold cross-validation on the training set: k models, each tested on a different part,
            # shows whether the score is stable or just luck of one particular split
            cv = cross_validate(build_model(args, NUMERIC), X_train, y_train, scoring=["roc_auc", "f1"],
                                cv=StratifiedKFold(args.cv, shuffle=True, random_state=args.seed),
                                params=fit_params(args, y_train), n_jobs=1)
            for name in ("roc_auc", "f1"):
                scores = cv[f"test_{name}"]
                metrics[f"cv_{name}_mean"] = round(float(scores.mean()), 4)
                metrics[f"cv_{name}_std"] = round(float(scores.std()), 4)
                mlflow.log_metric(f"cv_{name}_mean", metrics[f"cv_{name}_mean"])
                mlflow.log_metric(f"cv_{name}_std", metrics[f"cv_{name}_std"])
            mlflow.log_param("cv_folds", args.cv)
        # MLflow saves models with skops, which only loads types it has been told are safe.
        # We trained this model ourselves, so the tree node types are trusted.
        info = mlflow.sklearn.log_model(
            model, name="model", skops_trusted_types=TRUSTED_TYPES,
            registered_model_name=REGISTERED_MODEL if args.register else None)

        if args.save or args.register:
            os.makedirs("model", exist_ok=True)
            joblib.dump(model, MODEL_PATH)
            with open(META_PATH, "w") as f:
                json.dump(build_metadata(df, args, metrics, NUMERIC), f, indent=2)

    print(json.dumps({"algorithm": args.algorithm, "feature_set": args.feature_set, **metrics}, indent=2))
    if args.save or args.register:
        print(f"Saved model to {MODEL_PATH} (the API now serves this model)")
    else:
        print("Logged to MLflow only. The API model was not changed (use --save or --register for that).")
    if args.register:
        version = info.registered_model_version
        MlflowClient().set_registered_model_alias(REGISTERED_MODEL, "production", version)
        print(f"Registered {REGISTERED_MODEL} version {version} with alias 'production'")


if __name__ == "__main__":
    main()
