"""
Compares every training run recorded in MLflow and saves a chart and a table for the report.

    python compare_models.py

Writes:
    reports/model_comparison.png   bar chart: best ROC-AUC and F1 for each algorithm
    reports/model_comparison.md    table of all runs, best first
"""

import os

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402

NAMES = {
    "logreg": "Logistic Regression",
    "dt": "Decision Tree",
    "rf": "Random Forest",
    "et": "Extra Trees",
    "gb": "Gradient Boosting",
    "mlp": "Neural Network (MLP)",
}
OUT_DIR = "reports"


def describe(run):
    """Short label of the hyperparameters that matter for this algorithm."""
    def p(key):
        value = run.get(f"params.{key}")
        return None if value is None or value != value else value  # NaN (setting not logged) -> None
    algo = p("algorithm")
    if algo == "logreg":
        return f"C={p('C')}"
    if algo == "dt":
        return f"depth {p('max_depth')}"
    if algo in ("rf", "et"):
        return f"{p('n_estimators')} trees, depth {p('max_depth')}"
    if algo == "gb":
        text = f"{p('max_iter')} rounds, lr {p('learning_rate')}"
        if p("max_leaf_nodes"):
            text += f", {p('max_leaf_nodes')} leaves"
        if p("l2") not in (None, "0.0", "0"):
            text += f", L2 {p('l2')}"
        return text
    if algo == "mlp":
        return f"layers {p('hidden')}, alpha {p('alpha')}"
    return ""


def main():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    runs = mlflow.search_runs(experiment_names=["flight-delay"], filter_string="attributes.status = 'FINISHED'")
    runs = runs.dropna(subset=["metrics.roc_auc"]).sort_values("metrics.roc_auc", ascending=False)
    if runs.empty:
        print("No finished runs found. Train some models first with train.py.")
        return

    rows = []
    for _, run in runs.iterrows():
        rows.append({
            "algorithm": NAMES.get(run["params.algorithm"], run["params.algorithm"]),
            "features": run.get("params.feature_set") or "base",
            "settings": describe(run),
            "accuracy": run["metrics.accuracy"],
            "precision": run["metrics.precision"],
            "recall": run["metrics.recall"],
            "f1": run["metrics.f1"],
            "roc_auc": run["metrics.roc_auc"],
            "cv": (f"{run['metrics.cv_roc_auc_mean']:.3f} ± {run['metrics.cv_roc_auc_std']:.3f}"
                   if "metrics.cv_roc_auc_mean" in run and run["metrics.cv_roc_auc_mean"] == run["metrics.cv_roc_auc_mean"]
                   else ""),
        })

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- table of every run ----
    lines = ["| # | Algorithm | Features | Settings | Accuracy | Precision | Recall | F1 | ROC-AUC | 5-fold CV ROC-AUC |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {r['algorithm']} | {r['features']} | {r['settings']} | {r['accuracy']:.3f} | "
                     f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['roc_auc']:.3f} | {r['cv']} |")
    with open(os.path.join(OUT_DIR, "model_comparison.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- chart: best run of each algorithm (date features where available) ----
    best = {}
    for r in rows:  # rows are sorted best first, so the first one seen per algorithm is its best
        best.setdefault(r["algorithm"], r)
    order = sorted(best.values(), key=lambda r: r["roc_auc"])
    labels = [r["algorithm"] for r in order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 0.6 * len(order) + 1.8), sharey=True)
    for ax, metric, title in [(axes[0], "roc_auc", "ROC-AUC (ranking quality)"), (axes[1], "f1", "F1 score (delayed class)")]:
        values = [r[metric] for r in order]
        colors = ["#0f766e" if r is order[-1] else "#94a3b8" for r in order]
        bars = ax.barh(labels, values, color=colors)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=4, fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(min(values) - 0.05, max(values) + 0.03)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=10)
    fig.suptitle(f"Best run of each algorithm ({len(rows)} MLflow runs, 120,000 real 2024 flights)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "model_comparison.png")
    fig.savefig(path, dpi=160)
    print(f"\nSaved {path} and {os.path.join(OUT_DIR, 'model_comparison.md')}")


if __name__ == "__main__":
    main()
