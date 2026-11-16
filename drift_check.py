"""
Data drift check (bonus stage).

Compares the inputs the live API has received (logs/predictions.jsonl, written by app.py
on every /predict call) with the training data statistics saved in model/metadata.json.

Numeric features: how far the live mean has moved, measured in training standard deviations.
Categorical features: share of live requests whose airline/airport was rare in training.

Usage:
    python drift_check.py                      # uses logs/predictions.jsonl
    python drift_check.py --log path/to/log.jsonl --threshold 0.5
Exit code 1 means drift was detected (so it can be used in a scheduled CI job).
"""

import argparse
import json
import os
import sys

import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Check live inputs for drift against training data")
    p.add_argument("--log", default=os.path.join("logs", "predictions.jsonl"))
    p.add_argument("--meta", default=os.path.join("model", "metadata.json"))
    p.add_argument("--threshold", type=float, default=0.5,
                   help="flag a numeric feature when its mean moves more than this many std devs")
    p.add_argument("--min_requests", type=int, default=20)
    args = p.parse_args()

    with open(args.meta) as f:
        meta = json.load(f)
    if not os.path.exists(args.log):
        print(f"No prediction log found at {args.log}. Make some predictions first.")
        return 0

    live = pd.read_json(args.log, lines=True)
    print(f"Live requests: {len(live)}   (training rows: {meta['insights']['rows']})\n")
    if len(live) < args.min_requests:
        print(f"Fewer than {args.min_requests} requests: too few to judge drift reliably.")

    drifted = []
    print(f"{'feature':<14}{'train mean':>12}{'live mean':>12}{'shift (std)':>13}  status")
    for col, stats in meta["train_stats"].items():
        if col not in live:
            continue
        shift = abs(live[col].mean() - stats["mean"]) / (stats["std"] or 1)
        status = "DRIFT" if shift > args.threshold else "ok"
        if status == "DRIFT":
            drifted.append(col)
        print(f"{col:<14}{stats['mean']:>12.2f}{live[col].mean():>12.2f}{shift:>13.2f}  {status}")

    # flights far in the future: the model learned 2024 patterns, airline schedules change every year
    if "flight_date" in live:
        years = pd.to_datetime(live["flight_date"]).dt.year
        later = (years > 2025).mean()
        print(f"\nRequests for flights after 2025 (model trained on 2024): {later:.0%}")
        if later > 0.5:
            drifted.append("flight year (retrain on newer data)")

    for col, key in [("airline", "by_airline"), ("origin", "by_origin")]:
        flights = {k: v["flights"] for k, v in meta["insights"][key].items()}
        total = sum(flights.values())
        rare = {k for k, n in flights.items() if n / total < 0.01}
        share = live[col].isin(rare).mean()
        print(f"Requests with a rare {col} (<1% of training): {share:.0%}")

    print()
    if drifted:
        print("DRIFT DETECTED in: " + ", ".join(drifted))
        print("Action: download newer BTS months with download_data.py and retrain with train.py.")
        return 1
    print("No significant drift detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
