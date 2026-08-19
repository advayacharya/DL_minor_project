"""
Aggregates each model's saved results (reports/results/*.json) into one
comparison table. Run this only after all 4 models have been trained at
least once.
Run from repo root: python -m evaluate.compare
"""

import glob
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "results")


def main():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
    if not files:
        print("No results found. Run each model script first (see each model's main()).")
        return

    rows = [json.load(open(f)) for f in files]

    header = (
        f"{'Model':<10}{'Accuracy':>10}{'Precision':>11}"
        f"{'Recall':>9}{'F1':>8}{'Params':>12}{'Train(s)':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['model']:<10}{r['accuracy']:>10.3f}{r['precision']:>11.3f}"
            f"{r['recall']:>9.3f}{r['f1']:>8.3f}{r['param_count']:>12,}"
            f"{r['train_seconds']:>10.1f}"
        )


if __name__ == "__main__":
    main()
