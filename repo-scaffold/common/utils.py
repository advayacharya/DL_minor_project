"""
Shared evaluation + result-saving helpers used by every model script.
Every model calls evaluate_and_save() at the end of training so results
land in the same format for evaluate/compare.py to pick up.
"""

import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "results")


def evaluate_and_save(model_name, model, test_ds, train_seconds, param_count):
    """Runs the model on the test set, computes accuracy/precision/recall/F1,
    and writes them to reports/results/<model_name>.json."""
    y_true = []
    y_pred = []
    for features, labels in test_ds:
        preds = model.predict(features, verbose=0)
        preds = (preds.ravel() > 0.5).astype(int)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "train_seconds": round(train_seconds, 1),
        "param_count": int(param_count),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{model_name}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved results to {out_path}")
    print(json.dumps(metrics, indent=2))
    return metrics
