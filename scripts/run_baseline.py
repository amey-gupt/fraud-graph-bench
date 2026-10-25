"""Train and evaluate the XGBoost baseline on Elliptic. Single entry point."""

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from fraudbench.data import load_elliptic, temporal_split
from fraudbench.metrics import classification_metrics, false_positives_at_recall, inference_latency
from fraudbench.xgb import train_xgboost

ROOT = Path(__file__).resolve().parent.parent


def class_counts(y: np.ndarray) -> dict:
    return {"illicit": int(np.sum(y == 1)), "licit": int(np.sum(y == 0))}


def main() -> None:
    data = load_elliptic(ROOT / "data")
    split = temporal_split(data.time_step, data.y)

    train_steps = set(np.unique(data.time_step[split.train_idx]).tolist())
    test_steps = set(np.unique(data.time_step[split.test_idx]).tolist())
    assert not train_steps & test_steps, "time-step leakage between train and test"

    x_tr, y_tr = data.x[split.train_idx], data.y[split.train_idx]
    x_te, y_te = data.x[split.test_idx], data.y[split.test_idx]

    t0 = time.perf_counter()
    model, cfg = train_xgboost(x_tr, y_tr)
    train_s = time.perf_counter() - t0

    scores = model.predict_proba(x_te)[:, 1]
    metrics = classification_metrics(y_te, scores)
    fp80 = false_positives_at_recall(y_te, scores, 0.80)
    latency = inference_latency(model.predict_proba, x_te)

    import sklearn, torch, torch_geometric, xgboost
    results = {
        "model": "xgboost",
        "features": "node features only (165)",
        "split": {
            "train_time_steps": [min(train_steps), max(train_steps)],
            "test_time_steps": [min(test_steps), max(test_steps)],
            "train_counts": class_counts(y_tr),
            "test_counts": class_counts(y_te),
        },
        "params": cfg,
        "train_seconds": train_s,
        "test_metrics": metrics,
        "false_positives_at_80_recall": fp80,
        "latency": latency,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "xgboost": xgboost.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
        },
    }

    out = ROOT / "results" / "xgboost.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    tc, sc = results["split"]["train_counts"], results["split"]["test_counts"]
    rows = [
        ("Train (steps 1-34)", f"{tc['illicit']} illicit / {tc['licit']} licit"),
        ("Test (steps 35-49)", f"{sc['illicit']} illicit / {sc['licit']} licit"),
        ("scale_pos_weight", f"{cfg['scale_pos_weight']:.4f}"),
        ("Precision @0.5", f"{metrics['precision']:.4f}"),
        ("Recall @0.5", f"{metrics['recall']:.4f}"),
        ("F1 @0.5", f"{metrics['f1']:.4f}"),
        ("PR-AUC", f"{metrics['pr_auc']:.4f}"),
        ("ROC-AUC", f"{metrics['roc_auc']:.4f}"),
        ("FP @80% recall", f"{fp80['false_positives']} (thr {fp80['threshold']:.4f}, recall {fp80['achieved_recall']:.4f})"),
        ("Latency ms/tx (batched)", f"{latency['mean_ms_per_tx_batched']:.6f}"),
        ("Latency ms/tx (single)", f"{latency['mean_ms_per_tx_single']:.4f}"),
    ]
    w = max(len(k) for k, _ in rows)
    print("\nXGBoost baseline - Elliptic test set (illicit = positive)")
    print("-" * (w + 40))
    for k, v in rows:
        print(f"{k:<{w}}  {v}")
    print("-" * (w + 40))
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
