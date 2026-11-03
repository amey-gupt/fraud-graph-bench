"""Fit TabICL on train steps 1-34 and score test steps 35-49. No downsampling."""

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from fraudbench.data import TEST_STEPS, load_elliptic, temporal_split
from fraudbench.metrics import classification_metrics, false_positives_at_recall, threshold_metrics

ROOT = Path(__file__).resolve().parent.parent
T0 = time.perf_counter()
DEVICE = os.environ.get("DEVICE", "cpu")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))  # ensemble members per forward pass
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "2000"))  # test rows per predict_proba call


def log(msg: str) -> None:
    # Timestamped and flushed, so a killed run (e.g. out of memory) still shows how far it got.
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


def main() -> None:
    import tabicl
    from tabicl import TabICLClassifier

    data = load_elliptic(ROOT / "data")
    split = temporal_split(data.time_step, data.y)
    train_steps = set(np.unique(data.time_step[split.train_idx]).tolist())
    test_steps = set(np.unique(data.time_step[split.test_idx]).tolist())
    assert not train_steps & test_steps, "time-step leakage between train and test"

    x_tr, y_tr = data.x[split.train_idx], data.y[split.train_idx]
    x_te, y_te = data.x[split.test_idx], data.y[split.test_idx]
    log(f"train {x_tr.shape} ({int(y_tr.sum())} illicit), test {x_te.shape} ({int(y_te.sum())} illicit)")

    # BATCH_SIZE defaults to 1: run the 8 ensemble members one at a time. TabICL's default
    # (8 at once) needed 11-16 GB for one pass on this data, more than a 7 GB CPU machine.
    clf = TabICLClassifier(random_state=42, device=DEVICE, batch_size=BATCH_SIZE, verbose=True)
    t = time.perf_counter()
    clf.fit(x_tr, y_tr)
    fit_s = time.perf_counter() - t
    log(f"fit done in {fit_s:.1f}s")

    # Score the test set in row chunks to bound memory. Every test row is still scored
    # against the full training set, and test rows don't attend to each other
    # (TabICL's embed_with_test defaults to False), so chunking doesn't change scores.
    chunks = []
    t = time.perf_counter()
    for start in range(0, len(x_te), CHUNK_SIZE):
        tc = time.perf_counter()
        chunks.append(clf.predict_proba(x_te[start : start + CHUNK_SIZE]))
        log(f"chunk rows {start}-{min(start + CHUNK_SIZE, len(x_te)) - 1} in {time.perf_counter() - tc:.1f}s")
    infer_s = time.perf_counter() - t  # total wall time across all chunks
    proba = np.concatenate(chunks)
    log(f"predict_proba done in {infer_s:.1f}s")

    illicit_col = list(clf.classes_).index(1)
    scores = proba[:, illicit_col]

    metrics = classification_metrics(y_te, scores)
    fp80 = false_positives_at_recall(y_te, scores, 0.80)
    steps = data.time_step[split.test_idx]
    per_step = []
    for s in TEST_STEPS:
        m = steps == s
        r = threshold_metrics(y_te[m], scores[m], 0.5)
        per_step.append({"time_step": s, "n_illicit": r["n_illicit"], "recall": r["recall"],
                         "tp": r["tp"], "fn": r["fn"]})

    results = {
        "model": "tabicl",
        "tabicl_version": tabicl.__version__ if hasattr(tabicl, "__version__") else None,
        "params": {k: str(v) for k, v in clf.get_params().items()},
        "features": "all 165 node features, no scaling (TabICL normalizes internally)",
        "train_rows": int(len(y_tr)),
        "test_rows": int(len(y_te)),
        "fit_seconds": fit_s,
        "inference_seconds_total": infer_s,
        "inference_chunk_size": CHUNK_SIZE,
        "inference_ms_per_tx": infer_s * 1000 / len(y_te),
        "test_metrics": metrics,
        "false_positives_at_80_recall": fp80,
        "per_step_recall": per_step,
        "environment": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "device": DEVICE,
            "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "batch_size": BATCH_SIZE,
        },
    }
    out = ROOT / "results" / "tabicl.json"
    out.write_text(json.dumps(results, indent=2))

    print("\nTabICL - Elliptic test set (illicit = positive)")
    for k in ["precision", "recall", "f1", "pr_auc", "roc_auc"]:
        print(f"  {k:<10} {metrics[k]:.4f}")
    print(f"  FP @80% recall  {fp80['false_positives']} (thr {fp80['threshold']:.4f}, recall {fp80['achieved_recall']:.4f})")
    print(f"  fit {fit_s:.1f}s, inference total {infer_s:.1f}s ({infer_s * 1000 / len(y_te):.3f} ms/tx)")
    print("\n  step  illicit  recall")
    for r in per_step:
        rec = "n/a" if r["recall"] is None else f"{r['recall']:.4f}"
        print(f"  {r['time_step']:>4}  {r['n_illicit']:>7}  {rec:>6}")
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
