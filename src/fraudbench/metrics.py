"""Evaluation metrics (illicit = positive class) and inference latency."""

import time
from typing import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }


def false_positives_at_recall(y_true: np.ndarray, scores: np.ndarray, target_recall: float = 0.80) -> dict:
    """FPs at the highest threshold whose recall is at least `target_recall`."""
    _, recall, thresholds = precision_recall_curve(y_true, scores)
    # recall has one more entry than thresholds; drop the final (recall=0) point.
    recall = recall[:-1]
    ok = np.flatnonzero(recall >= target_recall)
    # thresholds are increasing, so the last qualifying index is the highest threshold.
    thr = float(thresholds[ok[-1]])
    y_pred = scores >= thr
    fp = int(np.sum(y_pred & (y_true == 0)))
    tp = int(np.sum(y_pred & (y_true == 1)))
    return {
        "target_recall": target_recall,
        "threshold": thr,
        "achieved_recall": tp / int(np.sum(y_true == 1)),
        "false_positives": fp,
        "true_positives": tp,
    }


def inference_latency(predict: Callable[[np.ndarray], np.ndarray], x: np.ndarray,
                      batch_repeats: int = 10, single_calls: int = 1000, seed: int = 0) -> dict:
    """Mean ms per transaction, both batched (full test set) and one row per call."""
    predict(x)  # warm-up
    batch_times = []
    for _ in range(batch_repeats):
        t0 = time.perf_counter()
        predict(x)
        batch_times.append(time.perf_counter() - t0)
    batch_ms_per_tx = float(np.mean(batch_times)) * 1000 / len(x)

    rng = np.random.default_rng(seed)
    rows = rng.integers(0, len(x), size=single_calls)
    predict(x[rows[:1]])  # warm-up
    t0 = time.perf_counter()
    for r in rows:
        predict(x[r : r + 1])
    single_ms = (time.perf_counter() - t0) * 1000 / single_calls

    return {
        "mean_ms_per_tx_batched": batch_ms_per_tx,
        "mean_ms_per_tx_single": single_ms,
        "method": (
            f"predict_proba via time.perf_counter. Batched: full test set, 1 warm-up + "
            f"{batch_repeats} timed runs, mean total time / n_test. Single: {single_calls} "
            f"one-row calls on random test rows, mean per call."
        ),
    }
