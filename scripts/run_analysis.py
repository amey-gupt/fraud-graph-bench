"""Follow-up analyses for the XGBoost baseline on Elliptic.

1. Per-time-step test metrics (steps 35..49) + results/per_step.png
2. Feature ablation: local-only vs all features
3. Honest threshold: pick the 80%-recall threshold on validation steps 30-34,
   then apply it to the test set with a model retrained on 1-34.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fraudbench.data import AGG_COLS, LOCAL_COLS, TEST_STEPS, load_elliptic, temporal_split
from fraudbench.metrics import classification_metrics, false_positives_at_recall, threshold_metrics
from fraudbench.xgb import train_xgboost

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def fit_score(data, train_idx, eval_idx, cols=None):
    cols = slice(None) if cols is None else cols
    model, cfg = train_xgboost(data.x[train_idx][:, cols], data.y[train_idx])
    return model.predict_proba(data.x[eval_idx][:, cols])[:, 1], cfg


def per_step(data, split, scores):
    rows = []
    steps = data.time_step[split.test_idx]
    y = data.y[split.test_idx]
    for s in TEST_STEPS:
        m = steps == s
        r = threshold_metrics(y[m], scores[m], 0.5)
        rows.append({"time_step": s, **r})
    return rows


def plot_per_step(rows, path):
    steps = [r["time_step"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    for key, style in [("precision", "o-"), ("recall", "s-"), ("f1", "^-")]:
        vals = [np.nan if r[key] is None else r[key] for r in rows]
        ax1.plot(steps, vals, style, label=key)
    ax1.set_xlabel("time step")
    ax1.set_ylabel("metric @ threshold 0.5")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(steps)
    ax2 = ax1.twinx()
    ax2.bar(steps, [r["n_illicit"] for r in rows], alpha=0.15, color="gray", label="# illicit")
    ax2.set_ylabel("# illicit (bars)")
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left")
    ax1.set_title("XGBoost (all features) per test time step")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summarize(y, scores):
    m = classification_metrics(y, scores)
    fp80 = false_positives_at_recall(y, scores, 0.80)
    return {"pr_auc": m["pr_auc"], "f1_at_0.5": m["f1"], "fp_at_80_recall": fp80}


def fmt(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def main() -> None:
    data = load_elliptic(ROOT / "data")
    split = temporal_split(data.time_step, data.y)  # train 1-34, test 35-49
    y_te = data.y[split.test_idx]

    # 1 + 2: all features (same as the baseline) and local-only
    scores_all, cfg = fit_score(data, split.train_idx, split.test_idx)
    scores_local, _ = fit_score(data, split.train_idx, split.test_idx, LOCAL_COLS)
    steps = per_step(data, split, scores_all)
    plot_per_step(steps, RESULTS / "per_step.png")

    ablation = {
        "all_features": {"n_features": int(data.x.shape[1]), **summarize(y_te, scores_all)},
        "local_only": {"n_features": len(LOCAL_COLS), **summarize(y_te, scores_local)},
    }

    # 3: honest threshold
    val_split = temporal_split(data.time_step, data.y, range(1, 30), range(30, 35))
    scores_val, _ = fit_score(data, val_split.train_idx, val_split.test_idx)
    val_pick = false_positives_at_recall(data.y[val_split.test_idx], scores_val, 0.80)
    thr = val_pick["threshold"]
    test_at_thr = threshold_metrics(y_te, scores_all, thr)  # scores_all = model retrained on 1-34
    honest = {
        "train_steps": [1, 29], "val_steps": [30, 34], "retrain_steps": [1, 34], "test_steps": [35, 49],
        "val_counts": {"illicit": int(np.sum(data.y[val_split.test_idx] == 1)),
                       "licit": int(np.sum(data.y[val_split.test_idx] == 0))},
        "val_selection": val_pick,
        "threshold": thr,
        "test": test_at_thr,
        "oracle_test_fp_at_80_recall": ablation["all_features"]["fp_at_80_recall"],
    }

    results = {
        "params": cfg,
        "feature_groups": {
            "source": "Weber et al. 2019, arXiv:1908.02591v1, Sec. 2.1.2: first 94 of 166 features are "
                      "local (incl. time step), remaining 72 are one-hop aggregated.",
            "pyg_local_cols": [int(LOCAL_COLS[0]), int(LOCAL_COLS[-1])],
            "pyg_aggregated_cols": [int(AGG_COLS[0]), int(AGG_COLS[-1])],
        },
        "per_step": steps,
        "ablation": ablation,
        "honest_threshold": honest,
    }
    (RESULTS / "analysis.json").write_text(json.dumps(results, indent=2))

    print("\nPer test time step (all features, threshold 0.5)")
    print(f"{'step':>4}  {'illicit':>7}  {'precision':>9}  {'recall':>7}  {'f1':>7}")
    for r in steps:
        print(f"{r['time_step']:>4}  {r['n_illicit']:>7}  {fmt(r['precision']):>9}  {fmt(r['recall']):>7}  {fmt(r['f1']):>7}")

    print("\nFeature ablation (test 35-49)")
    print(f"{'features':<14}  {'n':>3}  {'PR-AUC':>7}  {'F1@0.5':>7}  {'FP@80%rec':>9}")
    for name, a in ablation.items():
        print(f"{name:<14}  {a['n_features']:>3}  {a['pr_auc']:.4f}  {a['f1_at_0.5']:.4f}  {a['fp_at_80_recall']['false_positives']:>9}")

    print("\nHonest threshold (picked on val 30-34, applied to test with model retrained on 1-34)")
    print(f"  val: threshold {thr:.6f}, val recall {val_pick['achieved_recall']:.4f}, val FP {val_pick['false_positives']}")
    print(f"  test: recall {fmt(test_at_thr['recall'])}, precision {fmt(test_at_thr['precision'])}, "
          f"FP {test_at_thr['fp']} (of {test_at_thr['n_licit']} licit), TP {test_at_thr['tp']} of {test_at_thr['n_illicit']}")
    print(f"  for reference, test-tuned 80%-recall threshold gave FP "
          f"{honest['oracle_test_fp_at_80_recall']['false_positives']}")
    print(f"\nSaved results/analysis.json and results/per_step.png")


if __name__ == "__main__":
    main()
