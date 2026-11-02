# fraud-graph-bench

Fraud detection benchmarks on the [Elliptic Bitcoin dataset](https://arxiv.org/abs/1908.02591) (203,769 transactions, 4,545 labeled illicit, 42,019 licit), with a strict time-based split: train on time steps 1–34, test on 35–49.

The baseline is XGBoost on the 165 node features. On the test set it reaches F1 0.794 and PR-AUC 0.800 (illicit = positive). Those single numbers hide most of what matters. The three findings below come from `scripts/run_analysis.py`. Every number is copied from `results/xgboost.json` or `results/analysis.json`.

## Findings

### 1. The model fails after step 43, and the aggregate score hides it

![Per-step precision, recall and F1](results/per_step.png)

| Test steps | Illicit | Caught | False positives | Recall | F1 |
|---|---|---|---|---|---|
| 35–42 | 914 | 790 | 54 | 0.864 | 0.899 |
| 43–49 | 169 | 4 | 68 | 0.024 | 0.033 |

Recall is exactly 0 at steps 43, 45, 47 and 48. The headline F1 of 0.79 comes from steps 35–42. A single aggregate test number hid a near-total failure after step 43.

Weber et al. 2019 (Sec. 4, "The Dark Market Shutdown") report a dark market closing at time step 43 and say this "causes all methods to perform poorly after the shutdown". Even a Random Forest retrained after every test step "is not able to reliably capture new illicit transactions". Our model shows the same thing:

- The median score it gives illicit transactions is 0.9997 in steps 35–42 and 0.0006 in steps 43–49. It treats the later fraud as licit.
- The number of illicit transactions also drops, from 914 in steps 35–42 to 169 in steps 43–49.

One likely explanation is that the model learned the patterns of the actors active before the shutdown, and the illicit activity afterwards looks different. Fraud patterns change over time, which is why performance should be monitored per time period and not only as one test-set number.

### 2. Neighbor features help: 13% fewer alerts at 80% recall

Weber et al. 2019 (Sec. 2.1.2) split the features into 94 local ones (including the time step) and 72 aggregated over each transaction's one-hop neighbors. PyG drops the time step, which leaves 93 local and 72 aggregated (see [Feature groups](#feature-groups)).

| Features | Count | PR-AUC | F1 @ 0.5 | False positives at 80% recall |
|---|---|---|---|---|
| Local only | 93 | 0.785 | 0.739 | 2,226 |
| Local + neighbor aggregates | 165 | 0.800 | 0.794 | 1,945 |

PR-AUC barely moves. Where it matters, the gain is clear: F1 rises from 0.739 to 0.794, and false positives at 80% recall drop from 2,226 to 1,945. That is 12.6% fewer alerts for analysts to review.

These 72 features are hand-made one-hop summaries. They say nothing about what a model that learns across several hops could do.

**Takeaway:** simple neighbor aggregates already cut false positives by 13% at 80% recall. Whether learned multi-hop models do better is the next experiment.

### 3. A threshold chosen without the test set: 36 false positives, not 1,945

The baseline's "1,945 false positives at 80% recall" uses a threshold picked with the test labels. That is leakage, because it can't be done in deployment. Instead:

1. Train on steps 1–29.
2. Pick the threshold that gives 80% recall on steps 30–34 (validation).
3. Retrain on steps 1–34 and apply that threshold to the test set.

| Threshold chosen on | Threshold | Test recall | Test false positives | Illicit caught |
|---|---|---|---|---|
| Validation (steps 30–34) | 0.808 | 0.724 | 36 | 784 / 1,083 |
| Test set (leakage) | 0.006 | 0.801 | 1,945 | 867 / 1,083 |

On validation, threshold 0.808 gave 80.0% recall with 7 false positives. On the test set it gives 72% recall with 36 false positives. The target isn't met because of the same drift: the validation period comes before the shutdown, and 98% of illicit transactions after step 43 score below 0.5.

Reaching 80% recall on the test set requires dropping the threshold to 0.006. That catches 83 more illicit transactions (54 from steps 35–42, 29 from 43–49), but false positives rise from 36 to 1,945.

**The number to report** is the validation-chosen one: 72% recall at 36 false positives. Report it together with the warning from finding 1 that performance collapsed after the shutdown.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

## Run

```bash
python scripts/run_baseline.py   # downloads data/ on first run, writes results/xgboost.json
python scripts/run_analysis.py   # writes results/analysis.json and results/per_step.png
pytest -q                        # split / leakage tests
```

## Protocol

- **Split by time step, never randomly**: train on 1–34, test on 35–49. For the threshold analysis: train on 1–29, validate on 30–34.
- Only labeled nodes are used (illicit = 1, licit = 0); unknowns are dropped.
- At load time the code checks that time steps from the raw CSV line up with PyG's node order, and that the split matches PyG's own `train_mask`/`test_mask`.
- **XGBoost**: fixed settings (300 trees, depth 6, learning rate 0.1, seed 42), no tuning. `scale_pos_weight` = licit/illicit count in the training data.
- **Graph information**: the model doesn't use the edge list. It does use graph information indirectly, because 72 of the 165 features are aggregates over one-hop neighbors.

### Feature groups

Weber et al. 2019 ([arXiv:1908.02591](https://arxiv.org/abs/1908.02591), Sec. 2.1.2): "the first 94 features represent local information about the transaction – including the time step", and "the remaining 72 features, called aggregated features, are obtained by aggregating transaction information one-hop backward/forward from the center node". The raw CSV is `[txId, time_step, 165 features]`, and PyG's `x` equals CSV columns 2–166 (checked against the raw file). So in `x`:

- columns 0–92: local (93 features; the paper's 94 minus the time step)
- columns 93–164: aggregated over neighbors (72 features)

These are defined as `LOCAL_COLS` / `AGG_COLS` in `src/fraudbench/data.py`.

## Metrics (illicit = positive)

- Precision, recall, F1 at threshold 0.5
- PR-AUC (average precision), ROC-AUC
- False positives at 80% recall, using the highest threshold with recall ≥ 0.80. The baseline picks this threshold on the test set, so it is optimistic; see finding 3.
- Inference latency in mean ms per transaction, measured both batched (full test set / n) and one row per call

## Layout

```
src/fraudbench/   data.py (loading, split, feature groups), xgb.py, metrics.py
scripts/          run_baseline.py (baseline), run_analysis.py (findings above)
tests/            test_split.py
results/          JSON outputs and per_step.png
```
