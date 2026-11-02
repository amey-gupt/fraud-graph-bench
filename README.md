# fraud-graph-bench

Fraud detection benchmarks on the [Elliptic Bitcoin dataset](https://arxiv.org/abs/1908.02591), loaded via `torch_geometric.datasets.EllipticBitcoinDataset`.

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
python scripts/run_analysis.py   # per-step metrics, feature ablation, validation-chosen threshold
pytest -q                        # split / leakage tests
```

`run_analysis.py` writes `results/analysis.json` and `results/per_step.png`:

1. **Per time step**: illicit count, precision, recall, F1 at 0.5 for each test step 35–49.
2. **Feature ablation**: local-only (93) vs all (165) features. Reports PR-AUC, F1 at 0.5, and false positives at 80% recall.
3. **Threshold chosen without the test set**: train on steps 1–29 and pick the 80%-recall threshold on steps 30–34. Then retrain on 1–34 and apply that threshold to the test set. The baseline's "FP at 80% recall" picks its threshold on the test set itself, so it is optimistic; this version shows what you get in practice.

## Protocol

- **Split by time step, never randomly**: train on steps 1–34, test on 35–49.
- Only labeled nodes (illicit = 1, licit = 0) are used; unknowns are dropped.
- At load time the code checks that the time steps (read from the raw CSV) line up with PyG's node order and that the split matches PyG's own `train_mask`/`test_mask`.
- **XGBoost baseline**: trained on all 165 PyG node features; it does not use the edge list directly. It does use graph information indirectly: 72 of those features are aggregates over each transaction's one-hop neighbors (see below). `scale_pos_weight` = licit/illicit count in training.

### Feature groups

Weber et al. 2019 ([arXiv:1908.02591](https://arxiv.org/abs/1908.02591), Sec. 2.1.2) describe 166 features: "the first 94 features represent local information about the transaction – including the time step", and "the remaining 72 features, called aggregated features, are obtained by aggregating transaction information one-hop backward/forward from the center node". The raw CSV is `[txId, time_step, 165 features]`, and PyG's `x` equals CSV columns 2–166 (checked against the raw file). So in `x`:

- columns 0–92: local (93 features; the paper's 94 minus the time step)
- columns 93–164: aggregated over neighbors (72 features)

These are defined as `LOCAL_COLS` / `AGG_COLS` in `src/fraudbench/data.py`.

## Metrics (test set, illicit = positive)

- Precision, recall, F1 at threshold 0.5
- PR-AUC (average precision), ROC-AUC
- False positives at 80% recall: the highest threshold with recall ≥ 0.80
- Inference latency, mean ms per transaction: batched (full test set / n) and single-row calls

## Layout

```
src/fraudbench/   data.py (loading + split), xgb.py, metrics.py
scripts/          run_baseline.py (baseline), run_analysis.py (follow-up analyses)
tests/            test_split.py
results/          JSON outputs
```
