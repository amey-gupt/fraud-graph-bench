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
pytest -q                        # split / leakage tests
```

## Protocol

- **Split by time step, never randomly**: train on steps 1–34, test on 35–49.
- Only labeled nodes (illicit = 1, licit = 0) are used; unknowns are dropped.
- At load time the code checks that the time steps (read from the raw CSV) line up with PyG's node order and that the split matches PyG's own `train_mask`/`test_mask`.
- **XGBoost baseline**: the 165 node features only (no graph edges). `scale_pos_weight` = licit/illicit count in training.

## Metrics (test set, illicit = positive)

- Precision, recall, F1 at threshold 0.5
- PR-AUC (average precision), ROC-AUC
- False positives at 80% recall: the highest threshold with recall ≥ 0.80
- Inference latency, mean ms per transaction: batched (full test set / n) and single-row calls

## Layout

```
src/fraudbench/   data.py (loading + split), xgb.py, metrics.py
scripts/          run_baseline.py (entry point)
tests/            test_split.py
results/          JSON outputs
```
