"""Train and evaluate a multi-layer Perceptron on the Elliptic Bitcoin Dataset."""

import torch
from fraudbench.data import load_elliptic, temporal_split
from pathlib import Path
import numpy as np

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from fraudbench.metrics import classification_metrics


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    torch.manual_seed(42)  # same weights and batch order on every run

    data = load_elliptic(ROOT / "data")
    split = temporal_split(data.time_step, data.y)

    train_steps = set(np.unique(data.time_step[split.train_idx]).tolist())
    test_steps = set(np.unique(data.time_step[split.test_idx]).tolist())
    assert not train_steps & test_steps, "time-step leakage between train and test"

    x_tr, y_tr = data.x[split.train_idx], data.y[split.train_idx]
    x_te, y_te = data.x[split.test_idx], data.y[split.test_idx]

    # 1. STANDARDIZE (fit on train only)
    # The 1e-8 avoids dividing by zero for a feature that is constant in training.
    # Stats come from train only so no test information leaks into the model.
    mean = x_tr.mean(axis=0)
    std = x_tr.std(axis=0) + 1e-8
    x_tr = (x_tr - mean) / std
    x_te = (x_te - mean) / std

    # 2. TENSORS + BATCHES
    x_tr_tensor = torch.tensor(x_tr, dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32)  # BCE loss needs float labels
    x_te_tensor = torch.tensor(x_te, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_tr_tensor, y_tr_tensor), batch_size=256, shuffle=True)

    # 3. MODEL, LOSS, OPTIMIZER
    model = nn.Sequential(nn.Linear(165, 64), nn.ReLU(), nn.Linear(64, 1))
    pos_weight = torch.tensor(np.sum(y_tr == 0) / np.sum(y_tr == 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 4. TRAINING LOOP
    for epoch in range(20):
        model.train()
        for xb, yb in loader:
            logits = model(xb).squeeze(1)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch + 1:2d}  last batch loss {loss.item():.4f}")

    # 5. EVALUATE
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(x_te_tensor)).squeeze(1).numpy()
    print(classification_metrics(y_te, scores))


if __name__ == "__main__":
    main()
