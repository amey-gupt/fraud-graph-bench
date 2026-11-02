"""Load the Elliptic dataset and build a strict temporal train/test split."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LICIT, ILLICIT, UNKNOWN = 0, 1, 2  # PyG label encoding
TRAIN_STEPS = range(1, 35)  # time steps 1..34
TEST_STEPS = range(35, 50)  # time steps 35..49

# Feature groups, from Weber et al. 2019 ("Anti-Money Laundering in Bitcoin...",
# arXiv:1908.02591v1, Sec. 2.1.2): of 166 features, "the first 94 features represent
# local information about the transaction - including the time step"; "the remaining
# 72 features, called aggregated features, are obtained by aggregating transaction
# information one-hop backward/forward from the center node".
# The raw CSV is [txId, time_step, 165 features]; PyG's x is CSV columns 2..166
# (checked numerically against the raw file). So in x, the 93 local features
# (paper's 94 minus the time step) are columns 0..92 and the 72 aggregated are 93..164.
LOCAL_COLS = np.arange(0, 93)
AGG_COLS = np.arange(93, 165)


@dataclass
class EllipticData:
    x: np.ndarray  # (num_nodes, 165) node features
    y: np.ndarray  # (num_nodes,) 0=licit, 1=illicit, 2=unknown
    time_step: np.ndarray  # (num_nodes,) 1..49
    pyg_train_mask: np.ndarray
    pyg_test_mask: np.ndarray


@dataclass
class Split:
    train_idx: np.ndarray
    test_idx: np.ndarray


def load_elliptic(root: str | Path = "data") -> EllipticData:
    """Load via PyG and recover per-node time steps from the raw CSV.

    PyG drops the time-step column from `x`, so we read it back from the raw
    features file. PyG keeps the CSV row order as node order, and takes labels
    from the classes CSV by row position, so we check both files line up.
    """
    from torch_geometric.datasets import EllipticBitcoinDataset

    ds = EllipticBitcoinDataset(root=str(root))
    data = ds[0]

    raw_dir = Path(ds.raw_dir)
    feat = pd.read_csv(raw_dir / "elliptic_txs_features.csv", header=None, usecols=[0, 1])
    classes = pd.read_csv(raw_dir / "elliptic_txs_classes.csv")

    if data.x.shape[1] != len(LOCAL_COLS) + len(AGG_COLS):
        raise ValueError(f"expected 165 feature columns, got {data.x.shape[1]}")
    if len(feat) != data.num_nodes:
        raise ValueError(f"features CSV has {len(feat)} rows, PyG data has {data.num_nodes} nodes")
    if not np.array_equal(feat[0].values, classes["txId"].values):
        raise ValueError("txId order differs between features and classes CSVs; labels would be misaligned")

    time_step = feat[1].to_numpy()
    y = data.y.numpy()

    out = EllipticData(
        x=data.x.numpy(),
        y=y,
        time_step=time_step,
        pyg_train_mask=data.train_mask.numpy(),
        pyg_test_mask=data.test_mask.numpy(),
    )

    # Our split must match PyG's own masks; this also proves time-step alignment.
    split = temporal_split(time_step, y)
    ours_train = np.zeros_like(out.pyg_train_mask)
    ours_train[split.train_idx] = True
    ours_test = np.zeros_like(out.pyg_test_mask)
    ours_test[split.test_idx] = True
    if not (np.array_equal(ours_train, out.pyg_train_mask) and np.array_equal(ours_test, out.pyg_test_mask)):
        raise ValueError("temporal split does not match PyG's train_mask/test_mask")

    return out


def temporal_split(time_step: np.ndarray, y: np.ndarray,
                   train_steps=TRAIN_STEPS, test_steps=TEST_STEPS) -> Split:
    """Split by time step (default: train 1-34, test 35-49), labeled nodes only. Never random."""
    time_step = np.asarray(time_step)
    y = np.asarray(y)
    if set(train_steps) & set(test_steps):
        raise ValueError("train and test time steps overlap")
    labeled = y != UNKNOWN
    train = labeled & np.isin(time_step, list(train_steps))
    test = labeled & np.isin(time_step, list(test_steps))
    return Split(train_idx=np.flatnonzero(train), test_idx=np.flatnonzero(test))
