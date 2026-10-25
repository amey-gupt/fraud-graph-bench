from pathlib import Path

import numpy as np
import pytest

from fraudbench.data import ILLICIT, LICIT, UNKNOWN, TEST_STEPS, temporal_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def check_no_leakage(time_step, y, split):
    train_steps = set(time_step[split.train_idx].tolist())
    test_steps = set(time_step[split.test_idx].tolist())
    assert train_steps.isdisjoint(test_steps), "a test time step appears in training"
    assert train_steps <= set(range(1, 35))
    assert test_steps <= set(TEST_STEPS)
    assert set(split.train_idx).isdisjoint(split.test_idx)
    assert not np.any(y[split.train_idx] == UNKNOWN)
    assert not np.any(y[split.test_idx] == UNKNOWN)


def test_split_synthetic():
    rng = np.random.default_rng(0)
    time_step = rng.integers(1, 50, size=5000)
    y = rng.choice([LICIT, ILLICIT, UNKNOWN], size=5000)
    split = temporal_split(time_step, y)
    check_no_leakage(time_step, y, split)
    # every labeled node lands on exactly one side
    assert len(split.train_idx) + len(split.test_idx) == int(np.sum(y != UNKNOWN))


@pytest.mark.skipif(not (DATA_DIR / "processed" / "data.pt").exists(),
                    reason="Elliptic not downloaded yet; run scripts/run_baseline.py first")
def test_split_real_data():
    from fraudbench.data import load_elliptic

    data = load_elliptic(DATA_DIR)
    split = temporal_split(data.time_step, data.y)
    check_no_leakage(data.time_step, data.y, split)
    assert set(data.time_step[split.train_idx].tolist()) == set(range(1, 35))
    assert set(data.time_step[split.test_idx].tolist()) == set(TEST_STEPS)
