import numpy as np
import polars as pl
import pytest

from src.config import Config
from src.split import temporal_split, positive_items


def _ratings(n=1000):
    rng = np.random.default_rng(0)
    return pl.DataFrame({
        "userId": rng.integers(0, 50, n),
        "item_idx": rng.integers(0, 100, n),
        "rating": rng.uniform(0.5, 5.0, n).round(1),
        "timestamp": np.sort(rng.integers(1_000_000, 2_000_000, n)),
    })


def test_temporal_boundaries_no_leakage():
    cfg = Config(train_frac=0.8, val_frac=0.9)
    df = _ratings(5000)
    tr, va, te, t1, t2 = temporal_split(df, cfg)
    assert tr["timestamp"].max() < t1
    assert va["timestamp"].min() >= t1 and va["timestamp"].max() < t2
    assert te["timestamp"].min() >= t2
    assert tr.height + va.height + te.height == df.height


def test_positive_items_threshold():
    df = pl.DataFrame({
        "userId": [1, 1, 2],
        "item_idx": [10, 11, 12],
        "rating": [4.5, 2.0, 5.0],
        "timestamp": [1, 2, 3],
    })
    pos = positive_items(df, 3.5)
    assert pos[1] == {10}
    assert pos[2] == {12}
