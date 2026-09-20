"""Temporal train/validation/test split.

Global timestamp quantile cutoffs:
  ts < T1            -> train   (fit CF + content retrieval profiles)
  T1 <= ts < T2      -> val     (ranker labels / early stopping)
  ts >= T2           -> test    (final honest evaluation)

No leakage: every split boundary is a hard time cutoff, so models never
see future interactions when predicting the val/test windows.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .config import CFG, Config


def temporal_cutoffs(timestamps: np.ndarray, train_frac: float, val_frac: float):
    t1 = float(np.quantile(timestamps, train_frac))
    t2 = float(np.quantile(timestamps, val_frac))
    assert t1 < t2, f"degenerate cutoffs t1={t1} t2={t2}"
    return t1, t2


def temporal_split(ratings: pl.DataFrame, cfg: Config = CFG):
    """Returns (train, val, test, t1, t2). Ratings must have `timestamp`."""
    ts = ratings["timestamp"].to_numpy()
    t1, t2 = temporal_cutoffs(ts, cfg.train_frac, cfg.val_frac)
    train = ratings.filter(pl.col("timestamp") < t1)
    val = ratings.filter((pl.col("timestamp") >= t1) & (pl.col("timestamp") < t2))
    test = ratings.filter(pl.col("timestamp") >= t2)
    return train, val, test, t1, t2


def user_histories(df: pl.DataFrame) -> dict[int, np.ndarray]:
    """user_idx -> sorted array of item_idx (by timestamp)."""
    out: dict[int, np.ndarray] = {}
    for u, grp in df.sort("timestamp").group_by("userId", maintain_order=True):
        out[int(u[0])] = grp["item_idx"].to_numpy()
    return out


def positive_items(df: pl.DataFrame, threshold: float) -> dict[int, set[int]]:
    """user_idx -> set of item_idx with rating >= threshold."""
    pos = df.filter(pl.col("rating") >= threshold)
    return {int(u[0]): set(grp["item_idx"].to_list())
            for u, grp in pos.group_by("userId", maintain_order=True)}
