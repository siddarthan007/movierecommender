"""Extended evaluation metrics: novelty, popularity bias, personalization,
long-tail coverage, Gini concentration, MRR, segment buckets, bootstrap CIs.
"""
from __future__ import annotations

import numpy as np


def mrr_at_k(recs, truth, k):
    for r, i in enumerate(recs[:k], 1):
        if i in truth:
            return 1.0 / r
    return 0.0


def novelty(recs, item_pop: np.ndarray) -> float:
    """Mean -log2(popularity share) of recommended items."""
    pop = np.asarray(item_pop, dtype=np.float64)
    share = pop / pop.sum()
    vals = [-np.log2(share[i]) for i in recs if share[i] > 0]
    return float(np.mean(vals)) if vals else 0.0


def popularity_exposure(recs, item_pop: np.ndarray) -> float:
    """Mean normalized popularity of recs (0..1 by log-rank-free scale)."""
    pop = np.asarray(item_pop, dtype=np.float64)
    lo, hi = np.log1p(pop.min()), np.log1p(pop.max())
    if hi == lo:
        return 0.0
    vals = [(np.log1p(pop[i]) - lo) / (hi - lo) for i in recs]
    return float(np.mean(vals)) if vals else 0.0


def long_tail_coverage(recs_all, item_pop: np.ndarray, q: float = 0.8) -> float:
    """Fraction of recommended items in the least-popular 20% of catalog."""
    thresh = np.quantile(item_pop, q)
    recs = [i for u in recs_all for i in u]
    if not recs:
        return 0.0
    return float(np.mean(item_pop[np.asarray(recs)] <= thresh))


def gini(items: np.ndarray) -> float:
    """Gini coefficient over recommendation counts per item."""
    if len(items) == 0:
        return 0.0
    _, counts = np.unique(items, return_counts=True)
    c = np.sort(counts.astype(np.float64))
    n = len(c)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * c).sum()) / (n * c.sum()) - (n + 1) / n)


def personalization(recs_all: list[list[int]], k: int = 10) -> float:
    """1 - mean pairwise Jaccard overlap between users' rec lists."""
    sets = [set(r[:k]) for r in recs_all if r]
    n = len(sets)
    if n < 2:
        return 0.0
    rng = np.random.default_rng(0)
    idx = rng.choice(n, min(400, n), replace=False)
    pairs, s = 0, 0.0
    sel = [sets[i] for i in idx]
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            u = sel[i] | sel[j]
            if u:
                s += len(sel[i] & sel[j]) / len(u)
            pairs += 1
    return float(1 - s / pairs) if pairs else 0.0


def head_exposure(recs, head_items: set) -> float:
    if not recs:
        return 0.0
    return float(np.mean([i in head_items for i in recs]))


def segment_bucket(n_pos_history: int) -> str:
    if n_pos_history < 10:
        return "cold"
    if n_pos_history < 30:
        return "light"
    if n_pos_history < 100:
        return "medium"
    return "heavy"


def bootstrap_delta_ci(a: np.ndarray, b: np.ndarray, iters: int = 2000,
                       seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap CI of mean(b)-mean(a). Returns (delta, lo, hi)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    rng = np.random.default_rng(seed)
    deltas = np.empty(iters)
    for it in range(iters):
        idx = rng.integers(0, n, n)
        deltas[it] = b[idx].mean() - a[idx].mean()
    return float(b.mean() - a.mean()), float(np.percentile(deltas, 2.5)), \
        float(np.percentile(deltas, 97.5))
