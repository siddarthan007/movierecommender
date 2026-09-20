"""Ranking metrics for offline evaluation."""
from __future__ import annotations

import numpy as np


def recall_at_k(recs: list[int], truth: set[int], k: int) -> float:
    if not truth:
        return np.nan
    return len(set(recs[:k]) & truth) / len(truth)


def precision_at_k(recs: list[int], truth: set[int], k: int) -> float:
    return len(set(recs[:k]) & truth) / k


def ndcg_at_k(recs: list[int], truth: set[int], k: int) -> float:
    if not truth:
        return np.nan
    dcg = sum(1.0 / np.log2(i + 2) for i, r in enumerate(recs[:k]) if r in truth)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(truth), k)))
    return dcg / ideal if ideal else np.nan


def hit_rate_at_k(recs: list[int], truth: set[int], k: int) -> float:
    return 1.0 if set(recs[:k]) & truth else 0.0


def catalog_coverage(all_recs: list[list[int]], n_items: int) -> float:
    flat = {i for recs in all_recs for i in recs}
    return len(flat) / n_items


def intra_list_diversity(recs: list[int], emb: np.ndarray) -> float:
    """1 - mean pairwise cosine sim inside the list (emb L2-normalized)."""
    if len(recs) < 2:
        return np.nan
    v = emb[np.asarray(recs)]
    sim = v @ v.T
    n = len(recs)
    off = sim[~np.eye(n, dtype=bool)].mean()
    return float(1.0 - off)


def aggregate(per_user: list[dict], k_list=(10, 20)) -> dict:
    out = {}
    for k in k_list:
        out[f"recall@{k}"] = float(np.nanmean([u[f"recall@{k}"] for u in per_user]))
        out[f"ndcg@{k}"] = float(np.nanmean([u[f"ndcg@{k}"] for u in per_user]))
        out[f"precision@{k}"] = float(np.nanmean([u[f"precision@{k}"] for u in per_user]))
        out[f"hitrate@{k}"] = float(np.nanmean([u[f"hitrate@{k}"] for u in per_user]))
    return out
