"""Multi-source candidate generation.

content ANN + CF ANN + co-visitation + SASRec + popularity -> dedup union.
Returns candidate array + per-source score/rank lookup for the ranker.
"""
from __future__ import annotations

import numpy as np
import faiss

from .config import CFG, Config
from .index import query
from .profile import UserProfile


def popularity_universe(rating_count: np.ndarray, bayes: np.ndarray,
                        k: int) -> np.ndarray:
    """Top-k items by Bayesian-quality x reach."""
    score = bayes * np.log1p(rating_count)
    return np.argpartition(-score, min(k, len(score) - 1))[:k]


def _topk_unseen(scores: np.ndarray, seen: set, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k indices of `scores` excluding seen items."""
    s = scores.copy()
    if seen:
        s[list(seen)] = -np.inf
    k = min(k, len(s))
    part = np.argpartition(-s, k - 1)[:k]
    order = np.argsort(-s[part])
    idx = part[order]
    return idx, s[idx]


def generate_candidates(profile: UserProfile,
                        content_index: faiss.Index,
                        cf_index: faiss.Index | None,
                        pop_items: np.ndarray,
                        cfg: Config = CFG,
                        covis_scores: np.ndarray | None = None,
                        sasrec_logits: np.ndarray | None = None,
                        tt_logits: np.ndarray | None = None,
                        ) -> tuple[np.ndarray, dict]:
    seen = set(profile.items.tolist())
    cand: dict[int, dict] = {}

    d, i = query(content_index, profile.content_vec[None, :], cfg.content_k)
    for rank, (sc, idx) in enumerate(zip(d[0], i[0])):
        if idx < 0 or idx in seen:
            continue
        cand.setdefault(int(idx), {})["content_sim"] = float(sc)
        cand[int(idx)]["content_rank"] = rank

    if cf_index is not None:
        d, i = query(cf_index, profile.cf_vec[None, :], cfg.cf_k)
        for rank, (sc, idx) in enumerate(zip(d[0], i[0])):
            if idx < 0 or idx in seen:
                continue
            cand.setdefault(int(idx), {})["cf_sim"] = float(sc)
            cand[int(idx)]["cf_rank"] = rank

    if covis_scores is not None:
        idxs, vals = _topk_unseen(covis_scores, seen, cfg.covis_k)
        vmax = float(vals.max()) if len(vals) and vals.max() > 0 else 1.0
        for rank, (idx, v) in enumerate(zip(idxs, vals)):
            cand.setdefault(int(idx), {})["covis_score"] = float(v / vmax)
            cand[int(idx)]["covis_rank"] = rank

    if sasrec_logits is not None:
        idxs, vals = _topk_unseen(sasrec_logits, seen, cfg.sasrec_k)
        for rank, (idx, v) in enumerate(zip(idxs, vals)):
            cand.setdefault(int(idx), {})["sasrec_score"] = float(v)
            cand[int(idx)]["sasrec_rank"] = rank

    if tt_logits is not None:
        idxs, vals = _topk_unseen(tt_logits, seen, cfg.tt_k)
        for rank, (idx, v) in enumerate(zip(idxs, vals)):
            cand.setdefault(int(idx), {})["tt_score"] = float(v)
            cand[int(idx)]["tt_rank"] = rank

    for idx in pop_items:
        idx = int(idx)
        if idx not in seen:
            cand.setdefault(idx, {})["in_pop"] = 1

    idx_arr = np.fromiter(cand.keys(), dtype=np.int64)
    return idx_arr, cand
