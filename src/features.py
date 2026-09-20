"""Ranker feature builder + item-side feature store.

ItemStore precomputes per-item arrays indexed by item_idx so feature
extraction is vectorized numpy, not dataframe scans.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .profile import UserProfile

N_DECADES = 14

FEATURES = [
    # retrieval signals
    "content_sim", "content_rank_inv", "in_content",
    "cf_sim", "cf_rank_inv", "in_cf", "cf_bias",
    "covis_score", "covis_rank_inv", "in_covis",
    "sasrec_score", "sasrec_rank_inv", "in_sasrec",
    "in_pop",
    # quality / popularity
    "bayes_rating", "log_rating_count", "log_vote_count",
    "log_popularity", "imdb_rating", "vote_avg_tmdb",
    # temporal
    "year", "year_diff", "decade_affinity", "newer_than_profile",
    # item attrs
    "runtime", "genre_affinity", "genre_count",
    "director_match", "cast_overlap", "keyword_overlap",
    "company_match", "language_match",
    # history similarity (not captured by mean-pooled profile vec)
    "hist_cos_max", "hist_cos_mean", "hist_cos_top3",
    "hist_cf_max", "hist_cf_mean", "hist_cf_top3",
    "hist_cos_wavg",
    # covis structure exposed to the ranker
    "covis_max", "covis_mean", "covis_top3", "covis_recent", "covis_wavg",
    # profile context
    "profile_size", "pos_ratio",
    # two-tower learned retrieval (0 when source absent)
    "tt_score", "tt_rank_inv", "in_tt",
]


class ItemStore:
    """Per-item arrays + set lookups, indexed by item_idx."""

    def __init__(self, movies: pd.DataFrame):
        def num(col, fill=0.0):
            if col not in movies:
                return np.full(len(movies), fill, dtype=np.float32)
            s = pd.to_numeric(movies[col], errors="coerce")
            return s.fillna(fill).to_numpy(np.float32)

        self.movies = movies
        self.n = len(movies)
        self.bayes = num("bayes_rating", 3.5)
        self.rating_count = num("rating_count")
        self.vote_count = num("vote_count")
        self.popularity = num("popularity")
        self.imdb_rating = num("imdb_rating")
        self.vote_avg = num("vote_average")
        year = pd.to_numeric(movies["year"], errors="coerce")
        ymed = year.median()
        self.year = year.fillna(ymed if np.isfinite(ymed) else 2000.0).to_numpy(np.float32)
        runtime = pd.to_numeric(movies["runtime"], errors="coerce")
        rmed = runtime.median()
        self.runtime = runtime.fillna(rmed if np.isfinite(rmed) else 100.0).to_numpy(np.float32)
        self.language = movies["original_language"].fillna("").to_numpy(object)
        self.decade = np.clip((self.year // 10 * 10 - 1900) // 10, 0,
                              N_DECADES - 1).astype(np.int64)

        genre_lists = [gs if isinstance(gs, (list, np.ndarray)) else []
                       for gs in movies["genres"]]
        self.genre_vocab = sorted({g for gs in genre_lists for g in gs})
        gidx = {g: i for i, g in enumerate(self.genre_vocab)}
        self.genre_mat = np.zeros((self.n, len(self.genre_vocab)), dtype=np.float32)
        self.genre_count = np.zeros(self.n, dtype=np.float32)
        for i, gs in enumerate(genre_lists):
            self.genre_count[i] = len(gs)
            for g in gs:
                if g in gidx:
                    self.genre_mat[i, gidx[g]] = 1.0

        def to_sets(col):
            if col not in movies:
                return [frozenset() for _ in range(self.n)]
            return [frozenset(v) if isinstance(v, (list, np.ndarray)) else frozenset()
                    for v in movies[col]]

        self.directors = to_sets("director")
        self.casts = to_sets("cast")
        self.keywords = to_sets("keywords")
        self.companies = to_sets("production_companies")


def compute_features(profile: UserProfile, cand: np.ndarray,
                     cand_meta: dict, store: ItemStore,
                     content_emb: np.ndarray | None = None,
                     item_factors: np.ndarray | None = None,
                     item_bias: np.ndarray | None = None,
                     covis_mat=None) -> np.ndarray:
    """(n_cand, len(FEATURES)) float32 feature matrix for one query."""
    n = len(cand)
    X = np.zeros((n, len(FEATURES)), dtype=np.float32)

    metas = [cand_meta.get(int(c), {}) for c in cand]
    cs = np.array([m.get("content_sim", 0.0) for m in metas], dtype=np.float32)
    cr = np.array([m.get("content_rank", -1) for m in metas], dtype=np.float32)
    fs = np.array([m.get("cf_sim", 0.0) for m in metas], dtype=np.float32)
    fr = np.array([m.get("cf_rank", -1) for m in metas], dtype=np.float32)
    vs = np.array([m.get("covis_score", 0.0) for m in metas], dtype=np.float32)
    vr = np.array([m.get("covis_rank", -1) for m in metas], dtype=np.float32)
    ss = np.array([m.get("sasrec_score", 0.0) for m in metas], dtype=np.float32)
    sr = np.array([m.get("sasrec_rank", -1) for m in metas], dtype=np.float32)
    ts = np.array([m.get("tt_score", 0.0) for m in metas], dtype=np.float32)
    tr = np.array([m.get("tt_rank", -1) for m in metas], dtype=np.float32)

    X[:, 0] = cs
    np.divide(1.0, cr + 1.0, out=X[:, 1], where=cr >= 0)
    X[:, 2] = (cr >= 0).astype(np.float32)
    X[:, 3] = fs
    np.divide(1.0, fr + 1.0, out=X[:, 4], where=fr >= 0)
    X[:, 5] = (fr >= 0).astype(np.float32)
    if item_bias is not None:
        X[:, 6] = item_bias[cand]
    if item_factors is not None:
        dots = item_factors[cand] @ profile.cf_vec
        X[:, 3] = np.where(X[:, 5] > 0, X[:, 3], dots.astype(np.float32))
    X[:, 7] = vs
    np.divide(1.0, vr + 1.0, out=X[:, 8], where=vr >= 0)
    X[:, 9] = (vr >= 0).astype(np.float32)
    X[:, 10] = ss
    np.divide(1.0, sr + 1.0, out=X[:, 11], where=sr >= 0)
    X[:, 12] = (sr >= 0).astype(np.float32)
    X[:, 13] = np.array([m.get("in_pop", 0) for m in metas], dtype=np.float32)

    X[:, 14] = store.bayes[cand]
    X[:, 15] = np.log1p(store.rating_count[cand])
    X[:, 16] = np.log1p(store.vote_count[cand])
    X[:, 17] = np.log1p(np.clip(store.popularity[cand], 0, None))
    X[:, 18] = store.imdb_rating[cand]
    X[:, 19] = store.vote_avg[cand]

    X[:, 20] = store.year[cand]
    if not np.isnan(profile.year_mean):
        X[:, 21] = np.abs(store.year[cand] - profile.year_mean)
    else:
        X[:, 21] = np.abs(store.year[cand] - np.median(store.year))
    if profile.decade_w.size:
        X[:, 22] = profile.decade_w[store.decade[cand]]
    if not np.isnan(profile.year_max):
        X[:, 23] = np.clip(store.year[cand] - profile.year_max, -50, 50) / 50.0

    X[:, 24] = store.runtime[cand]
    X[:, 25] = (store.genre_mat[cand] * profile.genre_w[None, :]).sum(1) \
        if profile.genre_w.size else 0.0
    X[:, 26] = store.genre_count[cand]

    # per-candidate set overlaps (python loop; n is small vs profile sizes)
    for r, c in enumerate(cand):
        c = int(c)
        X[r, 27] = 1.0 if profile.directors & store.directors[c] else 0.0
        X[r, 28] = min(len(profile.cast & store.casts[c]), 3) / 3.0
        X[r, 29] = min(len(profile.keywords & store.keywords[c]), 5) / 5.0
        X[r, 30] = 1.0 if profile.companies & store.companies[c] else 0.0
        X[r, 31] = 1.0 if store.language[c] and store.language[c] in profile.languages else 0.0

    # history similarity: candidate vs each profile item
    if content_emb is not None and len(profile.items):
        pe = content_emb[profile.items]          # (P, d) normalized
        sims = content_emb[cand] @ pe.T          # (n, P)
        X[:, 32] = sims.max(1)
        X[:, 33] = sims.mean(1)
        k3 = min(3, sims.shape[1])
        X[:, 34] = np.sort(-sims, axis=1)[:, :k3].mean(1) * -1
        w = np.clip(profile.weights, 0, None)
        w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
        X[:, 38] = sims @ w
    if item_factors is not None and len(profile.items):
        pf = item_factors[profile.items]
        fsim = item_factors[cand] @ pf.T
        X[:, 35] = fsim.max(1)
        X[:, 36] = fsim.mean(1)
        k3 = min(3, fsim.shape[1])
        X[:, 37] = np.sort(-fsim, axis=1)[:, :k3].mean(1) * -1

    # covis evidence per profile item: C[p_i, cand] stats
    if covis_mat is not None and len(profile.items):
        E = np.asarray(covis_mat[profile.items][:, cand].todense()).T  # (n,P)
        if E.size:
            X[:, 39] = E.max(1)
            X[:, 40] = E.mean(1)
            k3 = min(3, E.shape[1])
            X[:, 41] = np.sort(-E, axis=1)[:, :k3].mean(1) * -1
            r3 = min(3, E.shape[1])
            X[:, 42] = E[:, -r3:].mean(1)          # most recent items last
            w = np.clip(profile.weights, 0, None)
            w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
            X[:, 43] = E @ w

    X[:, 44] = np.log1p(len(profile.items))
    X[:, 45] = profile.pos_ratio
    X[:, 46] = ts
    np.divide(1.0, tr + 1.0, out=X[:, 47], where=tr >= 0)
    X[:, 48] = (tr >= 0).astype(np.float32)
    return X
