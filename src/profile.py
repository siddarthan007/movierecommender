"""User profile: dense vectors + structured affinities from liked items.

Two embedding views (content / CF) are kept SEPARATE — scores are fused
downstream, never concatenated into one space.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


@dataclass
class UserProfile:
    items: np.ndarray                 # item_idx of profile items
    weights: np.ndarray               # per-item weight
    content_vec: np.ndarray           # (d_content,) normalized
    cf_vec: np.ndarray                # (d_cf,) normalized
    genre_w: np.ndarray               # (n_genres,) normalized affinity
    decade_w: np.ndarray              # (n_decade_buckets,) normalized affinity
    directors: frozenset = field(default_factory=frozenset)
    cast: frozenset = field(default_factory=frozenset)
    keywords: frozenset = field(default_factory=frozenset)
    companies: frozenset = field(default_factory=frozenset)
    languages: frozenset = field(default_factory=frozenset)
    year_mean: float = np.nan
    year_std: float = np.nan
    year_max: float = np.nan
    pos_ratio: float = 1.0


def build_profile(
    items: np.ndarray,
    weights: np.ndarray,
    content_emb: np.ndarray,
    item_factors: np.ndarray,
    genre_mat: np.ndarray,            # (n_items, n_genres)
    directors: list[frozenset],
    casts: list[frozenset],
    languages: np.ndarray,            # (n_items,) object/str
    years: np.ndarray,                # (n_items,)
    timestamps: np.ndarray | None = None,
    tau_years: float | None = 3.0,
    keywords: list[frozenset] | None = None,
    companies: list[frozenset] | None = None,
    n_decades: int = 14,
) -> UserProfile:
    """Weighted-average dual embedding + structured affinities.

    weights: e.g. rating - 2.5 (negatives push the profile away).
    timestamps: optional unix ts -> exponential recency decay
        w *= exp(-(t_now - t)/ (tau_years*365.25*86400))
    """
    items = np.asarray(items, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64).copy()
    if timestamps is not None and tau_years:
        t_now = timestamps.max()
        decay = np.exp(-(t_now - np.asarray(timestamps, dtype=np.float64))
                       / (tau_years * 365.25 * 86400))
        weights = weights * decay
    pos = weights > 0
    pos_ratio = float(pos.mean()) if len(pos) else 0.0
    if pos.sum() == 0:
        pos = np.ones_like(weights, dtype=bool)
        weights = np.ones_like(weights)
    pw = np.clip(weights, 0, None)
    if pw.sum() == 0:
        pw = np.ones_like(weights)

    content_vec = _norm((content_emb[items].astype(np.float64) * weights[:, None]).sum(0))
    cf_vec = _norm((item_factors[items].astype(np.float64) * weights[:, None]).sum(0))
    genre_w = (genre_mat[items].astype(np.float64) * pw[:, None]).sum(0)
    genre_w = genre_w / genre_w.sum() if genre_w.sum() > 0 else genre_w

    decade_idx = np.clip((np.nan_to_num(years[items], nan=2000) // 10 * 10 - 1900) // 10,
                         0, n_decades - 1).astype(int)
    decade_w = np.zeros(n_decades)
    np.add.at(decade_w, decade_idx, pw)
    decade_w = decade_w / decade_w.sum() if decade_w.sum() > 0 else decade_w

    dirs, cast_set, kw_set, comp_set = set(), set(), set(), set()
    for i in items[pos]:
        dirs |= set(directors[i])
        cast_set |= set(casts[i])
        if keywords is not None:
            kw_set |= set(keywords[i])
        if companies is not None:
            comp_set |= set(companies[i])
    langs = set(np.asarray(languages, dtype=object)[items[pos]].tolist()) - {None, ""}

    yrs = years[items[pos]]
    yrs = yrs[~np.isnan(yrs)]
    return UserProfile(
        items=items, weights=weights,
        content_vec=content_vec, cf_vec=cf_vec, genre_w=genre_w,
        decade_w=decade_w,
        directors=frozenset(dirs), cast=frozenset(cast_set),
        keywords=frozenset(kw_set), companies=frozenset(comp_set),
        languages=frozenset(langs),
        year_mean=float(yrs.mean()) if len(yrs) else np.nan,
        year_std=float(yrs.std()) if len(yrs) else np.nan,
        year_max=float(yrs.max()) if len(yrs) else np.nan,
        pos_ratio=pos_ratio,
    )
