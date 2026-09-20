"""Canonical movie entities + catalog deduplication.

Entity key resolution, in order:
  1. imdbId        -> 'tt{imdbId}'
  2. tmdbId        -> 'tmdb{tmdbId}'
  3. norm title + exact year           -> 't:{title}|{year}'
  4. norm title + year±1 + same director set -> merged into the neighbor's key

Representative item per entity = highest rating_count (reach).
Serving/eval filter recommended lists down to one item per entity.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _norm_title(t: str) -> str:
    t = str(t).lower().strip()
    t = re.sub(r"\s*\(\d{4}\)\s*$", "", t)
    m = re.match(r"^(.*),\s*(the|a|an)$", t)
    if m:
        t = f"{m.group(2)} {m.group(1)}"
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def build_entities(movies: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Returns (entity_key per item_idx, stats dict)."""
    norm = movies["title"].map(_norm_title).to_numpy()
    year = pd.to_numeric(movies["year"], errors="coerce").to_numpy()
    imdb = movies["imdbId"].to_numpy() if "imdbId" in movies else np.full(len(movies), np.nan)
    tmdb = movies["tmdbId"].to_numpy() if "tmdbId" in movies else np.full(len(movies), np.nan)
    directors = movies["director"] if "director" in movies else pd.Series([[]] * len(movies))
    dir_sets = [frozenset(d) if isinstance(d, (list, np.ndarray)) else frozenset()
                for d in directors]

    key_of = {}
    groups: dict[str, list[int]] = {}
    for i in range(len(movies)):
        if np.isfinite(imdb[i]):
            k = f"tt{int(imdb[i])}"
        elif np.isfinite(tmdb[i]):
            k = f"tmdb{int(tmdb[i])}"
        else:
            y = int(year[i]) if np.isfinite(year[i]) else -1
            k = f"t:{norm[i]}|{y}"
        key_of[i] = k
        groups.setdefault(k, []).append(i)

    # stage-4 pass: title+year±1+same-director merge for items with no strong ID
    # (fold their group into the ±1y group sharing the director)
    title_year_groups: dict[tuple[str, int], str] = {}
    for i in range(len(movies)):
        if not (np.isfinite(imdb[i]) or np.isfinite(tmdb[i])):
            continue
        y = int(year[i]) if np.isfinite(year[i]) else -1
        title_year_groups[(norm[i], y)] = key_of[i]
    for i in range(len(movies)):
        if np.isfinite(imdb[i]) or np.isfinite(tmdb[i]):
            continue
        y = int(year[i]) if np.isfinite(year[i]) else -1
        if y == -1:
            continue
        for dy in (-1, 1):
            g = title_year_groups.get((norm[i], y + dy))
            if g and dir_sets[i] and not dir_sets[i].isdisjoint(
                    set().union(*[dir_sets[j] for j in groups[g]])):
                # merge this item's group into g
                for j in groups[key_of[i]]:
                    key_of[j] = g
                    groups[g].append(j)
                groups.pop(key_of[i], None)
                break

    entity_key = np.array([key_of[i] for i in range(len(movies))], dtype=object)
    # representative = max rating_count within entity
    rc = pd.to_numeric(movies["rating_count"], errors="coerce").fillna(0).to_numpy()
    rep_of: dict[str, int] = {}
    for k, members in groups.items():
        rep_of[k] = max(members, key=lambda j: rc[j])
    merged = sum(len(v) for v in groups.values()) - len(groups)
    return entity_key, {
        "raw_catalog": len(movies),
        "canonical_catalog": len(groups),
        "duplicate_groups": int(sum(1 for v in groups.values() if len(v) > 1)),
        "movies_merged": int(merged),
        "rep_item": rep_of,
        "entity_key": entity_key,
    }


def dedupe_recs(rec_items: list[int], entity_key: np.ndarray,
                rep_of: dict[str, int]) -> list[int]:
    """Keep one item per entity; prefer canonical rep if dup hits."""
    seen, out = set(), []
    for i in rec_items:
        k = entity_key[int(i)]
        if k in seen:
            continue
        seen.add(k)
        out.append(int(i))
    return out
