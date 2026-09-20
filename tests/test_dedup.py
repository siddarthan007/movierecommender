import numpy as np
import pandas as pd

from src.dedup import build_entities, dedupe_recs


def _movies():
    return pd.DataFrame({
        "title": ["Matrix, The (1999)", "The Matrix (1999)",
                  "Beauty and the Beast (1991)", "Beauty and the Beast (2017)",
                  "Obscure Film (2000)", "Obscure Film (2000)"],
        "year": [1999, 1999, 1991, 2017, 2000, 2000],
        "imdbId": [133093, 133093, 106714, 2771200, np.nan, np.nan],
        "tmdbId": [603, 603, 321612, 20, np.nan, np.nan],
        "director": [["Wachowski"], ["Wachowski"], ["Trousdale"],
                     ["Condon"], ["X"], ["X"]],
        "rating_count": [100, 5, 80, 60, 10, 2],
    })


def test_entity_grouping():
    key, stats = build_entities(_movies())
    # matrix dup by imdbId, beauty split by year, obscure dup by title+year
    assert stats["duplicate_groups"] >= 2
    assert stats["canonical_catalog"] == 4
    assert key[0] == key[1]          # matrix rows merged
    assert key[2] != key[3]          # beauty 1991 vs 2017 split
    assert key[4] == key[5]          # title+year merge


def test_dedupe_recs():
    key, stats = build_entities(_movies())
    out = dedupe_recs([0, 1, 2, 4, 5], key, stats["rep_item"])
    assert len(out) == 3
    assert 0 in out and 2 in out     # first occurrence kept


def test_rep_prefers_popular():
    _, stats = build_entities(_movies())
    # matrix rep = row 0 (100 ratings)
    assert stats["rep_item"]["tt133093"] == 0
