import numpy as np
import polars as pl

from src.covis import build_covis, covis_scores


def test_covis_captures_cooccurrence():
    # items 0,1 co-rated by many users; 2,3 by others
    rows = []
    for u in range(100):
        if u < 50:
            rows += [(u, 0, 4.5, u), (u, 1, 4.0, u)]
        else:
            rows += [(u, 2, 4.5, u), (u, 3, 4.0, u)]
    df = pl.DataFrame(rows, schema=["userId", "item_idx", "rating", "timestamp"],
                      orient="row")
    C = build_covis(df, 4, 3.5, top_k=10, item_block=50)
    assert C[0, 1] > 0 and C[2, 3] > 0
    assert C[0, 2] == 0


def test_covis_scores_rank_neighbors():
    rows = []
    for u in range(80):
        rows += [(u, 0, 4.5, u), (u, 1, 4.0, u), (u, 5, 1.0, u)]
    df = pl.DataFrame(rows, schema=["userId", "item_idx", "rating", "timestamp"],
                      orient="row")
    C = build_covis(df, 6, 3.5, top_k=10, item_block=100)
    s = covis_scores(C, np.array([0]), np.array([1.0]))
    assert s[1] > s[4]  # co-rated neighbor > unrelated
    assert s[0] == 0    # self excluded (diag=0)
