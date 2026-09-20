import numpy as np
import pandas as pd

from src.features import FEATURES, ItemStore, compute_features
from src.profile import build_profile


def _toy_store(n=20):
    movies = pd.DataFrame({
        "bayes_rating": np.linspace(3, 4.5, n),
        "rating_count": np.linspace(10, 5000, n),
        "vote_count": np.linspace(50, 9000, n),
        "popularity": np.linspace(1, 50, n),
        "imdb_rating": np.linspace(5, 9, n),
        "year": np.linspace(1980, 2020, n),
        "runtime": np.full(n, 110.0),
        "original_language": ["en"] * (n - 1) + ["fr"],
        "genres": [["Drama"] if i % 2 else ["Action", "Sci-Fi"] for i in range(n)],
        "director": [["Dir A"] if i < n // 2 else ["Dir B"] for i in range(n)],
        "cast": [["Act A", "Act B"] if i % 3 == 0 else ["Act C"] for i in range(n)],
        "movieId": np.arange(n) + 1,
        "title": [f"M{i}" for i in range(n)],
    })
    return ItemStore(movies)


def test_build_profile_shapes():
    store = _toy_store()
    emb = np.random.rand(20, 16).astype(np.float32)
    fac = np.random.rand(20, 8).astype(np.float32)
    items = np.array([0, 2, 4])
    prof = build_profile(items, np.array([1.0, 2.0, 1.0]), emb, fac,
                         store.genre_mat, store.directors, store.casts,
                         store.language, store.year)
    assert prof.content_vec.shape == (16,)
    assert prof.cf_vec.shape == (8,)
    assert np.isclose(np.linalg.norm(prof.content_vec), 1.0)
    assert prof.genre_w.sum() > 0
    assert "Dir A" in prof.directors


def test_compute_features_shape_finite():
    store = _toy_store()
    emb = np.random.rand(20, 16).astype(np.float32)
    fac = np.random.rand(20, 8).astype(np.float32)
    bias = np.random.rand(20).astype(np.float32)
    prof = build_profile(np.array([0, 2]), np.ones(2), emb, fac,
                         store.genre_mat, store.directors, store.casts,
                         store.language, store.year)
    cand = np.array([5, 6, 7, 8])
    meta = {5: {"content_sim": 0.9, "content_rank": 0, "in_pop": 1},
            6: {"cf_sim": 0.7, "cf_rank": 3}}
    X = compute_features(prof, cand, meta, store, item_factors=fac, item_bias=bias)
    assert X.shape == (4, len(FEATURES))
    assert np.isfinite(X).all()
    # candidate 5 had content hit -> in_content flag 1
    fi = FEATURES.index
    assert X[0, fi("in_content")] == 1.0
    assert X[0, fi("in_pop")] == 1.0
    assert X[1, fi("in_cf")] == 1.0
    assert X[2, fi("in_cf")] == 0.0


def test_negative_weights_push_profile():
    store = _toy_store()
    emb = np.eye(20, 4, dtype=np.float32)  # item i -> axis i
    fac = np.eye(20, 4, dtype=np.float32)
    # like item 0, dislike item 1
    prof = build_profile(np.array([0, 1]), np.array([2.0, -2.0]), emb, fac,
                         store.genre_mat, store.directors, store.casts,
                         store.language, store.year, tau_years=None)
    # content vec should point toward +axis0 -axis1
    assert prof.content_vec[0] > 0
    assert prof.content_vec[1] < 0
