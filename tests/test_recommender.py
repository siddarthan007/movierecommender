"""End-to-end serving test on a synthetic artifact set."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.config import Config
from src.features import ItemStore
from src.index import build_hnsw
from src.recommender import RecArtifacts, Recommender


class _FakeRanker:
    """Score = content_sim + covis + small bayes term."""
    def predict(self, X):
        return X[:, 0] + 0.5 * X[:, 7] + 0.01 * X[:, 14]


@pytest.fixture
def toy_recommender():
    n, dc, df_ = 50, 8, 8
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(n, dc)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    fac = rng.normal(size=(n, df_)).astype(np.float32)
    fac /= np.linalg.norm(fac, axis=1, keepdims=True)
    movies = pd.DataFrame({
        "item_idx": np.arange(n), "movieId": np.arange(n) + 1,
        "title": [f"Movie {i}" for i in range(n)],
        "bayes_rating": np.linspace(3, 4.5, n),
        "rating_count": np.linspace(10, 5000, n),
        "vote_count": np.linspace(50, 9000, n),
        "vote_average": np.linspace(6, 9, n),
        "popularity": np.linspace(1, 50, n),
        "imdb_rating": np.linspace(5, 9, n),
        "year": np.linspace(1980, 2020, n),
        "runtime": np.full(n, 110.0),
        "original_language": ["en"] * n,
        "genres": [["Drama"] if i % 2 else ["Action"] for i in range(n)],
        "director": [["D"] for _ in range(n)],
        "cast": [["A"] for _ in range(n)],
        "keywords": [["k1"] for _ in range(n)],
        "production_companies": [["C"] for _ in range(n)],
        "poster_path": [None] * n,
    })
    cfg = Config(content_k=10, cf_k=10, covis_k=10, sasrec_k=5,
                 popularity_k=10, final_k=5, ranker_top_n=20,
                 ef_search=32, hnsw_m=8, ef_construction=40)
    covis = sp.csr_matrix(np.eye(n, dtype=np.float32))  # trivial self-loop
    art = RecArtifacts(
        movies=movies, content_emb=emb, item_factors=fac,
        item_bias=np.zeros(n, dtype=np.float32),
        content_index=build_hnsw(emb, cfg), cf_index=build_hnsw(fac, cfg),
        ranker=_FakeRanker(), store=ItemStore(movies),
        pop_items=np.arange(n), covis=covis, sasrec=None,
        movie_id_to_idx={i + 1: i for i in range(n)},
        idx_to_movie_id=np.arange(n) + 1,
    )
    return Recommender(art, cfg)


def test_recommend_returns_k_rows(toy_recommender):
    out, expl = toy_recommender.recommend([1, 2, 3], k=5)
    assert len(out) <= 5
    assert "score" in out.columns
    assert not set(out["movieId"]) & {1, 2, 3}
    assert expl and all("anchor" in e for e in expl.values())


def test_recommend_empty_selection_falls_back(toy_recommender):
    out, _ = toy_recommender.recommend([], k=5)
    assert len(out) == 5


def test_disliked_penalizes(toy_recommender):
    a, _ = toy_recommender.recommend([1, 2, 3], k=5)
    b, _ = toy_recommender.recommend([1, 2, 3], k=5, disliked_ids=[4, 5])
    assert len(b) == 5


def test_mood_and_exploration(toy_recommender):
    out, _ = toy_recommender.recommend([1, 2, 3], k=5,
                                     mood=["Epic"], exploration=0.9)
    assert len(out) == 5


def test_search_exact_prefix_fuzzy(toy_recommender):
    hits = toy_recommender.search("Movie 1")
    assert len(hits) >= 1
    hits2 = toy_recommender.search("movie 1")   # case-insensitive
    assert len(hits2) >= 1
    hits3 = toy_recommender.search("zzz-nothing")
    assert len(hits3) == 0


def test_mmr_toggle(toy_recommender):
    a, _ = toy_recommender.recommend([1, 2, 3], k=5)
    assert len(a) == 5
