import numpy as np
import faiss

from src.config import Config
from src.index import build_hnsw, query
from src.candidates import generate_candidates, popularity_universe
from src.profile import build_profile


def _norm(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_hnsw_recall_exact_neighbors():
    rng = np.random.default_rng(1)
    base = _norm(rng.normal(size=(2000, 32)).astype(np.float32))
    q = _norm(rng.normal(size=(50, 32)).astype(np.float32))
    idx = build_hnsw(base, Config(hnsw_m=16, ef_construction=80, ef_search=64))
    D, I = query(idx, q, 10)
    # brute-force ground truth
    sims = q @ base.T
    truth = np.argsort(-sims, axis=1)[:, :10]
    hit = np.mean([len(set(I[r]) & set(truth[r])) / 10 for r in range(50)])
    assert hit > 0.95


def test_popularity_universe_prefers_quality_reach():
    rc = np.array([10, 10000, 5, 1000], dtype=np.float32)
    bayes = np.array([4.9, 4.5, 5.0, 4.2], dtype=np.float32)
    pop = popularity_universe(rc, bayes, k=2)
    assert 1 in pop and 3 in pop


def test_generate_candidates_excludes_seen():
    rng = np.random.default_rng(0)
    emb = _norm(rng.normal(size=(60, 8)).astype(np.float32))
    fac = _norm(rng.normal(size=(60, 8)).astype(np.float32))
    cfg = Config(content_k=10, cf_k=10, popularity_k=10)
    ci = build_hnsw(emb, cfg)
    fi = build_hnsw(fac, cfg)
    genre_mat = np.ones((60, 2), dtype=np.float32)
    dirs = [frozenset() for _ in range(60)]
    casts = [frozenset() for _ in range(60)]
    langs = np.array(["en"] * 60, dtype=object)
    years = np.full(60, 2000.0)
    prof = build_profile(np.array([0, 1]), np.ones(2), emb, fac,
                         genre_mat, dirs, casts, langs, years)
    pop = np.arange(60)
    cand, meta = generate_candidates(prof, ci, fi, pop, cfg)
    assert 0 not in cand and 1 not in cand
    assert len(cand) > 0
    assert all(int(c) in meta for c in cand)
