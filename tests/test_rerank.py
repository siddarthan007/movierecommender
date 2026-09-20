import numpy as np

from src.rerank import mmr_select


def test_mmr_prefers_diverse():
    # 4 candidates: 0,1 near-identical; 2,3 diverse. All scores ~equal.
    emb = np.array([
        [1.0, 0.0],
        [0.99, 0.1],
        [0.0, 1.0],
        [-1.0, 0.0],
    ], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    scores = np.array([1.0, 0.99, 0.98, 0.97])
    cand = np.arange(4)
    sel = mmr_select(scores, cand, emb, k=3, lam=0.5)
    assert sel[0] == 0
    # with lam=0.5 MMR should avoid picking the near-duplicate at slot 2
    assert sel[1] in (2, 3)
    assert len(set(sel)) == 3


def test_mmr_pure_relevance_when_lambda_1():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(10, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    scores = np.arange(10, dtype=np.float64)
    sel = mmr_select(scores, np.arange(10), emb, k=4, lam=1.0)
    assert list(sel) == [9, 8, 7, 6]


def test_mmr_respects_k():
    emb = np.eye(6, dtype=np.float32)
    sel = mmr_select(np.ones(6), np.arange(6), emb, k=3, lam=0.5)
    assert len(sel) == 3
