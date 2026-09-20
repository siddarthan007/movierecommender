import numpy as np
import polars as pl
import torch

from src.twotower import (TwoTower, build_tt_examples, tt_item_features,
                          tt_user_logits, tt_user_vec)


def _toy_feat(n=50, d_in=20):
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, d_in)).astype(np.float32)


def test_tower_shapes():
    feat = _toy_feat()
    m = TwoTower(20, dim=16)
    v = m.encode_items(torch.tensor(feat[:5]))
    assert v.shape == (5, 16)
    assert np.allclose(np.linalg.norm(v.detach().numpy(), axis=1), 1, atol=1e-4)


def test_user_vec_norm_and_dim():
    feat = _toy_feat()
    m = TwoTower(20, dim=16)
    items = np.array([1, 2, 3, 4])
    u = tt_user_vec(m, feat, items, np.array([4.0, 2.0, 5.0, 3.5]))
    assert u.shape == (16,)
    assert abs(np.linalg.norm(u) - 1.0) < 1e-4


def test_logits_full_catalog():
    feat = _toy_feat()
    m = TwoTower(20, dim=16)
    emb = m.encode_items(torch.tensor(feat)).detach().numpy()
    lg = tt_user_logits(m, feat, emb, np.array([0, 1]),
                        np.array([4.0, 4.5]))
    assert lg.shape == (50,)
    assert np.isfinite(lg).all()


def test_examples_within_bounds():
    items = np.arange(20)
    rts = np.array([4.0] * 20)
    rng = np.random.default_rng(0)
    pairs = build_tt_examples(items, rts, 3.5, snaps=4, min_hist=5, rng=rng)
    assert pairs
    for cut, tgt in pairs:
        assert 5 <= cut < 20
        assert tgt >= cut


def test_examples_empty_when_no_future():
    items = np.arange(8)
    rts = np.array([4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    rng = np.random.default_rng(0)
    # positives only at idx 0,1 -> cuts >= 5 leave no future positive
    pairs = build_tt_examples(items, rts, 3.5, snaps=6, min_hist=5, rng=rng)
    assert pairs == []


def test_item_feat_concat():
    e = np.ones((10, 8), dtype=np.float32)
    f = np.ones((10, 4), dtype=np.float32)
    assert tt_item_features(e, f, 10).shape == (10, 12)
    assert tt_item_features(e, None, 10).shape == (10, 8)
