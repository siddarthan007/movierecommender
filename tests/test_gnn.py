import numpy as np
import polars as pl
import torch

from src.gnn import LightGCN, SeenIndex, build_norm_adj, train_lightgcn
from src.config import Config


def test_norm_adj_symmetric():
    pos_u = np.array([0, 0, 1])
    pos_i = np.array([0, 1, 0])
    adj = build_norm_adj(pos_u, pos_i, 2, 2, "cpu")
    dense = adj.to_dense()
    assert torch.allclose(dense, dense.T)
    assert dense.shape == (4, 4)


def test_propagation_shape():
    pos_u = np.array([0, 0, 1])
    pos_i = np.array([0, 1, 0])
    adj = build_norm_adj(pos_u, pos_i, 2, 2, "cpu")
    m = LightGCN(2, 2, 8, n_layers=2)
    out = m.propagate(adj)
    assert out.shape == (4, 8)


def test_seen_index_membership():
    pos_u = np.array([0, 0, 1])
    pos_i = np.array([5, 7, 5])
    si = SeenIndex(pos_u, pos_i, 2, "cpu")
    assert si.is_positive(torch.tensor([0]), torch.tensor([5])).item()
    assert not si.is_positive(torch.tensor([0]), torch.tensor([6])).item()
    assert si.is_positive(torch.tensor([1]), torch.tensor([5])).item()
    assert not si.is_positive(torch.tensor([1]), torch.tensor([7])).item()


def test_lightgcn_learns_blocks():
    rng = np.random.default_rng(0)
    rows = []
    for u in range(150):
        grp = 0 if u < 75 else 1
        items = range(0, 40) if grp == 0 else range(40, 80)
        for i in rng.choice(list(items), 12, replace=False):
            rows.append((u, int(i), 4.5, 1000 + u))
    df = pl.DataFrame(rows, schema=["userId", "item_idx", "rating", "timestamp"],
                      orient="row")
    cfg = Config(cf_batch=2048, cf_neg=2, cf_pos_threshold=3.5)
    res = train_lightgcn(df, 80, None, cfg, dim=16, layers=2, epochs=15,
                         lr=0.02, reg=1e-5, device="cpu")
    u0 = res["user_factors"][res["user_map"][0]]
    s = res["item_factors"] @ u0
    top = set(np.argsort(-s)[:10])
    assert len(top & set(range(40))) >= 6
