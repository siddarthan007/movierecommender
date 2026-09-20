import numpy as np
import polars as pl
import torch

from src.cf import BPRMF, train_bpr
from src.config import Config


def test_bpr_forward_shape():
    m = BPRMF(10, 30, 8)
    u = torch.tensor([0, 1, 2])
    i = torch.tensor([3, 4, 5])
    s = m.score(u, i)
    assert s.shape == (3,)


def test_bpr_learns_toy_pattern():
    # 2 user groups x 2 item groups, block-structured positives
    rng = np.random.default_rng(0)
    rows = []
    for u in range(200):
        grp = 0 if u < 100 else 1
        items = range(0, 50) if grp == 0 else range(50, 100)
        for i in rng.choice(list(items), 15, replace=False):
            rows.append((u, int(i), 4.5, 1000 + u))
    df = pl.DataFrame(rows, schema=["userId", "item_idx", "rating", "timestamp"],
                      orient="row")
    cfg = Config(cf_dim=16, cf_epochs=6, cf_lr=0.05, cf_reg=1e-5,
                 cf_batch=4096, cf_neg=2, cf_pos_threshold=3.5)
    res = train_bpr(df, 100, cfg=cfg, device="cpu")
    fac = res["item_factors"]
    # group-0 user should prefer items 0..49
    u0 = res["user_factors"][res["user_map"][0]]
    s = fac @ u0
    top = set(np.argsort(-s)[:10])
    assert len(top & set(range(50))) >= 8
