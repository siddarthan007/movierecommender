import numpy as np
import polars as pl
import torch

from src.sasrec import SASRec, build_sequences, sasrec_score_items


def test_build_sequences_padding():
    df = pl.DataFrame({
        "userId": [1, 1, 1, 2, 2, 2, 2, 2],
        "item_idx": [5, 3, 8, 1, 2, 3, 4, 5],
        "rating": [4.0] * 8,
        "timestamp": [1, 2, 3, 1, 2, 3, 4, 5],
    })
    seqs, uids = build_sequences(df, max_len=6, min_len=3)
    assert seqs.shape == (2, 6)
    assert (seqs[0] == [6, 4, 9, 0, 0, 0]).all()   # item_idx+1, right-pad
    assert (seqs[1] == [2, 3, 4, 5, 6, 0]).all()


def test_sasrec_forward_shapes():
    m = SASRec(50, dim=16, max_len=10)
    seq = torch.randint(0, 50, (4, 10)) + 1
    h = m.encode(seq)
    assert h.shape == (4, 10, 16)
    lg = m.logits(seq)
    assert lg.shape == (4, 50)


def test_sasrec_score_items():
    m = SASRec(50, dim=16, max_len=10)
    sc = sasrec_score_items(m, np.array([1, 2, 3]), np.array([4, 5]), 10, "cpu")
    assert sc.shape == (2,)
    assert np.isfinite(sc).all()
