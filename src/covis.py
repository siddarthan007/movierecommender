"""Item-item collaborative retriever: co-visitation graph.

C[i,j] = sum over users of w_u * 1[u rated i positively] * 1[u rated j positively]
with w_u = 1/log(1 + n_u) damping heavy raters, then cosine-normalized:
  C_norm = D^-1/2 C D^-1/2, pruned to top-K neighbors per item.

Computed ITEM-BLOCKED (rows of C at a time) so peak memory stays bounded:
  C[blk, :] = (R[:, blk].T @ R)  -> normalize -> top-K per row.

Serving score:  s(c) = sum_{i in profile} w_i * C_norm[i, c]
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .config import CFG, Config


def build_covis(ratings_train, n_items: int,
                threshold: float, top_k: int = 100,
                user_block: int = 40000, item_block: int = 4096,
                max_items_per_user: int = 300,
                recency_floor: float = 1.0,
                norm: str = "cosine") -> sp.csr_matrix:
    """recency_floor < 1.0 weights each user's edges by recency rank:
    oldest kept item gets recency_floor*w, newest gets w (linear ramp).
    norm: 'cosine' = C/sqrt(d_i d_j); 'jaccard' = C/(d_i + d_j - C)."""
    import polars as pl
    pos = ratings_train.filter(pl.col("rating") >= threshold)
    uniq_users, u_dense = np.unique(pos["userId"].to_numpy(), return_inverse=True)
    n_users = len(uniq_users)
    items = pos["item_idx"].to_numpy().astype(np.int64)
    print(f"[covis] users={n_users:,} positives={len(items):,}")

    # per-user weights; cap ultra-long histories at the most recent items
    order = np.lexsort((pos["timestamp"].to_numpy(), u_dense))
    u_s, i_s = u_dense[order], items[order]
    ptr = np.zeros(n_users + 1, dtype=np.int64)
    np.add.at(ptr, u_s + 1, 1)
    ptr = np.cumsum(ptr)
    keep_rows, keep_cols, keep_vals = [], [], []
    for u in range(n_users):
        lo, hi = int(ptr[u]), int(ptr[u + 1])
        if hi - lo > max_items_per_user:
            lo = hi - max_items_per_user          # most recent max_items
        n = hi - lo
        w = 1.0 / np.log1p(n)
        keep_rows.append(np.full(n, u))
        keep_cols.append(i_s[lo:hi])
        if recency_floor < 1.0 and n > 1:
            ramp = np.linspace(recency_floor, 1.0, n, dtype=np.float32)
            keep_vals.append(w * ramp)
        else:
            keep_vals.append(np.full(n, w, dtype=np.float32))
    R = sp.csr_matrix(
        (np.concatenate(keep_vals),
         (np.concatenate(keep_rows), np.concatenate(keep_cols))),
        shape=(n_users, n_items), dtype=np.float32)
    print(f"[covis] R nnz={R.nnz:,} (capped histories)")

    # column degrees for normalization
    if norm == "jaccard":
        deg = np.asarray(R.sum(axis=0)).ravel()
        deg[deg == 0] = 1.0
    else:
        deg = np.asarray(R.power(2).sum(axis=0)).ravel() ** 0.5
        deg[deg == 0] = 1.0
    dinv = 1.0 / deg

    out_r, out_c, out_v = [], [], []
    Rt = R.T.tocsr()          # (items, users)
    for s in range(0, n_items, item_block):
        e = min(s + item_block, n_items)
        Cb = (Rt[s:e] @ R).tocsr()               # (blk, n_items)
        for r in range(Cb.shape[0]):
            row = Cb.getrow(r)
            idx, val = row.indices, row.data
            keep = idx != (s + r)
            idx, val = idx[keep], val[keep]
            if norm == "jaccard":
                val = val / np.maximum(deg[s + r] + deg[idx] - val, 1e-9)
            else:
                val = val * dinv[s + r] * dinv[idx]
            if len(idx) > top_k:
                sel = np.argpartition(-val, top_k)[:top_k]
                idx, val = idx[sel], val[sel]
            out_r.append(np.full(len(idx), s + r))
            out_c.append(idx); out_v.append(val)
        if (s // item_block) % 6 == 0:
            print(f"  items {s:,}-{e:,}", flush=True)

    C = sp.csr_matrix(
        (np.concatenate(out_v), (np.concatenate(out_r), np.concatenate(out_c))),
        shape=(n_items, n_items), dtype=np.float32)
    print(f"[covis] pruned nnz={C.nnz:,}")
    return C


def covis_scores(C: sp.csr_matrix, items: np.ndarray,
                 weights: np.ndarray) -> np.ndarray:
    """s(c) = sum_i w_i * C[i,c] for all items; returns (n_items,)."""
    w = np.clip(np.asarray(weights, dtype=np.float32), 0, None)
    if w.sum() == 0:
        w = np.ones(len(items), dtype=np.float32)
    return np.asarray(C[items].T @ w).ravel()
