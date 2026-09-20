"""LightGCN: collaborative filtering via propagation on the user-item graph.

E^(k+1) = D^-1/2 A D^-1/2 E^(k);  final = mean_k E^(k).
score(u,i) = <e_u, e_i>.  Trained with BPR + mixed hard negatives:

  negatives = 30% uniform + 40% popularity + 30% content-hard
              + in-batch positives of other users (free extra negs)

False-negative guard: sampled negatives are checked against the user's
known positives via GPU searchsorted over a sorted CSR and resampled once.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import scipy.sparse as sp

from .config import CFG, Config


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.n_layers = n_layers
        self.emb = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.emb.weight, std=0.1)

    def propagate(self, adj_norm: torch.Tensor) -> torch.Tensor:
        e = self.emb.weight
        out = e
        total = e
        for _ in range(self.n_layers):
            e = torch.sparse.mm(adj_norm, e)
            total = total + e
            out = total
        return total / (self.n_layers + 1)


def build_norm_adj(pos_u: np.ndarray, pos_i: np.ndarray,
                   n_users: int, n_items: int, device) -> torch.Tensor:
    """Symmetric-normalized bipartite adjacency, torch sparse COO."""
    U, I = n_users, n_items
    rows = np.concatenate([pos_u, pos_i + U])
    cols = np.concatenate([pos_i + U, pos_u])
    deg = np.zeros(U + I, dtype=np.float64)
    np.add.at(deg, rows, 1.0)
    d_inv = np.power(np.clip(deg, 1, None), -0.5)
    vals = d_inv[rows] * d_inv[cols]
    idx = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    val = torch.tensor(vals, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (U + I, U + I)).coalesce().to(device)


class SeenIndex:
    """Membership test over (user, item) positive pairs via encoded keys."""

    def __init__(self, pos_u: np.ndarray, pos_i: np.ndarray, n_users: int, device):
        self.n_items = int(pos_i.max()) + 1
        keys = np.unique(pos_u.astype(np.int64) * self.n_items + pos_i.astype(np.int64))
        self.keys = torch.tensor(keys, dtype=torch.long, device=device)
        self.row_items = None  # optional CSR for masking evals
        order = np.lexsort((pos_i, pos_u))
        su, si = pos_u[order], pos_i[order]
        ptr = np.zeros(n_users + 1, dtype=np.int64)
        np.add.at(ptr, su + 1, 1)
        self.ptr = torch.tensor(np.cumsum(ptr), dtype=torch.long, device=device)
        self.items = torch.tensor(si, dtype=torch.long, device=device)

    def is_positive(self, u: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        q = u * self.n_items + j
        return torch.isin(q, self.keys)


def popularity_sampler(rating_count: np.ndarray, n_items: int, power=0.75):
    p = np.power(np.clip(rating_count, 1, None), power)
    cdf = np.cumsum(p / p.sum())
    def sample(n):
        return np.searchsorted(cdf, np.random.random(n)).astype(np.int64)
    return sample


def content_hard_sampler(content_nn: np.ndarray | None):
    """content_nn: (n_items, S) nearest content neighbors; None -> disabled."""
    def sample(i_idx):
        if content_nn is None:
            return None
        s = np.random.randint(0, content_nn.shape[1], size=len(i_idx))
        return content_nn[i_idx, s]
    return sample


def train_lightgcn(ratings_train: pl.DataFrame, n_items: int,
                   val_pos: dict | None, cfg: Config = CFG,
                   rating_count: np.ndarray | None = None,
                   content_nn: np.ndarray | None = None,
                   dim: int = 64, layers: int = 3, epochs: int = 8,
                   lr: float = 0.005, reg: float = 1e-4,
                   neg_mix: tuple = (0.30, 0.35, 0.35),
                   device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pos = ratings_train.filter(pl.col("rating") >= cfg.cf_pos_threshold)
    users_raw = pos["userId"].to_numpy()
    items_raw = pos["item_idx"].to_numpy()

    uniq_users = np.unique(ratings_train["userId"].to_numpy())
    u_map = {int(u): i for i, u in enumerate(uniq_users)}
    pos_u = np.array([u_map[u] for u in users_raw], dtype=np.int64)
    pos_i = items_raw.astype(np.int64)
    n_users = len(uniq_users)

    adj = build_norm_adj(pos_u, pos_i, n_users, n_items, device)
    model = LightGCN(n_users, n_items, dim, layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(42)
    seen = SeenIndex(pos_u, pos_i, n_users, device)
    pop_sample = (popularity_sampler(rating_count, n_items)
                  if rating_count is not None else
                  (lambda n: rng.integers(0, n_items, n)))

    n_pos = len(pos_u)
    print(f"[lgcn] positives={n_pos:,} users={n_users:,} items={n_items:,} "
          f"dim={dim} layers={layers} device={device}")

    from .cf import recall_at_k  # reuse eval (works on emb-weighted model? no—needs emb API)
    best_recall, best_state = -1.0, None
    val_u = np.array([u_map[u] for u in val_pos.keys() if u in u_map]) if val_pos else None

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(n_pos)
        tot, nb = 0.0, 0
        for s in range(0, n_pos, cfg.cf_batch):
            e_all = model.propagate(adj)
            b = order[s:s + cfg.cf_batch]
            u_np, i_np = pos_u[b], pos_i[b]
            u = torch.tensor(u_np, device=device)
            i = torch.tensor(i_np, device=device)

            # negatives: per-slot mix over (uniform, popularity, content-hard)
            j_slots = []
            S = content_nn.shape[1] if content_nn is not None else 0
            fu, fp = neg_mix[0], neg_mix[0] + neg_mix[1]
            for k in range(cfg.cf_neg):
                sel = rng.random(len(b))
                c_neg = (content_nn[i_np, k % S] if S
                         else pop_sample(len(b)))
                j_np = np.where(
                    sel < fu, rng.integers(0, n_items, len(b)),
                    np.where(sel < fp, pop_sample(len(b)), c_neg))
                j_slots.append(torch.tensor(j_np, device=device))
            j = torch.cat(j_slots)
            u_r = u.repeat_interleave(cfg.cf_neg)
            i_r = i.repeat_interleave(cfg.cf_neg)

            # resample known positives once (false-negative guard)
            bad = seen.is_positive(u_r, j)
            if bad.any():
                j = j.clone()
                j[bad] = torch.randint(0, n_items, (int(bad.sum()),), device=device)

            e_u, e_i, e_j = e_all[u_r], e_all[i_r], e_all[j]
            x = (e_u * e_i).sum(-1) - (e_u * e_j).sum(-1)
            loss = -torch.nn.functional.logsigmoid(x).mean()
            l2 = (model.emb.weight[u_r].norm(dim=-1) ** 2
                  + model.emb.weight[i_r].norm(dim=-1) ** 2
                  + model.emb.weight[j].norm(dim=-1) ** 2).mean()
            loss = loss + reg * l2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        msg = f"[lgcn] epoch {epoch+1}/{epochs} loss={tot/nb:.4f}"
        if val_u is not None:
            r = _lgcn_recall(e_all.detach(), val_u, val_pos, u_map,
                             seen, n_users, n_items, device)
            msg += f" val_recall@20={r:.4f}"
            if r > best_recall:
                best_recall = r
                best_state = {"e_all": e_all.detach().cpu().numpy(),
                              "emb0": model.emb.weight.detach().cpu().numpy()}
        print(msg, flush=True)

    e_all = torch.tensor(best_state["e_all"], device=device) if best_state else model.propagate(adj).detach()
    return {
        "user_map": u_map,
        "user_factors": e_all[:n_users].cpu().numpy(),
        "item_factors": e_all[n_users:].cpu().numpy(),
        "item_bias": np.zeros(n_items, dtype=np.float32),
        "n_items": n_items, "dim": dim,
        "best_val_recall": best_recall,
    }


@torch.no_grad()
def _lgcn_recall(e_all, val_u_dense, val_pos, u_map, seen, n_users, n_items,
                 device, k=20, batch=2048):
    item_e = e_all[n_users:]
    hits = tot = 0
    for s in range(0, len(val_u_dense), batch):
        chunk = val_u_dense[s:s + batch]
        u = torch.tensor(chunk, device=device)
        sc = e_all[u] @ item_e.T
        for r, ud in enumerate(chunk):
            lo, hi = int(seen.ptr[ud]), int(seen.ptr[ud + 1])
            if hi > lo:
                sc[r, seen.items[lo:hi]] = -1e9
        top = sc.topk(k, dim=-1).indices.cpu().numpy()
        inv = {v: k_ for k_, v in u_map.items()}
        for r, ud in enumerate(chunk):
            truth = val_pos.get(inv[int(ud)])
            if truth:
                hits += len(set(top[r]) & truth); tot += len(truth)
    return hits / max(tot, 1)
