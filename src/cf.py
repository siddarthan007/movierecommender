"""Collaborative filtering: BPR matrix factorization in PyTorch (GPU).

Model:  score(u, i) = <p_u, q_i> + b_u + c_i + mu
Loss:   BPR  -log sigmoid(score(u,i) - score(u,j)) + L2 reg
        i = observed positive (rating >= threshold), j = sampled negative.

Trains fully on GPU with large batches; validates Recall@K each epoch on
the held-out val window and keeps the best item/user factors.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from .config import CFG, Config


class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def score(self, u, i):
        return (self.user_emb(u) * self.item_emb(i)).sum(-1) \
            + self.user_bias(u).squeeze(-1) + self.item_bias(i).squeeze(-1)

    def bpr_loss(self, u, i, j, reg):
        x = self.score(u, i) - self.score(u, j)
        l2 = (self.user_emb(u).norm(dim=-1) ** 2
              + self.item_emb(i).norm(dim=-1) ** 2
              + self.item_emb(j).norm(dim=-1) ** 2).mean()
        return -torch.nn.functional.logsigmoid(x).mean() + reg * l2


@torch.no_grad()
def recall_at_k(model: BPRMF, user_ids, train_seen, val_pos,
                n_items: int, k: int = 20, batch: int = 4096,
                device="cuda") -> float:
    """Recall@K: fraction of val positives appearing in top-K predictions."""
    model.eval()
    item_w = model.item_emb.weight
    item_b = model.item_bias.weight.squeeze(-1)
    hits, total = 0, 0
    u_list = list(user_ids)
    for s in range(0, len(u_list), batch):
        chunk = u_list[s:s + batch]
        u = torch.tensor(chunk, device=device)
        sc = model.user_emb(u) @ item_w.T + item_b.unsqueeze(0) \
            + model.user_bias(u).squeeze(-1).unsqueeze(1)
        for r, uid in enumerate(chunk):
            seen = train_seen.get(uid)
            if seen is not None and len(seen):
                sc[r, torch.tensor(list(seen), device=device)] = -1e9
        top = sc.topk(k, dim=-1).indices.cpu().numpy()
        for r, uid in enumerate(chunk):
            pos = val_pos.get(uid)
            if pos:
                hits += len(set(top[r]) & pos)
                total += len(pos)
    return hits / max(total, 1)


def train_bpr(ratings_train: pl.DataFrame, n_items: int,
              val_seen: dict | None = None, val_pos: dict | None = None,
              cfg: Config = CFG, device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pos = ratings_train.filter(pl.col("rating") >= cfg.cf_pos_threshold)
    users = pos["userId"].to_numpy()
    items = pos["item_idx"].to_numpy()

    # remap user ids to dense index
    uniq_users, u_idx = np.unique(ratings_train["userId"].to_numpy(), return_inverse=True)
    u_map = {int(u): i for i, u in enumerate(uniq_users)}
    pos_u = np.array([u_map[u] for u in users], dtype=np.int64)
    pos_i = items.astype(np.int64)
    n_users_dense = len(uniq_users)

    model = BPRMF(n_users_dense, n_items, cfg.cf_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.cf_lr)
    rng = np.random.default_rng(42)

    seen_train = None
    if val_pos is not None:
        seen_train = {}
        for u, grp in pos.group_by("userId", maintain_order=True):
            seen_train[int(u[0])] = set(grp["item_idx"].to_list())

    n_pos = len(pos_u)
    print(f"[cf] positives={n_pos:,} users={n_users_dense:,} items={n_items:,} "
          f"dim={cfg.cf_dim} device={device}")
    best_recall, best_state = -1.0, None
    val_u = [u_map[u] for u in val_pos.keys() if u in u_map] if val_pos else []
    seen_dense = {u_map[k]: v for k, v in (seen_train or {}).items() if k in u_map}

    for epoch in range(cfg.cf_epochs):
        model.train()
        order = rng.permutation(n_pos)
        tot_loss, nb = 0.0, 0
        for s in range(0, n_pos, cfg.cf_batch):
            b = order[s:s + cfg.cf_batch]
            u = torch.tensor(pos_u[b], device=device)
            i = torch.tensor(pos_i[b], device=device)
            # tile for neg_per_pos negatives
            u_r = u.repeat_interleave(cfg.cf_neg)
            i_r = i.repeat_interleave(cfg.cf_neg)
            j = torch.randint(0, n_items, (len(b) * cfg.cf_neg,), device=device)
            loss = model.bpr_loss(u_r, i_r, j, cfg.cf_reg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += loss.item(); nb += 1
        msg = f"[cf] epoch {epoch+1}/{cfg.cf_epochs} loss={tot_loss/nb:.4f}"
        if val_pos:
            r = recall_at_k(model, val_u, seen_dense, val_pos,
                            n_items, k=20, device=device)
            msg += f" val_recall@20={r:.4f}"
            if r > best_recall:
                best_recall, best_state = r, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {
        "model": model,
        "user_map": u_map,                    # raw userId -> dense idx
        "user_factors": model.user_emb.weight.detach().cpu().numpy(),
        "item_factors": model.item_emb.weight.detach().cpu().numpy(),
        "item_bias": model.item_bias.weight.detach().cpu().numpy().ravel(),
        "n_items": n_items,
        "dim": cfg.cf_dim,
    }
