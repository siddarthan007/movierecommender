"""Two-tower contrastive retriever (v12).

Item tower : concat(MiniLM384, BPR128) -> Linear -> LayerNorm -> L2 norm (256d)
User tower : attention pooling over history item embeddings, split into
             positive / negative heads (signed rating bias) -> Linear -> LN
             -> L2 norm (256d)

Training   : temporal prefix -> future positive, InfoNCE over
             [pos | in-batch targets | shared popularity negs | content-hard negs]
             with false-negative resampling against the user's full seen set.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG, Config


def tt_item_features(content_emb: np.ndarray, item_factors: np.ndarray | None,
                     n_items: int) -> np.ndarray:
    """Raw per-item input rows for both towers."""
    e = content_emb.astype(np.float32)
    if item_factors is None:
        return e
    f = item_factors.astype(np.float32)
    return np.concatenate([e, f[:n_items]], axis=1)


class ItemTower(nn.Module):
    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):                    # (..., in_dim)
        return F.normalize(self.ln(self.proj(x)), dim=-1)


class UserTower(nn.Module):
    """Two-head attention pooling: separate positive/negative history pools,
    attention logits biased by signed rating and recency position."""

    def __init__(self, dim: int):
        super().__init__()
        self.q_pos = nn.Parameter(torch.randn(dim) * 0.02)
        self.q_neg = nn.Parameter(torch.randn(dim) * 0.02)
        self.w_rel = nn.Parameter(torch.tensor(1.0))
        self.w_rec = nn.Parameter(torch.tensor(0.5))
        self.out = nn.Linear(2 * dim, dim)
        self.ln = nn.LayerNorm(dim)

    def _pool(self, h, rel, pos, mask, q):   # (B,L,D),(B,L),(B,L),(B,L),(D)
        logits = (h @ q) / np.sqrt(h.size(-1))
        logits = logits + self.w_rel * rel + self.w_rec * pos
        logits = logits.masked_fill(~mask, -1e9)
        a = torch.softmax(logits, dim=1)
        a = torch.where(mask.any(1, keepdim=True), a, torch.zeros_like(a))
        return (a.unsqueeze(-1) * h).sum(1)  # (B,D)

    def forward(self, h, rel, pos, mask, neg_mask):
        u_pos = self._pool(h, rel, pos, mask & ~neg_mask, self.q_pos)
        u_neg = self._pool(h, rel, pos, mask & neg_mask, self.q_neg)
        u = self.ln(self.out(torch.cat([u_pos, u_neg], dim=-1)))
        return F.normalize(u, dim=-1)


class TwoTower(nn.Module):
    def __init__(self, item_in_dim: int, dim: int = 256):
        super().__init__()
        self.item_tower = ItemTower(item_in_dim, dim)
        self.user_tower = UserTower(dim)
        self.tau = 0.1                     # fixed InfoNCE temperature

    def encode_items(self, feat):            # (..., F) -> (..., d)
        return self.item_tower(feat)

    def encode_user(self, feat_hist, rel, pos, mask, neg_mask):
        h = self.item_tower(feat_hist)       # (B,L,d)
        return self.user_tower(h, rel, pos, mask, neg_mask)


# --------------------------- training data -------------------------------

def build_tt_examples(items_u: np.ndarray, rts_u: np.ndarray,
                      threshold: float, snaps: int, min_hist: int,
                      rng: np.random.Generator) -> list[tuple[int, int]]:
    """Return list of (cut, target_pos) index pairs into the user's sorted arrays.

    items_u/rts_u: the user's full history sorted by timestamp.
    A cut must leave >= min_hist items before it and >= 1 positive after it.
    """
    n = len(items_u)
    pos_idx = np.where(rts_u >= threshold)[0]
    if len(pos_idx) < 2:
        return []
    cuts = rng.integers(min_hist, n, size=snaps)
    out = []
    for c in np.unique(cuts):
        fut = pos_idx[pos_idx >= c]
        if len(fut) == 0:
            continue
        out.append((int(c), int(fut[rng.integers(len(fut))])))
    return out


def train_twotower(ratings_train, item_feat: np.ndarray, cfg: Config = CFG,
                   dim: int = 256, epochs: int = 3, batch: int = 1024,
                   lr: float = 2e-3, snaps: int = 4, max_len: int = 100,
                   pop_neg: int = 512, hard_neg: int = 8,
                   content_nn: np.ndarray | None = None,
                   pop_items: np.ndarray | None = None,
                   device: str | None = None) -> dict:
    """ratings_train: polars df [userId, item_idx, rating, timestamp]."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_items, in_dim = item_feat.shape
    import polars as pl

    grp = (ratings_train.sort("timestamp")
           .group_by("userId", maintain_order=True)
           .agg([pl.col("item_idx"), pl.col("rating")]))
    users, ex_items, ex_rts = [], [], []
    examples = []                                # (uid_row, cut, target)
    rng = np.random.default_rng(0)
    for r_i, (u, it, rt) in enumerate(grp.iter_rows()):
        it = np.asarray(it); rt = np.asarray(rt, dtype=np.float32)
        if len(it) < cfg.min_user_ratings + 1:
            continue
        pairs = build_tt_examples(it, rt, cfg.cf_pos_threshold, snaps,
                                  cfg.min_profile_items, rng)
        if not pairs:
            continue
        users.append(int(u))
        ex_items.append(it); ex_rts.append(rt)
        for cut, tgt in pairs:
            examples.append((len(users) - 1, cut, tgt))
    print(f"[tt] users={len(users):,} examples={len(examples):,}")

    # seen-set per user row for false-negative resampling
    seen_sets = [set(it.tolist()) for it in ex_items]

    if pop_items is None:
        pop_items = np.arange(n_items)
    pop_items = np.asarray(pop_items)

    feat_t = torch.tensor(item_feat, device=device)
    model = TwoTower(in_dim, dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    ex = np.asarray(examples, dtype=np.int64)
    order = np.arange(len(ex))
    for ep in range(epochs):
        rng.shuffle(order)
        tot = 0.0
        nb = 0
        for s in range(0, len(order), batch):
            bi = order[s:s + batch]
            rows = ex[bi]
            B = len(rows)
            L = min(max_len, max(int(r[1]) for r in rows))
            fh = np.zeros((B, L, in_dim), dtype=np.float32)
            rel = np.zeros((B, L), dtype=np.float32)
            pos = np.zeros((B, L), dtype=np.float32)
            mask = np.zeros((B, L), dtype=bool)
            negm = np.zeros((B, L), dtype=bool)
            tgt = np.zeros(B, dtype=np.int64)
            for b, (ur, cut, t_) in enumerate(rows):
                it = ex_items[ur]; rt = ex_rts[ur]
                lo = max(0, cut - L)
                seq_i = it[lo:cut]; seq_r = rt[lo:cut]
                m = len(seq_i)
                fh[b, :m] = item_feat[seq_i]
                rel[b, :m] = seq_r - 3.0
                pos[b, :m] = np.linspace(0, 1, m) if m > 1 else 1.0
                mask[b, :m] = True
                negm[b, :m] = seq_r < cfg.cf_pos_threshold
                tgt[b] = it[t_]
            u = model.encode_user(
                torch.tensor(fh, device=device),
                torch.tensor(rel, device=device),
                torch.tensor(pos, device=device),
                torch.tensor(mask, device=device),
                torch.tensor(negm, device=device))
            v_pos = model.encode_items(feat_t[torch.tensor(tgt, device=device)])

            # shared popularity negatives
            neg_ids = pop_items[rng.integers(0, len(pop_items),
                                             size=(B, pop_neg))]
            # per-example content-hard negatives (top-32 nn of target)
            if content_nn is not None and hard_neg:
                hn_pool = content_nn[tgt, :32]
                hn = hn_pool[np.arange(B)[:, None],
                             rng.integers(0, hn_pool.shape[1],
                                          size=(B, hard_neg))]
                neg_ids = np.concatenate([neg_ids, hn], axis=1)
            # false-negative guard: resample collisions with user seen set
            # (single pass — residual collisions are rare and harmless)
            for b, (ur, _, _t) in enumerate(rows):
                seen = seen_sets[ur]
                bad = np.fromiter((x in seen for x in neg_ids[b]),
                                  dtype=bool, count=neg_ids.shape[1])
                if bad.any():
                    neg_ids[b][bad] = pop_items[rng.integers(
                        0, len(pop_items), size=int(bad.sum()))]
            v_neg = model.encode_items(feat_t[torch.tensor(neg_ids, device=device)])

            pos_logit = (u * v_pos).sum(1, keepdim=True)          # (B,1)
            inbatch = u @ v_pos.T                                # (B,B)
            inbatch.fill_diagonal_(float("-inf"))
            neg_logit = torch.einsum("bd,bnd->bn", u, v_neg)     # (B,Ns)
            logits = torch.cat([pos_logit, inbatch, neg_logit], 1) / model.tau
            loss = F.cross_entropy(logits, torch.zeros(B, dtype=torch.long,
                                                       device=device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        print(f"[tt] epoch {ep + 1} loss={tot / max(nb, 1):.4f}",
              flush=True)

    with torch.no_grad():
        chunks = [model.encode_items(feat_t[s:s + 8192])
                  for s in range(0, n_items, 8192)]
        item_emb = torch.cat(chunks).cpu().numpy().astype(np.float32)
    return {"model": model.cpu(), "item_emb": item_emb,
            "in_dim": in_dim, "dim": dim}


def tt_user_vec(model: TwoTower, item_feat: np.ndarray,
                items: np.ndarray, ratings: np.ndarray,
                max_len: int = 100) -> np.ndarray:
    """256-d user vector for one history (items sorted chronologically)."""
    it = np.asarray(items, dtype=np.int64)[-max_len:]
    rt = np.asarray(ratings, dtype=np.float32)[-max_len:]
    rel = rt - 3.0
    pos = np.linspace(0, 1, len(it)).astype(np.float32) if len(it) > 1 \
        else np.ones(1, dtype=np.float32)
    mask = np.ones(len(it), dtype=bool)
    negm = rt < CFG.cf_pos_threshold
    dev = next(model.parameters()).device
    with torch.no_grad():
        u = model.encode_user(
            torch.tensor(item_feat[it][None], device=dev),
            torch.tensor(rel[None], device=dev),
            torch.tensor(pos[None], device=dev),
            torch.tensor(mask[None], device=dev),
            torch.tensor(negm[None], device=dev))
    return u[0].cpu().numpy()


def tt_user_logits(model: TwoTower, item_feat: np.ndarray,
                   tt_item_emb: np.ndarray,
                   items: np.ndarray, ratings: np.ndarray,
                   max_len: int = 100) -> np.ndarray:
    """Dense score vector (n_items,) = user_vec . item_emb."""
    u = tt_user_vec(model, item_feat, items, ratings, max_len)
    return tt_item_emb @ u
