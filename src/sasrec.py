"""SASRec: self-attentive sequential recommendation (causal Transformer).

Sequences = user history ordered by timestamp (last `max_len` items).
Training: at every position predict the next item with sampled-softmax CE
(uniform + popularity negatives; all positions contribute).
Inference: encode prefix -> score all items -> top-k retrieval signal.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import CFG, Config


class SASRec(nn.Module):
    def __init__(self, n_items: int, dim: int = 64, max_len: int = 50,
                 n_heads: int = 2, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.n_items, self.max_len = n_items, max_len
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)  # 0 = pad
        self.pos_emb = nn.Embedding(max_len, dim)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.out_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, L) item ids, 0-padded RIGHT. Causal mask alone suffices:
        pad positions sit in the future of every real item and are never
        attended to (avoids pad-mask NaN propagation)."""
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        h = self.drop(self.item_emb(seq) + self.pos_emb(pos))
        return self.out_norm(self.encoder(h, mask=self.causal_mask[:L, :L]))

    def logits(self, seq: torch.Tensor) -> torch.Tensor:
        """Next-item logits from the last REAL position: (B, n_items)."""
        h = self.encode(seq)
        lengths = (seq != 0).sum(1).clamp(min=1) - 1
        last = h[torch.arange(seq.size(0), device=seq.device), lengths]
        return last @ self.item_emb.weight[1:].T


def build_sequences(ratings: pl.DataFrame, max_len: int, min_len: int = 4):
    """Per-user item sequence sorted by ts -> padded (n_users, max_len) int64.

    Returns (seqs, user_ids). Item ids stored as item_idx + 1 (0 = pad).
    """
    df = ratings.sort("timestamp")
    seqs, uids = [], []
    for u, grp in df.group_by("userId", maintain_order=True):
        it = grp["item_idx"].to_numpy()
        if len(it) < min_len:
            continue
        seqs.append(it[-max_len:] + 1)
        uids.append(int(u[0]))
    out = np.zeros((len(seqs), max_len), dtype=np.int64)
    for r, s in enumerate(seqs):
        out[r, :len(s)] = s
    return out, np.array(uids)


def train_sasrec(ratings_train: pl.DataFrame, n_items: int,
                 val_seq: np.ndarray | None, val_pos: dict | None,
                 cfg: Config = CFG, dim: int = 64, max_len: int = 50,
                 epochs: int = 6, lr: float = 1e-3, batch: int = 256,
                 n_neg: int = 256, device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seqs, uids = build_sequences(ratings_train, max_len, min_len=cfg.min_profile_items)
    model = SASRec(n_items, dim, max_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    n = len(seqs)
    print(f"[sasrec] users={n:,} max_len={max_len} dim={dim} device={device}")

    best, best_state = -1.0, None
    # validation: context = train sequence of users that have val positives
    vseq = vuids = None
    if val_pos:
        val_set = set(val_pos.keys())
        vmask = np.array([int(u) in val_set for u in uids])
        vseq, vuids = seqs[vmask], uids[vmask]
        print(f"[sasrec] val users={vmask.sum():,}")
    for ep in range(epochs):
        model.train()
        order = rng.permutation(n)
        tot, nb = 0.0, 0
        for s in range(0, n, batch):
            b = order[s:s + batch]
            seq = torch.tensor(seqs[b], device=device)
            h = model.encode(seq)                      # (B,L,D)
            B, L, D = h.shape
            # target at pos t = seq[t+1]; last pos target = its own next (unknown->skip)
            tgt = torch.zeros(B, L, dtype=torch.long, device=device)
            tgt[:, :-1] = seq[:, 1:]
            mask = (tgt > 0) & (seq != 0)
            h_f = h[mask]                              # (M,D)
            t_f = tgt[mask] - 1                        # shift to 0..n_items-1
            neg = torch.randint(0, n_items, (len(t_f), n_neg), device=device)
            it_w = model.item_emb.weight[1:]
            s_pos = (h_f * it_w[t_f]).sum(-1)
            # chunked sampled-softmax to bound VRAM
            neg_chunks = []
            for hc, nc in zip(h_f.split(8192), neg.split(8192)):
                neg_chunks.append(torch.einsum("md,mnd->mn", hc, it_w[nc]))
            s_neg = torch.cat(neg_chunks)
            all_s = torch.cat([s_pos[:, None], s_neg], dim=1)
            loss = (-s_pos + torch.logsumexp(all_s, dim=1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        msg = f"[sasrec] epoch {ep+1}/{epochs} loss={tot/nb:.4f}"
        if vseq is not None:
            r = sasrec_recall(model, vseq, val_pos, vuids, n_items, device)
            msg += f" val_recall@20={r:.4f}"
            if r > best:
                best = r
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
        print(msg, flush=True)

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return {"model": model, "seq_user_ids": uids, "max_len": max_len,
            "dim": dim, "item_emb": model.item_emb.weight.detach().cpu().numpy()}


@torch.no_grad()
def sasrec_recall(model: SASRec, seqs: np.ndarray, val_pos: dict,
                  seq_uids: np.ndarray, n_items: int, device,
                  seen_seq: np.ndarray | None = None,
                  k: int = 20, batch: int = 512) -> float:
    model.eval()
    hits = tot = 0
    for s in range(0, len(seqs), batch):
        seq = torch.tensor(seqs[s:s + batch], device=device)
        sc = model.logits(seq)                          # (B,n_items)
        for r in range(seq.size(0)):
            seen_items = seq[r][seq[r] > 0] - 1
            sc[r, seen_items] = -1e9
        top = sc.topk(k, dim=-1).indices.cpu().numpy()
        for r in range(seq.size(0)):
            truth = val_pos.get(int(seq_uids[s + r]))
            if truth:
                hits += len(set(top[r]) & truth); tot += len(truth)
    return hits / max(tot, 1)


def sasrec_score_items(model: SASRec, seq_items: np.ndarray,
                       cand: np.ndarray, max_len: int, device) -> np.ndarray:
    """Score arbitrary candidate items given an item sequence (serving)."""
    model.eval()
    seq = np.zeros(max_len, dtype=np.int64)
    tail = np.asarray(seq_items, dtype=np.int64)[-max_len:] + 1
    seq[:len(tail)] = tail
    with torch.no_grad():
        lg = model.logits(torch.tensor(seq[None, :], device=device))[0]
        return lg[torch.tensor(np.asarray(cand), device=device)].cpu().numpy()
