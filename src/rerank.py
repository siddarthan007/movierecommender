"""MMR diversification:  MMR(d) = lam*rel(d) - (1-lam)*max_sim(d, S)."""
from __future__ import annotations

import numpy as np


def mmr_select(scores: np.ndarray, cand_idx: np.ndarray,
               emb: np.ndarray, k: int, lam: float = 0.7) -> np.ndarray:
    """Greedy MMR over candidates. emb must be L2-normalized fp32."""
    order = np.argsort(-scores)
    cand_idx = np.asarray(cand_idx)
    selected: list[int] = []
    selected_local: list[int] = []
    remaining = list(range(len(order)))
    pos_of = {j: i for i, j in enumerate(order)}  # global cand pos -> rank pos

    while remaining and len(selected) < k:
        best_pos, best_val = None, -np.inf
        for rp in remaining:
            g = order[rp]
            if selected_local:
                sim = emb[cand_idx[g]] @ emb[cand_idx[selected_local]].T
                max_sim = float(sim.max())
            else:
                max_sim = 0.0
            val = lam * scores[g] - (1 - lam) * max_sim
            if val > best_val:
                best_val, best_pos = val, rp
        g = order[best_pos]
        selected.append(int(cand_idx[g]))
        selected_local.append(int(g))
        remaining.remove(best_pos)
    return np.array(selected, dtype=np.int64)
