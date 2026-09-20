"""Offline evaluation: per-retriever + full stack on the test window.

Per eval user:
  profile <- interactions with ts < T2  (weights = rating - 2.5)
  seq     <- same items ordered by timestamp (for SASRec)
  truth   <- test items with rating >= positive_threshold

run_eval returns (summary_table, details):
  details = {
    "per_user":  {method: [ {metric: val} per user ]},
    "recs":      {method: [ [item_idx] per user ]},
    "segments":  {method: {segment: {ndcg@10, recall@10, n}}},
    "funnel":    {"cand_recall@K": ..., "n_users": ...},
    "extra":     {method: {mrr, novelty, pop_exposure, long_tail, gini,
                           personalization, head_exposure}},
  }
"""
from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch

from . import metrics
from . import exp_metrics as xm
from .candidates import _topk_unseen, generate_candidates
from .config import CFG, Config
from .covis import covis_scores
from .features import ItemStore, compute_features
from .index import query as ann_query
from .profile import build_profile
from .rerank import mmr_select


def _norm_scores(d: dict[int, float], cand: np.ndarray) -> np.ndarray:
    vals = np.array([d.get(int(c), np.nan) for c in cand], dtype=np.float64)
    ok = ~np.isnan(vals)
    if ok.sum() == 0:
        return np.zeros(len(cand))
    lo, hi = np.nanmin(vals[ok]), np.nanmax(vals[ok])
    out = np.zeros(len(cand))
    if hi > lo:
        out[ok] = (vals[ok] - lo) / (hi - lo)
    return out


def eval_user(profile, truth, cand_idx, cand_meta, store, item_factors,
              item_bias, content_emb, ranker, pop_items, cfg: Config,
              k: int, covis: np.ndarray | None, sasrec_logits: np.ndarray | None,
              content_index, cf_index, covis_mat=None,
              tt_logits: np.ndarray | None = None) -> dict[str, list[int]]:
    out = {}
    seen = set(profile.items.tolist())

    out["popularity"] = [int(i) for i in pop_items if i not in seen][:k]

    d, i = ann_query(content_index, profile.content_vec[None, :], k + len(seen))
    out["content"] = [int(x) for x in i[0] if x not in seen][:k]

    d, i = ann_query(cf_index, profile.cf_vec[None, :], k + len(seen))
    out["cf"] = [int(x) for x in i[0] if x not in seen][:k]

    if covis is not None:
        idxs, _ = _topk_unseen(covis, seen, k)
        out["covis"] = idxs.tolist()[:k]

    if sasrec_logits is not None:
        idxs, _ = _topk_unseen(sasrec_logits, seen, k)
        out["sasrec"] = idxs.tolist()[:k]

    if tt_logits is not None:
        idxs, _ = _topk_unseen(tt_logits, seen, k)
        out["tt"] = idxs.tolist()[:k]

    cs = _norm_scores({c: m.get("content_sim") for c, m in cand_meta.items()}, cand_idx)
    fs = _norm_scores({c: m.get("cf_sim") for c, m in cand_meta.items()}, cand_idx)
    vs = _norm_scores({c: m.get("covis_score") for c, m in cand_meta.items()}, cand_idx)
    hy = 0.4 * cs + 0.4 * fs + 0.2 * vs
    out["union_naive"] = cand_idx[np.argsort(-hy)][:k].tolist()

    X = compute_features(profile, cand_idx, cand_meta, store,
                         content_emb=content_emb,
                         item_factors=item_factors, item_bias=item_bias,
                         covis_mat=covis_mat)
    n_f = ranker.booster_.num_feature() if hasattr(ranker, "booster_") else X.shape[1]
    rs = ranker.predict(X[:, :n_f])
    order = np.argsort(-rs)
    out["ranker"] = cand_idx[order][:k].tolist()
    top = order[:cfg.ranker_top_n]
    out["ranker_mmr"] = mmr_select(rs[top], cand_idx[top], content_emb,
                                   k, lam=cfg.mmr_lambda).tolist()
    out["ranker_mmr_div"] = mmr_select(rs[top], cand_idx[top], content_emb,
                                     k, lam=0.5).tolist()
    return out


METHODS = ["popularity", "content", "cf", "covis", "sasrec", "tt",
           "union_naive", "ranker", "ranker_mmr", "ranker_mmr_div"]

CAND_KS = (20, 50, 100, 250, 500)


def run_eval(movies, ratings_train, ratings_val, ratings_test,
             content_emb, item_factors, item_bias,
             content_index, cf_index, ranker, pop_items,
             cfg: Config = CFG, covis_mat=None, sasrec_model=None,
             n_users: int | None = None,
             k: int = 10, seed: int = 0,
             user_hist=None, truth_map=None,
             dedupe_fn=None, tt=None):
    """Evaluates against test positives. Profile = train+val (<T2).

    dedupe_fn: optional recs -> recs entity-collapse filter.
    Returns (summary, details).
    """
    store = ItemStore(movies)
    n_items = store.n

    if user_hist is None:
        hist = (ratings_train.vstack(ratings_val).sort("timestamp")
                .group_by("userId", maintain_order=True)
                .agg([pl.col("item_idx"), pl.col("rating"), pl.col("timestamp")]))
        hist_map = {int(r[0]): (np.asarray(r[1]), np.asarray(r[2]), np.asarray(r[3]))
                    for r in hist.iter_rows()}
    else:
        hist_map = user_hist
    if truth_map is None:
        truth = (ratings_test.filter(pl.col("rating") >= cfg.cf_pos_threshold)
                 .group_by("userId", maintain_order=True)
                 .agg(pl.col("item_idx")))
        truth_map = {int(r[0]): set(r[1]) for r in truth.iter_rows()}

    rng = np.random.default_rng(seed)
    eligible = []
    for u in truth_map.keys():
        items, ratings, ts = hist_map.get(u, (np.array([]), np.array([]), np.array([])))
        n_pos = int((ratings >= cfg.cf_pos_threshold).sum())
        if n_pos >= cfg.min_profile_items:
            eligible.append((u, n_pos))
    n_users = n_users or cfg.ranker_eval_users
    if len(eligible) > n_users:
        pick = rng.choice(len(eligible), n_users, replace=False)
        eligible = [eligible[i] for i in pick]

    item_pop = store.rating_count.astype(np.float64)
    head_items = set(np.argsort(-item_pop)[:int(0.05 * n_items)])

    device = ("cuda" if sasrec_model is not None
              and __import__("torch").cuda.is_available() else "cpu")

    if tt is not None:
        from .twotower import tt_user_logits
        tt_model, tt_feat, tt_emb = tt
        tt_device = "cuda" if torch.cuda.is_available() else "cpu"
        tt_model = tt_model.to(tt_device).eval()

    per_method = {m: [] for m in METHODS}
    recs_all = {m: [] for m in METHODS}
    segs = {m: {} for m in METHODS}
    cand_recall_hits, cand_tot = 0, 0
    src_hits: dict[str, np.ndarray] = {}
    print(f"[eval] users={len(eligible)}")
    for u, n_pos in eligible:
        items, ratings, ts = hist_map[u]
        prof = build_profile(items, ratings - 2.5, content_emb,
                             item_factors, store.genre_mat,
                             store.directors, store.casts,
                             store.language, store.year, timestamps=ts,
                             keywords=store.keywords, companies=store.companies)
        cov = covis_scores(covis_mat, items, ratings - 2.5) if covis_mat is not None else None
        sl = None
        if sasrec_model is not None:
            order = np.argsort(ts)
            seq = items[order]
            seq_t = np.zeros(sasrec_model.max_len, dtype=np.int64)
            tail = seq[-sasrec_model.max_len:] + 1
            seq_t[:len(tail)] = tail
            with torch.no_grad():
                sl = sasrec_model.logits(
                    torch.tensor(seq_t[None, :], device=device))[0].cpu().numpy()
        tl = tt_user_logits(tt_model, tt_feat, tt_emb, items, ratings) \
            if tt is not None else None
        cand_idx, cand_meta = generate_candidates(
            prof, content_index, cf_index, pop_items, cfg,
            covis_scores=cov, sasrec_logits=sl, tt_logits=tl)
        t = truth_map[u]
        cand_recall_hits += len(set(cand_idx.tolist()) & t)
        cand_tot += len(t)

        # per-source candidate recall @K
        seen = set(items.tolist())
        src_top: dict[str, np.ndarray] = {}
        kmax = max(CAND_KS) + len(seen)
        d_, i_ = ann_query(content_index, prof.content_vec[None, :], kmax)
        src_top["content"] = np.array([x for x in i_[0] if x not in seen])
        d_, i_ = ann_query(cf_index, prof.cf_vec[None, :], kmax)
        src_top["cf"] = np.array([x for x in i_[0] if x not in seen])
        if cov is not None:
            src_top["covis"] = _topk_unseen(cov, seen, max(CAND_KS))[0]
        if sl is not None:
            src_top["sasrec"] = _topk_unseen(sl, seen, max(CAND_KS))[0]
        if tl is not None:
            src_top["tt"] = _topk_unseen(tl, seen, max(CAND_KS))[0]
        src_top["popularity"] = np.array([i for i in pop_items if i not in seen])
        for sname, arr in src_top.items():
            hits = src_hits.setdefault(sname, np.zeros(len(CAND_KS)))
            for ki, kk in enumerate(CAND_KS):
                hits[ki] += len(set(arr[:kk].tolist()) & t) / len(t)
        uh = src_hits.setdefault("union", np.zeros(len(CAND_KS)))
        for ki, kk in enumerate(CAND_KS):
            uh[ki] += len(set(cand_idx.tolist()) & t) / len(t)
        res = eval_user(prof, t, cand_idx, cand_meta, store,
                        item_factors, item_bias, content_emb, ranker,
                        pop_items, cfg, k, cov, sl, content_index, cf_index,
                        covis_mat=covis_mat, tt_logits=tl)
        seg = xm.segment_bucket(n_pos)
        for m in METHODS:
            if m not in res:
                continue
            recs = res[m]
            if dedupe_fn:
                recs = dedupe_fn(recs)
            recs_all[m].append(recs)
            row = {}
            for kk in (5, 10, 20):
                row[f"recall@{kk}"] = metrics.recall_at_k(recs, t, kk)
                row[f"ndcg@{kk}"] = metrics.ndcg_at_k(recs, t, kk)
                row[f"precision@{kk}"] = metrics.precision_at_k(recs, t, kk)
                row[f"hitrate@{kk}"] = metrics.hit_rate_at_k(recs, t, kk)
            row["mrr"] = xm.mrr_at_k(recs, t, 20)
            row["ild"] = metrics.intra_list_diversity(recs, content_emb)
            row["novelty"] = xm.novelty(recs, item_pop)
            row["pop_exposure"] = xm.popularity_exposure(recs, item_pop)
            row["head_exposure"] = xm.head_exposure(recs, head_items)
            per_method[m].append(row)
            sb = segs[m].setdefault(seg, {"ndcg@10": 0.0, "recall@10": 0.0, "n": 0})
            sb["ndcg@10"] += row["ndcg@10"]; sb["recall@10"] += row["recall@10"]; sb["n"] += 1

    n_eval = max(len(eligible), 1)
    table = {}
    details = {"per_user": {}, "recs": recs_all, "segments": {},
               "funnel": {"cand_recall_union": cand_recall_hits / max(cand_tot, 1),
                          "n_users": len(eligible)},
               "cand_recall": {s: {f"r@{kk}": float(h[ki] / n_eval)
                                   for ki, kk in enumerate(CAND_KS)}
                               for s, h in src_hits.items()},
               "extra": {}}
    for m in METHODS:
        if not per_method[m]:
            continue
        agg = metrics.aggregate(per_method[m])
        agg["coverage"] = metrics.catalog_coverage(recs_all[m], n_items)
        agg["ild"] = float(np.nanmean([u["ild"] for u in per_method[m]]))
        agg["mrr"] = float(np.mean([u["mrr"] for u in per_method[m]]))
        agg["novelty"] = float(np.mean([u["novelty"] for u in per_method[m]]))
        agg["pop_exposure"] = float(np.mean([u["pop_exposure"] for u in per_method[m]]))
        agg["head_exposure"] = float(np.mean([u["head_exposure"] for u in per_method[m]]))
        agg["long_tail_cov"] = xm.long_tail_coverage(recs_all[m], item_pop)
        agg["gini"] = xm.gini(np.array([i for u in recs_all[m] for i in u]))
        agg["personalization"] = xm.personalization(recs_all[m], k=10)
        table[m] = agg
        details["per_user"][m] = per_method[m]
        details["segments"][m] = {
            s: {"ndcg@10": v["ndcg@10"] / v["n"],
                "recall@10": v["recall@10"] / v["n"], "n": v["n"]}
            for s, v in segs[m].items()}
    return table, details
