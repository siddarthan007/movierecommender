"""Pipeline orchestrator.

Steps (each cached on disk; run subset with --steps):
  etl      raw CSV -> movies.parquet + ratings.parquet
  embed    canonical text -> content.f16.npy (GPU MiniLM)
  cf       MovieLens train -> BPR factors (GPU) baseline
  lightgcn MovieLens train -> LightGCN factors (GPU) + hard negatives
  covis    item-item co-visitation graph -> covis.npz
  sasrec   sequential model -> sasrec.pt (GPU)
  index    FAISS HNSW for content emb + CF item factors + pop universe
  ranker   candidate gen + features -> LightGBM LambdaRank
  eval     honest test-window metrics for all retrievers + stack
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
import torch

from .config import CFG
from .ranker import grade


def _t(msg):
    print(f"\n===== {msg} =====", flush=True)
    return time.time()


def step_etl(cfg):
    from .etl import run_etl
    t = _t("ETL")
    run_etl(cfg)
    print(f"etl done in {time.time()-t:.0f}s")


def step_embed(cfg):
    from .embeddings import run_embeddings
    t = _t("EMBEDDINGS")
    movies = pl.read_parquet(cfg.processed_path / "movies.parquet")
    run_embeddings(movies["text"].to_list(), cfg)
    print(f"embed done in {time.time()-t:.0f}s")


def _ratings(cfg):
    return pl.read_parquet(cfg.processed_path / "ratings.parquet")


def _n_items(cfg):
    return pl.read_parquet(cfg.processed_path / "movies.parquet").height


def step_cf(cfg):
    from .cf import train_bpr
    from .split import temporal_split, positive_items
    t = _t("CF / BPR (baseline)")
    ratings = _ratings(cfg)
    train, val, test, t1, t2 = temporal_split(ratings, cfg)
    np.savez(cfg.processed_path / "splits.npz", t1=t1, t2=t2)
    val_pos = positive_items(val, cfg.cf_pos_threshold)
    res = train_bpr(train, _n_items(cfg), val_pos=val_pos, cfg=cfg)
    np.save(cfg.models_path / "bpr_item_factors.npy", res["item_factors"])
    np.save(cfg.models_path / "item_bias.npy", res["item_bias"])
    print(f"cf done in {time.time()-t:.0f}s")


def _content_nn(cfg, k=32):
    """Top-k content neighbors per item (excl self) for hard negatives."""
    from .index import build_hnsw, query
    p = cfg.models_path / "content_nn.npy"
    if p.exists():
        return np.load(p)
    emb = np.load(cfg.emb_path / "content.f16.npy").astype(np.float32)
    idx = build_hnsw(emb, cfg)
    _, I = query(idx, emb, k + 1)
    nn = np.zeros((emb.shape[0], k), dtype=np.int32)
    for r in range(emb.shape[0]):
        row = I[r][I[r] != r][:k]
        nn[r, :len(row)] = row
        if len(row) < k:
            nn[r, len(row):] = row[-1] if len(row) else r
    np.save(p, nn)
    return nn


def step_lightgcn(cfg, neg_mix=(0.30, 0.35, 0.35), epochs=8, tag=""):
    from .gnn import train_lightgcn
    from .split import temporal_split, positive_items
    t = _t("LightGCN")
    ratings = _ratings(cfg)
    movies = pl.read_parquet(cfg.processed_path / "movies.parquet")
    train, val, test, t1, t2 = temporal_split(ratings, cfg)
    val_pos = positive_items(val, cfg.cf_pos_threshold)
    content_nn = _content_nn(cfg, 32)
    res = train_lightgcn(train, movies.height, val_pos, cfg,
                         rating_count=movies["rating_count"].to_numpy(),
                         content_nn=content_nn,
                         dim=64, layers=3, epochs=epochs, lr=0.005, reg=1e-4,
                         neg_mix=neg_mix)
    if tag:
        np.save(cfg.models_path / f"lgcn_{tag}_item_factors.npy",
                res["item_factors"])
        return res
    np.save(cfg.models_path / "item_factors.npy", res["item_factors"])
    np.save(cfg.models_path / "user_factors.npy", res["user_factors"])
    joblib.dump(res["user_map"], cfg.models_path / "user_map.joblib")
    print(f"lgcn done in {time.time()-t:.0f}s best_val_recall={res['best_val_recall']:.4f}")


def step_covis(cfg, recency_floor: float = 1.0, norm: str = "cosine",
               out_name: str = "covis.npz"):
    from .covis import build_covis
    from .split import temporal_split
    t = _t("COVIS")
    ratings = _ratings(cfg)
    train, val, test, t1, t2 = temporal_split(ratings, cfg)
    C = build_covis(train, _n_items(cfg), cfg.cf_pos_threshold, top_k=100,
                    recency_floor=recency_floor, norm=norm)
    sp.save_npz(cfg.models_path / out_name, C)
    print(f"covis({out_name}) done in {time.time()-t:.0f}s nnz={C.nnz:,}")


def step_twotower(cfg, tag: str = "", epochs: int = 3, snaps: int = 4,
                  hard_neg: int = 8):
    from .split import temporal_split
    from .twotower import train_twotower, tt_item_features
    t = _t("TWO-TOWER")
    movies_pd, emb, fac, bias, ratings = _load_stack(cfg)
    train, val, test, t1, t2 = temporal_split(ratings, cfg)
    item_feat = tt_item_features(emb, fac, len(movies_pd))
    content_nn = _content_nn(cfg, k=32) if hard_neg else None
    rc = movies_pd["rating_count"].to_numpy()
    pop_items = np.argsort(-rc)[:5000]          # popularity-negative pool
    res = train_twotower(train, item_feat, cfg, epochs=epochs, snaps=snaps,
                         hard_neg=hard_neg, content_nn=content_nn,
                         pop_items=pop_items)
    suf = f"_{tag}" if tag else ""
    torch.save({"state": res["model"].state_dict(),
                "in_dim": res["in_dim"], "dim": res["dim"]},
               cfg.models_path / f"twotower{suf}.pt")
    np.save(cfg.models_path / f"tt_item_emb{suf}.npy", res["item_emb"])
    print(f"twotower done in {time.time()-t:.0f}s")


def _load_tt(cfg, ov=None):
    """Return (model, item_feat, tt_item_emb) or None."""
    from .twotower import TwoTower, tt_item_features
    ov = ov or {}
    mp = Path(ov.get("tt_model", cfg.models_path / "twotower.pt"))
    ep = Path(ov.get("tt_emb", cfg.models_path / "tt_item_emb.npy"))
    if not (mp.exists() and ep.exists()):
        return None
    ck = torch.load(mp, map_location="cpu", weights_only=True)
    model = TwoTower(int(ck["in_dim"]), int(ck["dim"]))
    model.load_state_dict(ck["state"])
    model.eval()
    movies_pd, emb, fac, bias, ratings = _load_stack(cfg)
    feat = tt_item_features(emb, fac, len(movies_pd))
    return model, feat, np.load(ep)


def step_sasrec(cfg):
    from .sasrec import build_sequences, train_sasrec
    from .split import temporal_split, positive_items
    t = _t("SASRec")
    ratings = _ratings(cfg)
    train, val, test, t1, t2 = temporal_split(ratings, cfg)
    # val sequences: train items as context (ordered), label = val positives
    val_pos = positive_items(val, cfg.cf_pos_threshold)
    res = train_sasrec(train, _n_items(cfg), None, val_pos, cfg,
                       dim=64, max_len=50, epochs=6)
    torch.save(res["model"].state_dict(), cfg.models_path / "sasrec.pt")
    np.save(cfg.models_path / "sasrec_item_emb.npy", res["item_emb"])
    joblib.dump({"seq_user_ids": res["seq_user_ids"],
                 "max_len": res["max_len"], "dim": res["dim"]},
                cfg.models_path / "sasrec_meta.joblib")
    print(f"sasrec done in {time.time()-t:.0f}s")


def step_index(cfg):
    from .index import build_hnsw, save_index
    t = _t("FAISS INDEXES")
    emb = np.load(cfg.emb_path / "content.f16.npy").astype(np.float32)
    save_index(build_hnsw(emb, cfg), cfg.models_path / "content_hnsw.faiss")
    fac_src = cfg.models_path / "item_factors.npy"
    if not fac_src.exists():
        fac_src = cfg.models_path / "bpr_item_factors.npy"
    fac = np.load(fac_src).astype(np.float32)
    save_index(build_hnsw(fac, cfg), cfg.models_path / "cf_hnsw.faiss")
    movies = pl.read_parquet(cfg.processed_path / "movies.parquet")
    from .candidates import popularity_universe
    pop = popularity_universe(movies["rating_count"].to_numpy(),
                              movies["bayes_rating"].to_numpy(), cfg.popularity_k)
    np.save(cfg.models_path / "pop_items.npy", pop)
    print(f"index done in {time.time()-t:.0f}s (cf source: {fac_src.name})")


def _load_stack(cfg):
    movies_pd = pd.read_parquet(cfg.processed_path / "movies.parquet")
    emb = np.load(cfg.emb_path / "content.f16.npy").astype(np.float32)
    n = np.linalg.norm(emb, axis=1, keepdims=True); n[n == 0] = 1
    emb = emb / n
    fac_p = cfg.models_path / "item_factors.npy"
    if not fac_p.exists():
        fac_p = cfg.models_path / "bpr_item_factors.npy"
    fac = np.load(fac_p).astype(np.float32)
    fn = np.linalg.norm(fac, axis=1, keepdims=True); fn[fn == 0] = 1
    fac = fac / fn
    bias_p = cfg.models_path / "item_bias.npy"
    bias = np.load(bias_p).astype(np.float32) if bias_p.exists() \
        else np.zeros(len(movies_pd), dtype=np.float32)
    ratings = _ratings(cfg)
    return movies_pd, emb, fac, bias, ratings


def _load_sasrec(cfg):
    meta_p = cfg.models_path / "sasrec_meta.joblib"
    if not meta_p.exists():
        return None
    from .sasrec import SASRec
    meta = joblib.load(meta_p)
    model = SASRec(_n_items(cfg), meta["dim"], meta["max_len"])
    model.load_state_dict(torch.load(cfg.models_path / "sasrec.pt",
                                     map_location="cpu"))
    model.eval()
    return model, meta


def step_ranker(cfg, covis_path=None, out_name: str = "ranker.joblib",
                snapshots: tuple = (1.0,), tt=None):
    from .candidates import generate_candidates
    from .covis import covis_scores
    from .twotower import tt_user_logits
    from .features import ItemStore, compute_features
    from .index import load_index
    from .profile import build_profile
    from .ranker import assemble, train_ranker, importance_frame
    from .split import temporal_split
    t = _t("RANKER DATA")
    movies_pd, emb, fac, bias, ratings = _load_stack(cfg)
    store = ItemStore(movies_pd)
    train, val, test, t1, t2 = temporal_split(ratings, cfg)
    content_index = load_index(cfg.models_path / "content_hnsw.faiss", cfg)
    cf_index = load_index(cfg.models_path / "cf_hnsw.faiss", cfg)
    pop_items = np.load(cfg.models_path / "pop_items.npy")
    covis_p = covis_path or cfg.models_path / "covis.npz"
    covis_mat = sp.load_npz(covis_p) if Path(covis_p).exists() else None
    sasrec = _load_sasrec(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if sasrec:
        sasrec[0].to(device)
    if tt is not None:
        tt[0].to(device)

    hist = (train.group_by("userId", maintain_order=True)
            .agg([pl.col("item_idx"), pl.col("rating"), pl.col("timestamp")]))
    val_lab = (val.group_by("userId", maintain_order=True)
               .agg([pl.col("item_idx"), pl.col("rating")]))
    hist_map = {int(r[0]): (np.asarray(r[1]), np.asarray(r[2]), np.asarray(r[3]))
                for r in hist.iter_rows()}
    lab_map = {int(r[0]): dict(zip(r[1], r[2]))
               for r in val_lab.iter_rows()}

    eligible = [u for u, l in lab_map.items()
                if u in hist_map
                and (hist_map[u][1] >= cfg.cf_pos_threshold).sum() >= 3
                and len(l) > 0]
    rng = np.random.default_rng(0)
    rng.shuffle(eligible)
    n_val = min(cfg.ranker_eval_users, max(200, int(0.2 * len(eligible))))
    val_users = eligible[:n_val]
    train_users = eligible[n_val:n_val + cfg.ranker_users]
    print(f"[ranker] eligible={len(eligible)} train_queries={len(train_users)} "
          f"val_queries={len(val_users)}")

    def build_row(items, rts, ts, lab):
        prof = build_profile(items, rts - 2.5, emb, fac, store.genre_mat,
                             store.directors, store.casts, store.language,
                             store.year, timestamps=ts,
                             keywords=store.keywords,
                             companies=store.companies)
        cov = covis_scores(covis_mat, items, rts - 2.5) \
            if covis_mat is not None else None
        sl = None
        if sasrec is not None:
            model, meta = sasrec
            order = np.argsort(ts)
            seq = np.zeros(meta["max_len"], dtype=np.int64)
            tail = items[order][-meta["max_len"]:] + 1
            seq[:len(tail)] = tail
            with torch.no_grad():
                sl = model.logits(torch.tensor(seq[None, :],
                                               device=device))[0].cpu().numpy()
        tl = tt_user_logits(tt[0], tt[1], tt[2], items, rts) \
            if tt is not None else None
        cand_idx, meta_d = generate_candidates(
            prof, content_index, cf_index, pop_items, cfg,
            covis_scores=cov, sasrec_logits=sl, tt_logits=tl)
        X = compute_features(prof, cand_idx, meta_d, store,
                             content_emb=emb, item_factors=fac,
                             item_bias=bias, covis_mat=covis_mat)
        y = np.array([grade(lab.get(int(c), 0.0)) for c in cand_idx],
                     dtype=np.int32)
        return X, y

    min_cut = max(3, cfg.min_profile_items)

    def rows_for(users, snaps=(1.0,)):
        rows = []
        for i, u in enumerate(users):
            items, rts, ts = hist_map[u]
            for cut in sorted({int(len(items) * f) for f in snaps}):
                if cut < min_cut:
                    continue
                if cut >= len(items):
                    lab = dict(lab_map.get(u, {}))
                else:
                    lab = dict(zip(items[cut:].tolist(), rts[cut:].tolist()))
                    lab.update(lab_map.get(u, {}))
                if not lab:
                    continue
                pi, pr, pt = items[:cut], rts[:cut], ts[:cut]
                if (pr >= cfg.cf_pos_threshold).sum() < 2:
                    continue
                rows.append(build_row(pi, pr, pt, lab))
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(users)} users -> {len(rows)} rows",
                      flush=True)
        return rows

    tr = rows_for(train_users, snaps=snapshots)
    vr = rows_for(val_users) if val_users else []
    X, y, g = assemble(tr)
    Xv = yv = gv = None
    if vr:
        Xv, yv, gv = assemble(vr)
    print(f"[ranker] X={X.shape} pos_rate={(y>0).mean():.3f}")
    ranker = train_ranker(X, y, g, Xv, yv, gv, cfg)
    joblib.dump(ranker, cfg.models_path / out_name)
    print(importance_frame(ranker).to_string(index=False))
    print(f"ranker done in {time.time()-t:.0f}s")


def step_eval(cfg):
    t = _t("EVAL")
    from .evaluate import run_eval
    from .index import load_index
    from .split import temporal_split
    movies_pd, emb, fac, bias, ratings = _load_stack(cfg)
    train, val, test, t1, t2 = temporal_split(ratings, cfg)
    content_index = load_index(cfg.models_path / "content_hnsw.faiss", cfg)
    cf_index = load_index(cfg.models_path / "cf_hnsw.faiss", cfg)
    ranker = joblib.load(cfg.models_path / "ranker.joblib")
    pop_items = np.load(cfg.models_path / "pop_items.npy")
    covis_p = cfg.models_path / "covis.npz"
    covis_mat = sp.load_npz(covis_p) if covis_p.exists() else None
    sasrec = _load_sasrec(cfg)
    sasrec_model = sasrec[0] if sasrec else None
    if sasrec_model is not None and torch.cuda.is_available():
        sasrec_model = sasrec_model.to("cuda")

    table, details = run_eval(movies_pd, train, val, test, emb, fac, bias,
                              content_index, cf_index, ranker, pop_items, cfg,
                              covis_mat=covis_mat, sasrec_model=sasrec_model)
    df = pd.DataFrame(table).T
    pd.set_option("display.width", 220)
    print(df.round(4).to_string())
    df.to_csv(cfg.models_path / "eval_results.csv")
    print(f"eval done in {time.time()-t:.0f}s")


STEPS = {"etl": step_etl, "embed": step_embed, "cf": step_cf,
         "lightgcn": step_lightgcn, "covis": step_covis, "sasrec": step_sasrec,
         "twotower": step_twotower,
         "index": step_index, "ranker": step_ranker, "eval": step_eval}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="all",
                    help="comma list or 'all'")
    args = ap.parse_args()
    CFG.ensure_dirs()
    order = ["etl", "embed", "cf", "lightgcn", "covis", "sasrec",
             "index", "ranker", "eval"]
    todo = order if args.steps == "all" else [s.strip() for s in args.steps.split(",")]
    for s in todo:
        STEPS[s](CFG)


if __name__ == "__main__":
    main()
