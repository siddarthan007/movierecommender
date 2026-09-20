"""Serving layer: loads artifacts once, serves recommendations.

Pipeline per request:
  selected movies (+weights) -> pseudo-user profile (dual embeddings)
  -> multi-source candidate generation (content / CF / covis / SASRec / pop)
  -> LightGBM LambdaRank (41 features)
  -> mood re-weighting -> MMR diversification (adaptive lambda)
  -> deterministic explanations
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .candidates import generate_candidates, popularity_universe
from .config import CFG, Config
from .covis import covis_scores
from .dedup import build_entities, dedupe_recs
from .features import ItemStore, compute_features
from .index import load_index
from .profile import build_profile
from .rerank import mmr_select

MOOD_GENRES = {
    "Mind-bending": ["Sci-Fi", "Mystery", "Thriller"],
    "Dark": ["Thriller", "Crime", "Horror", "Film-Noir"],
    "Relaxed": ["Comedy", "Animation", "Children", "Romance"],
    "Funny": ["Comedy"],
    "Epic": ["Adventure", "Action", "War", "Western", "Fantasy"],
    "Emotional": ["Drama", "Romance"],
}


@dataclass
class RecArtifacts:
    movies: pd.DataFrame
    content_emb: np.ndarray        # fp32 normalized
    item_factors: np.ndarray       # fp32 normalized
    item_bias: np.ndarray
    content_index: object
    cf_index: object
    ranker: object
    store: ItemStore
    pop_items: np.ndarray
    covis: object | None
    sasrec: object | None          # (model, meta)
    movie_id_to_idx: dict
    idx_to_movie_id: np.ndarray
    tt: object | None = None       # (model, item_feat, tt_item_emb)
    entity_key: np.ndarray | None = None
    entity_rep: dict | None = None


def load_artifacts(cfg: Config = CFG) -> RecArtifacts:
    mdir = cfg.models_path
    movies = pd.read_parquet(cfg.processed_path / "movies.parquet")
    content_emb = np.load(cfg.emb_path / "content.f16.npy").astype(np.float32)
    norms = np.linalg.norm(content_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    content_emb /= norms

    fac_p = mdir / "item_factors.npy"
    if not fac_p.exists():
        fac_p = mdir / "bpr_item_factors.npy"
    cf = np.load(fac_p).astype(np.float32)
    cf_norm = np.linalg.norm(cf, axis=1, keepdims=True)
    cf_norm[cf_norm == 0] = 1.0
    item_factors = cf / cf_norm
    bias_p = mdir / "item_bias.npy"
    item_bias = np.load(bias_p).astype(np.float32) if bias_p.exists() \
        else np.zeros(len(movies), dtype=np.float32)

    from .index import build_hnsw
    content_index = (load_index(mdir / "content_hnsw.faiss", cfg)
                     if (mdir / "content_hnsw.faiss").exists()
                     else build_hnsw(content_emb, cfg))
    cf_index = (load_index(mdir / "cf_hnsw.faiss", cfg)
                if (mdir / "cf_hnsw.faiss").exists()
                else build_hnsw(item_factors, cfg))
    ranker = joblib.load(mdir / "ranker.joblib")
    store = ItemStore(movies)
    pop_items = np.load(mdir / "pop_items.npy")

    covis_p = mdir / "covis.npz"
    covis = sp.load_npz(covis_p) if covis_p.exists() else None

    sasrec = None
    meta_p = mdir / "sasrec_meta.joblib"
    if meta_p.exists():
        from .sasrec import SASRec
        meta = joblib.load(meta_p)
        model = SASRec(len(movies), meta["dim"], meta["max_len"])
        model.load_state_dict(torch.load(mdir / "sasrec.pt", map_location="cpu"))
        model.eval()
        if torch.cuda.is_available():
            model = model.to("cuda")
        sasrec = (model, meta)

    tt = None
    tt_p = mdir / "twotower.pt"
    tt_e = mdir / "tt_item_emb.npy"
    if tt_p.exists() and tt_e.exists():
        from .twotower import TwoTower, tt_item_features
        ck = torch.load(tt_p, map_location="cpu", weights_only=True)
        tm = TwoTower(int(ck["in_dim"]), int(ck["dim"]))
        tm.load_state_dict(ck["state"])
        tm.eval()
        if torch.cuda.is_available():
            tm = tm.to("cuda")
        tt = (tm, tt_item_features(content_emb, cf, len(movies)),
              np.load(tt_e))

    entity_key, dstats = build_entities(movies)
    print(f"[load] catalog dedup: {dstats['raw_catalog']:,} -> "
          f"{dstats['canonical_catalog']:,} entities "
          f"({dstats['movies_merged']} merged)")

    movie_ids = movies["movieId"].to_numpy()
    return RecArtifacts(
        movies=movies, content_emb=content_emb, item_factors=item_factors,
        item_bias=item_bias, content_index=content_index, cf_index=cf_index,
        ranker=ranker, store=store, pop_items=pop_items, covis=covis,
        sasrec=sasrec, tt=tt,
        movie_id_to_idx={int(m): i for i, m in enumerate(movie_ids)},
        idx_to_movie_id=movie_ids,
        entity_key=entity_key, entity_rep=dstats["rep_item"],
    )


class Recommender:
    def __init__(self, art: RecArtifacts, cfg: Config = CFG):
        self.a = art
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def load(cls, cfg: Config = CFG) -> "Recommender":
        return cls(load_artifacts(cfg), cfg)

    def profile_for(self, movie_ids, weights=None):
        a = self.a
        idx = np.array([a.movie_id_to_idx[int(m)] for m in movie_ids
                        if int(m) in a.movie_id_to_idx], dtype=np.int64)
        if len(idx) == 0:
            return None
        w = np.ones(len(idx), dtype=np.float64) if weights is None \
            else np.asarray(weights, float)
        return build_profile(idx, w, a.content_emb, a.item_factors,
                             a.store.genre_mat, a.store.directors, a.store.casts,
                             a.store.language, a.store.year,
                             keywords=a.store.keywords, companies=a.store.companies), idx

    def _sasrec_logits(self, ordered_items: np.ndarray):
        if self.a.sasrec is None or len(ordered_items) == 0:
            return None
        model, meta = self.a.sasrec
        seq = np.zeros(meta["max_len"], dtype=np.int64)
        tail = ordered_items[-meta["max_len"]:] + 1
        seq[:len(tail)] = tail
        with torch.no_grad():
            return model.logits(torch.tensor(seq[None, :], device=self.device))[0].cpu().numpy()

    def explain(self, item_idx: int, prof, cand_meta: dict) -> dict:
        """Deterministic 'why this movie' evidence."""
        a = self.a
        m = a.movies.iloc[item_idx]
        # most similar profile item (content space)
        sims = a.content_emb[prof.items] @ a.content_emb[item_idx]
        anchor_i = int(np.argmax(sims))
        anchor = a.movies.iloc[int(prof.items[anchor_i])]
        signals = []
        meta = cand_meta.get(int(item_idx), {})
        if meta.get("covis_rank", 999) < 30:
            signals.append("Often watched together")
        if prof.directors & a.store.directors[item_idx]:
            signals.append("Same director")
        shared_cast = prof.cast & a.store.casts[item_idx]
        if shared_cast:
            signals.append(f"Shares cast: {sorted(shared_cast)[0]}")
        if meta.get("sasrec_rank", 999) < 30:
            signals.append("Fits your current run")
        if meta.get("cf_rank", 999) < 30:
            signals.append("Strong collaborative match")
        g = m["genres"] if isinstance(m["genres"], (list, np.ndarray)) else []
        shared_g = [x for x in g if x in set(
            gg for gs in [a.movies.iloc[int(p)]["genres"] for p in prof.items[:5]]
            if isinstance(gs, (list, np.ndarray)) for gg in gs)]
        if shared_g:
            signals.append(" · ".join(list(dict.fromkeys(shared_g))[:3]))
        return {"anchor": anchor["title"], "signals": signals[:4]}

    def _run(self, movie_ids, weights, disliked_ids, mood,
             exploration, k, explain=True):
        """Full pipeline; returns intermediates for serving + inspection."""
        a, cfg = self.a, self.cfg
        out = self.profile_for(movie_ids, weights)
        if out is None:
            return None
        prof, prof_idx = out

        # disliked movies subtract from profile
        if disliked_ids:
            d_idx = np.array([a.movie_id_to_idx[int(m)] for m in disliked_ids
                              if int(m) in a.movie_id_to_idx], dtype=np.int64)
            if len(d_idx):
                prof, _ = self.profile_for(
                    list(movie_ids) + list(disliked_ids),
                    list(weights or np.ones(len(movie_ids))) + [-1.5] * len(d_idx))

        cov = covis_scores(a.covis, prof.items, prof.weights) if a.covis is not None else None
        sl = self._sasrec_logits(np.asarray(movie_ids and
                [a.movie_id_to_idx[int(m)] for m in movie_ids if int(m) in a.movie_id_to_idx] or []))
        tl = None
        if a.tt is not None:
            from .twotower import tt_user_logits
            tl = tt_user_logits(a.tt[0], a.tt[1], a.tt[2],
                                prof.items, prof.weights + 3.0)

        cand_idx, cand_meta = generate_candidates(
            prof, a.content_index, a.cf_index, a.pop_items, cfg,
            covis_scores=cov, sasrec_logits=sl, tt_logits=tl)
        X = compute_features(prof, cand_idx, cand_meta, a.store,
                             content_emb=a.content_emb,
                             item_factors=a.item_factors, item_bias=a.item_bias,
                             covis_mat=a.covis)
        n_f = a.ranker.booster_.num_feature() if hasattr(a.ranker, "booster_") else X.shape[1]
        scores = a.ranker.predict(X[:, :n_f]).astype(np.float64)

        # mood boost: multiplicative on normalized genre overlap
        if mood:
            mood_set = {g for m in mood for g in MOOD_GENRES.get(m, [])}
            if mood_set:
                gi = [a.store.genre_vocab.index(g) for g in mood_set
                      if g in a.store.genre_vocab]
                if gi:
                    hit = a.store.genre_mat[cand_idx][:, gi].max(1)
                    scores = scores + 0.15 * hit

        order = np.argsort(-scores)
        top = order[:cfg.ranker_top_n]
        lam = float(np.clip(0.85 - 0.4 * exploration, 0.45, 0.9))
        sel = mmr_select(scores[top], cand_idx[top], a.content_emb,
                         k, lam=lam)
        # entity-level dedup: one item per canonical movie
        if getattr(a, "entity_key", None) is not None:
            sel = np.asarray(dedupe_recs(sel.tolist(), a.entity_key,
                                         a.entity_rep), dtype=np.int64)[:k]
        res = a.movies.iloc[sel].copy()
        score_lookup = {int(cand_idx[t]): float(scores[t]) for t in top}
        res["score"] = [score_lookup.get(int(i), 0.0) for i in sel]
        expl = {int(a.movies.iloc[i]["movieId"]): self.explain(int(i), prof, cand_meta)
                for i in sel} if explain else {}
        return {"res": res, "expl": expl, "prof": prof, "prof_idx": prof_idx,
                "cand_idx": cand_idx, "cand_meta": cand_meta, "X": X,
                "scores": scores, "top": top, "sel": np.asarray(sel)}

    def recommend(self, movie_ids, k: int | None = None,
                  weights=None, disliked_ids=None,
                  mood: list[str] | None = None,
                  exploration: float = 0.5,
                  explain: bool = True) -> tuple[pd.DataFrame, dict]:
        a = self.a
        k = k or self.cfg.final_k
        d = self._run(movie_ids, weights, disliked_ids, mood,
                      exploration, k, explain=explain)
        if d is None:
            return a.movies.iloc[a.pop_items[:k]].copy(), {}
        return d["res"], d["expl"]

    def diagnose(self, movie_ids, weights=None, disliked_ids=None,
                 mood: list[str] | None = None, exploration: float = 0.5,
                 k: int | None = None) -> dict | None:
        """Same pipeline as recommend(), returns internals for inspection."""
        return self._run(movie_ids, weights, disliked_ids, mood,
                         exploration, k or self.cfg.final_k)

    # ---------- search ----------
    @staticmethod
    def _norm_title(t: str) -> str:
        t = t.lower().strip()
        t = re.sub(r"\s*\(\d{4}\)\s*$", "", t)
        t = re.sub(r"^(the|a|an)\s+", "", t)
        m = re.match(r"^(.*),\s*(the|a|an)$", t)
        if m:
            t = f"{m.group(2)} {m.group(1)}"
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()

    def _search_arrays(self):
        """Lazy normalized-title index for fuzzy search."""
        if not hasattr(self, "_norm"):
            m = self.a.movies
            self._norm = np.array([self._norm_title(t)
                                   for t in m["title"].tolist()])
            self._years = m["year"].fillna(0).to_numpy()
            self._pop = np.log1p(m["rating_count"].fillna(0).to_numpy())
        return self._norm, self._years, self._pop

    def search(self, query_str: str, limit: int = 20) -> pd.DataFrame:
        from rapidfuzz import fuzz, process
        m = self.a.movies
        q = query_str.strip()
        if not q:
            return m.iloc[[]]
        # imdb id search
        imdb = re.fullmatch(r"(?:tt)?0*(\d{5,7})", q)
        if imdb:
            hits = m[m["imdbId"] == int(imdb.group(1))]
            if len(hits):
                return hits
        # year-aware: "dune 2021"
        yr = re.search(r"\b(19\d{2}|20\d{2})\b", q)
        yq = int(yr.group(0)) if yr else None
        qtext = (q[:yr.start()] + q[yr.end():]).strip() if yr else q
        nq = self._norm_title(qtext)
        if not nq:
            return m.iloc[[]]
        norm, years, pop = self._search_arrays()
        exact = norm == nq
        if exact.any():
            pos = np.flatnonzero(exact)
            s = 200.0 + pop[pos]
            if yq:
                s += (years[pos] == yq) * 50
            return m.iloc[pos[np.argsort(-s)][:limit]]
        # fuzzy: token-aware, typo-tolerant, popularity tie-break
        sims = process.cdist([nq], norm, scorer=fuzz.WRatio,
                             score_cutoff=55)[0]
        pos = np.flatnonzero(sims > 0)
        if not len(pos):
            return m.iloc[[]]
        # candidate pool by WRatio, reranked by per-token coverage so
        # unmatched query tokens (e.g. "zzz-nothing") can't hit junk titles
        pos = pos[np.argsort(-sims[pos])][:150]
        qt = nq.split()
        scored = []
        for i in pos:
            tt = norm[i].split()
            cov = float(min(max(fuzz.ratio(a, b) for b in tt)
                            for a in qt)) if tt else 0.0
            if cov >= 55:
                scored.append((0.55 * sims[i] + 0.45 * cov
                               + pop[i], i))
        if not scored:
            return m.iloc[[]]
        if yq:
            scored = [(s + (25 if years[i] == yq else
                            -min(abs(years[i] - yq), 20)), i)
                      for s, i in scored]
        scored.sort(key=lambda x: -x[0])
        return m.iloc[[i for _, i in scored][:limit]]
