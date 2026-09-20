"""Central configuration. Values come from configs/config.yaml if present."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_yaml() -> dict:
    cfg_path = ROOT / "configs" / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class Config:
    ml_dir: str = "ml-32m"
    tmdb_csv: str = "tmdb/TMDB_all_movies.csv"
    processed_dir: str = "data/processed"
    emb_dir: str = "data/embeddings"
    models_dir: str = "models"

    bayes_m: float = 100.0
    cast_top_n: int = 5
    min_ratings_per_movie: int = 1

    train_frac: float = 0.80
    val_frac: float = 0.90
    min_user_ratings: int = 5

    emb_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    emb_batch: int = 512
    emb_max_chars: int = 1200

    cf_dim: int = 128
    cf_epochs: int = 10
    cf_lr: float = 0.05
    cf_reg: float = 1e-5
    cf_batch: int = 524288
    cf_neg: int = 4
    cf_pos_threshold: float = 3.5

    hnsw_m: int = 32
    ef_construction: int = 200
    ef_search: int = 128

    content_k: int = 150
    cf_k: int = 150
    covis_k: int = 150
    sasrec_k: int = 100
    popularity_k: int = 200
    tt_k: int = 200

    ranker_users: int = 15000
    ranker_eval_users: int = 4000
    min_profile_items: int = 5
    lgbm_estimators: int = 600
    lgbm_lr: float = 0.05
    lgbm_leaves: int = 63
    lgbm_early_stop: int = 50

    mmr_lambda: float = 0.7
    ranker_top_n: int = 50
    final_k: int = 10

    _sections: dict = field(default_factory=dict, repr=False)

    # ---- resolved absolute paths ----
    @property
    def ml_path(self) -> Path:
        return ROOT / self.ml_dir

    @property
    def tmdb_path(self) -> Path:
        return ROOT / self.tmdb_csv

    @property
    def processed_path(self) -> Path:
        return ROOT / self.processed_dir

    @property
    def emb_path(self) -> Path:
        return ROOT / self.emb_dir

    @property
    def models_path(self) -> Path:
        return ROOT / self.models_dir

    def ensure_dirs(self) -> None:
        for p in (self.processed_path, self.emb_path, self.models_path):
            p.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    raw = _load_yaml()
    cfg = Config()
    paths = raw.get("paths", {})
    for k in ("ml_dir", "tmdb_csv", "processed_dir", "emb_dir", "models_dir"):
        if k in paths:
            setattr(cfg, k, paths[k])
    etl = raw.get("etl", {})
    cfg.bayes_m = float(etl.get("bayes_m", cfg.bayes_m))
    cfg.cast_top_n = int(etl.get("cast_top_n", cfg.cast_top_n))
    cfg.min_ratings_per_movie = int(etl.get("min_ratings_per_movie", cfg.min_ratings_per_movie))
    sp = raw.get("split", {})
    cfg.train_frac = float(sp.get("train_frac", cfg.train_frac))
    cfg.val_frac = float(sp.get("val_frac", cfg.val_frac))
    cfg.min_user_ratings = int(sp.get("min_user_ratings", cfg.min_user_ratings))
    emb = raw.get("embeddings", {})
    cfg.emb_model = emb.get("model_name", cfg.emb_model)
    cfg.emb_batch = int(emb.get("batch_size", cfg.emb_batch))
    cfg.emb_max_chars = int(emb.get("max_chars", cfg.emb_max_chars))
    cf = raw.get("cf", {})
    cfg.cf_dim = int(cf.get("dim", cfg.cf_dim))
    cfg.cf_epochs = int(cf.get("epochs", cfg.cf_epochs))
    cfg.cf_lr = float(cf.get("lr", cfg.cf_lr))
    cfg.cf_reg = float(cf.get("reg", cfg.cf_reg))
    cfg.cf_batch = int(cf.get("batch_size", cfg.cf_batch))
    cfg.cf_neg = int(cf.get("neg_per_pos", cfg.cf_neg))
    cfg.cf_pos_threshold = float(cf.get("positive_threshold", cfg.cf_pos_threshold))
    idx = raw.get("index", {})
    cfg.hnsw_m = int(idx.get("hnsw_m", cfg.hnsw_m))
    cfg.ef_construction = int(idx.get("ef_construction", cfg.ef_construction))
    cfg.ef_search = int(idx.get("ef_search", cfg.ef_search))
    rt = raw.get("retrieval", {})
    cfg.content_k = int(rt.get("content_k", cfg.content_k))
    cfg.cf_k = int(rt.get("cf_k", cfg.cf_k))
    cfg.covis_k = int(rt.get("covis_k", cfg.covis_k))
    cfg.sasrec_k = int(rt.get("sasrec_k", cfg.sasrec_k))
    cfg.popularity_k = int(rt.get("popularity_k", cfg.popularity_k))
    rk = raw.get("ranker", {})
    cfg.ranker_users = int(rk.get("n_users_train", cfg.ranker_users))
    cfg.ranker_eval_users = int(rk.get("n_users_eval", cfg.ranker_eval_users))
    cfg.min_profile_items = int(rk.get("min_profile_items", cfg.min_profile_items))
    cfg.lgbm_estimators = int(rk.get("n_estimators", cfg.lgbm_estimators))
    cfg.lgbm_lr = float(rk.get("learning_rate", cfg.lgbm_lr))
    cfg.lgbm_leaves = int(rk.get("num_leaves", cfg.lgbm_leaves))
    cfg.lgbm_early_stop = int(rk.get("early_stopping", cfg.lgbm_early_stop))
    rr = raw.get("rerank", {})
    cfg.mmr_lambda = float(rr.get("mmr_lambda", cfg.mmr_lambda))
    cfg.ranker_top_n = int(rr.get("top_n", cfg.ranker_top_n))
    cfg.final_k = int(rr.get("final_k", cfg.final_k))
    cfg._sections = raw
    return cfg


CFG = load_config()
