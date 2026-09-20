"""LightGBM LambdaRank over generated candidates.

Labels are graded relevance from the *future* window:
  rating >= 4.5 -> 2,  3.5 <= rating < 4.5 -> 1, else 0.
group = candidates per query (LightGBM requirement).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np

from .config import CFG, Config
from .features import FEATURES


def grade(rating: float) -> int:
    if rating >= 4.5:
        return 4
    if rating >= 4.0:
        return 3
    if rating >= 3.5:
        return 2
    if rating >= 3.0:
        return 1
    return 0


def train_ranker(X: np.ndarray, y: np.ndarray, group: np.ndarray,
                 Xv=None, yv=None, gv=None, cfg: Config = CFG) -> lgb.LGBMRanker:
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=cfg.lgbm_estimators,
        learning_rate=cfg.lgbm_lr,
        num_leaves=cfg.lgbm_leaves,
        min_child_samples=50,
        subsample=0.9, subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=-1,
        importance_type="gain",
        random_state=42,
    )
    fit_kw = {}
    if Xv is not None:
        fit_kw = dict(
            eval_set=[(Xv, yv)], eval_group=[list(gv)],
            eval_metric="ndcg", eval_at=[10, 20],
            callbacks=[lgb.early_stopping(cfg.lgbm_early_stop, verbose=False),
                       lgb.log_evaluation(50)],
        )
    ranker.fit(X, y, group=list(group), **fit_kw)
    return ranker


def assemble(user_rows: list[tuple[np.ndarray, np.ndarray]]):
    """[(X_u, y_u)] -> X, y, group."""
    X = np.vstack([r[0] for r in user_rows])
    y = np.concatenate([r[1] for r in user_rows])
    group = np.array([len(r[1]) for r in user_rows], dtype=np.int64)
    return X, y, group


def importance_frame(ranker: lgb.LGBMRanker):
    import pandas as pd
    return pd.DataFrame({
        "feature": FEATURES,
        "gain": ranker.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
