"""FAISS HNSW ANN indexes.

Vectors are L2-normalized and indexed with METRIC_INNER_PRODUCT so inner
product == cosine similarity.
"""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from .config import CFG, Config


def build_hnsw(vectors: np.ndarray, cfg: Config = CFG) -> faiss.Index:
    v = np.ascontiguousarray(vectors.astype(np.float32))
    faiss.normalize_L2(v)
    index = faiss.IndexHNSWFlat(v.shape[1], cfg.hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = cfg.ef_construction
    index.add(v)
    index.hnsw.efSearch = cfg.ef_search
    return index


def save_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path, cfg: Config = CFG) -> faiss.Index:
    index = faiss.read_index(str(path))
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = cfg.ef_search
    return index


def query(index: faiss.Index, vectors: np.ndarray, k: int):
    """Returns (scores, indices); scores are cosine sims for normed inputs."""
    q = np.ascontiguousarray(vectors.astype(np.float32))
    faiss.normalize_L2(q)
    return index.search(q, k)
