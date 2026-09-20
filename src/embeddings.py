"""GPU sentence embeddings for canonical movie documents.

all-MiniLM-L6-v2 -> 384-d, L2-normalized, stored fp16 (~66MB for 86k films).
FAISS consumes the fp32 cast produced by `load_embeddings`.
"""
from __future__ import annotations

import numpy as np
import torch

from .config import CFG, Config


def encode_corpus(texts: list[str], cfg: Config = CFG,
                  device: str | None = None) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(cfg.emb_model, device=device)
    emb = model.encode(
        texts,
        batch_size=cfg.emb_batch,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=device,
    )
    return emb.astype(np.float16)


def load_embeddings(cfg: Config = CFG) -> np.ndarray:
    """Returns fp32 L2-normalized (n_items, dim) content embeddings."""
    e = np.load(cfg.emb_path / "content.f16.npy").astype(np.float32)
    norms = np.linalg.norm(e, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return e / norms


def run_embeddings(texts: list[str], cfg: Config = CFG) -> np.ndarray:
    cfg.ensure_dirs()
    emb = encode_corpus(texts, cfg)
    np.save(cfg.emb_path / "content.f16.npy", emb)
    print(f"[emb] saved {emb.shape} -> {cfg.emb_path/'content.f16.npy'}")
    return emb
