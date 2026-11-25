"""Frozen sentence-encoder embeddings, cached to disk (no fine-tuning; proposal §4.2.1 step 3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

_ENCODERS: dict = {}


def get_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    if model_name not in _ENCODERS:
        _ENCODERS[model_name] = SentenceTransformer(model_name)
    return _ENCODERS[model_name]


def embed(texts, model_name: str, cache: str | Path | None = None, batch_size: int = 64) -> np.ndarray:
    """L2-normalized embeddings; loaded from `cache` (.npy) if it exists."""
    if cache is not None and Path(cache).exists():
        return np.load(cache)
    E = get_encoder(model_name).encode(
        list(texts), batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
    )
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, E)
    return E
