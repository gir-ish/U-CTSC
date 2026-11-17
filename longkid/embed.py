"""Sentence embeddings (CPU) for the risk model and controller, with an on-disk cache.

Each row gets three 384-d blocks: the current child message, all of today's child messages,
and the companion's own earlier replies today. Model: BAAI/bge-small-en-v1.5 (normalised).
"""
from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path

import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384
CACHE = Path(os.environ.get("LONGKID_EMB_CACHE", ".cache/emb.pkl"))
_model = None
_cache: dict | None = None


def _key(t: str) -> str:
    return hashlib.sha1(t.encode()).hexdigest()


def _load():
    global _model, _cache
    if _cache is None:
        _cache = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL, device="cpu")


def embed(texts: list[str], save: bool = True) -> np.ndarray:
    _load()
    missing = sorted({t for t in texts if t and _key(t) not in _cache})
    if missing:
        vecs = _model.encode(missing, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        for t, v in zip(missing, vecs):
            _cache[_key(t)] = v.astype(np.float32)
        if save:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_bytes(pickle.dumps(_cache))
    zero = np.zeros(DIM, np.float32)
    return np.stack([_cache[_key(t)] if t else zero for t in texts]) if texts else np.zeros((0, DIM), np.float32)


def add_embeddings(rows: list[dict], save: bool = True) -> list[dict]:
    """Attach r['emb'] = [msg | today's child text | today's companion text] to each feature row."""
    if not rows:
        return rows
    blocks = [embed([r[k] for r in rows], save) for k in ("msg", "text", "pip")]
    E = np.hstack(blocks)
    for r, e in zip(rows, E):
        r["emb"] = e
    return rows
