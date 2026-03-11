"""Prompt embedding — MiniLM backend.

Embeddings are cached to data/embeddings_minilm.npy so they are only
computed once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).parent.parent / "data"

_MINILM_MODEL = "all-MiniLM-L6-v2"
_CACHE_EMB    = CACHE_DIR / "embeddings_minilm.npy"
_CACHE_PROMPTS = CACHE_DIR / "embeddings_minilm_prompts.txt"


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_or_compute_embeddings(
    df: pd.DataFrame,
    force_recompute: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Return (embeddings, prompts) for all unique prompts in df.

    Loads from cache if available, otherwise runs all-MiniLM-L6-v2 and
    saves the result.
    """
    prompts = df["input"].dropna().unique().tolist()

    if _CACHE_EMB.exists() and _CACHE_PROMPTS.exists() and not force_recompute:
        cached_prompts = _CACHE_PROMPTS.read_text(encoding="utf-8").splitlines()
        if cached_prompts == prompts:
            print(f"[features] Loading cached MiniLM embeddings ({len(prompts)} prompts)")
            return np.load(_CACHE_EMB), prompts
        print("[features] Prompt list changed — recomputing MiniLM embeddings.")

    embeddings = _embed_minilm(prompts)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(_CACHE_EMB, embeddings)
    _CACHE_PROMPTS.write_text("\n".join(prompts), encoding="utf-8")
    print(f"[features] MiniLM embeddings saved → {_CACHE_EMB}")
    return embeddings, prompts


def embed_single(prompt: str) -> np.ndarray:
    """Embed one prompt string."""
    return _embed_minilm([prompt])[0]


# ──────────────────────────────────────────────────────────────────────────────
# Internal
# ──────────────────────────────────────────────────────────────────────────────

def _embed_minilm(prompts: list[str]) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise ImportError("pip install sentence-transformers") from exc

    print(f"[features] Encoding {len(prompts)} prompts with {_MINILM_MODEL}…")
    model = SentenceTransformer(_MINILM_MODEL)
    return model.encode(prompts, show_progress_bar=True, convert_to_numpy=True)
