"""Prompt embedding using sentence-transformers.

Uses all-MiniLM-L6-v2 (384-dim, ~80 MB, runs locally, no API key needed).
Embeddings are cached to data/embeddings_mbpp.npy so they are only computed once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR        = Path(__file__).parent.parent / "data"
EMBEDDING_CACHE  = CACHE_DIR / "embeddings_mbpp.npy"
PROMPT_CACHE     = CACHE_DIR / "embeddings_mbpp_prompts.txt"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   # 384-dim, fast, MIT licence


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_or_compute_embeddings(
    df: pd.DataFrame,
    force_recompute: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Return (embeddings, prompts) for all unique prompts in df.

    Loads from cache if available, otherwise runs the sentence-transformer model
    and writes the results to data/embeddings_mbpp.npy.

    embeddings shape: (n_prompts, 384)
    prompts: list of prompt strings in the same row order as embeddings
    """
    prompts = df["input"].dropna().unique().tolist()

    if EMBEDDING_CACHE.exists() and PROMPT_CACHE.exists() and not force_recompute:
        cached_prompts = PROMPT_CACHE.read_text(encoding="utf-8").splitlines()
        if cached_prompts == prompts:
            print(f"[features] Loading cached embeddings ({len(prompts)} prompts) from {EMBEDDING_CACHE}")
            return np.load(EMBEDDING_CACHE), prompts
        else:
            print("[features] Prompt list changed — recomputing embeddings.")

    embeddings = _embed(prompts)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDING_CACHE, embeddings)
    PROMPT_CACHE.write_text("\n".join(prompts), encoding="utf-8")
    print(f"[features] Embeddings saved → {EMBEDDING_CACHE}")
    return embeddings, prompts


def embed_single(prompt: str) -> np.ndarray:
    """Embed one prompt string. Returns shape (384,)."""
    return _embed([prompt])[0]


# ──────────────────────────────────────────────────────────────────────────────
# Internal
# ──────────────────────────────────────────────────────────────────────────────

def _embed(prompts: list[str]) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Install sentence-transformers:  pip install sentence-transformers"
        ) from exc

    print(f"[features] Encoding {len(prompts)} prompts with {EMBEDDING_MODEL}…")
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model.encode(prompts, show_progress_bar=True, convert_to_numpy=True)
