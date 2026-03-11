"""Prompt embedding — supports multiple embedding backends.

Backends
--------
  minilm   : all-MiniLM-L6-v2 (384-dim, sentence-transformers, fast)
  codebert : microsoft/codebert-base (768-dim, transformers, CLS token)
  codet5   : Salesforce/codet5-base (768-dim, T5 encoder, mean pool)

Embeddings are cached to data/embeddings_<model>.npy so they are only
computed once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).parent.parent / "data"

EMBEDDING_MODELS = {
    "minilm":        "all-MiniLM-L6-v2",
    "codebert":      "microsoft/codebert-base",
    "graphcodebert": "microsoft/graphcodebert-base",
}


def _cache_paths(model: str) -> tuple[Path, Path]:
    slug = model.replace("/", "-").replace("\\", "-")
    return CACHE_DIR / f"embeddings_{slug}.npy", CACHE_DIR / f"embeddings_{slug}_prompts.txt"


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_or_compute_embeddings(
    df: pd.DataFrame,
    model: str = "minilm",
    force_recompute: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Return (embeddings, prompts) for all unique prompts in df.

    model: one of 'minilm', 'codebert', 'codet5' (short name) or a full
           HuggingFace model id.

    Loads from cache if available, otherwise runs the embedding model and
    saves the result.
    """
    full_name = EMBEDDING_MODELS.get(model, model)
    emb_cache, prompt_cache = _cache_paths(model)

    prompts = df["input"].dropna().unique().tolist()

    if emb_cache.exists() and prompt_cache.exists() and not force_recompute:
        cached_prompts = prompt_cache.read_text(encoding="utf-8").splitlines()
        if cached_prompts == prompts:
            print(f"[features] Loading cached {model} embeddings ({len(prompts)} prompts)")
            return np.load(emb_cache), prompts
        print(f"[features] Prompt list changed — recomputing {model} embeddings.")

    embeddings = _embed(prompts, model, full_name)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(emb_cache, embeddings)
    prompt_cache.write_text("\n".join(prompts), encoding="utf-8")
    print(f"[features] {model} embeddings saved → {emb_cache}")
    return embeddings, prompts


def embed_single(prompt: str, model: str = "minilm") -> np.ndarray:
    """Embed one prompt string."""
    return _embed([prompt], model, EMBEDDING_MODELS.get(model, model))[0]


# ──────────────────────────────────────────────────────────────────────────────
# Internal dispatchers
# ──────────────────────────────────────────────────────────────────────────────

def _embed(prompts: list[str], model_key: str, full_name: str) -> np.ndarray:
    if model_key == "minilm" or (model_key not in EMBEDDING_MODELS and "MiniLM" in full_name):
        return _embed_sentence_transformers(prompts, full_name)
    elif model_key in ("codebert",) or "codebert" in full_name.lower():
        return _embed_bert(prompts, full_name)
    elif model_key in ("graphcodebert",) or "graphcodebert" in full_name.lower():
        return _embed_bert(prompts, full_name)
    else:
        # Fall back to sentence-transformers for unknown models
        return _embed_sentence_transformers(prompts, full_name)


def _embed_sentence_transformers(prompts: list[str], model_name: str) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise ImportError("pip install sentence-transformers") from exc

    print(f"[features] Encoding {len(prompts)} prompts with {model_name}…")
    model = SentenceTransformer(model_name)
    return model.encode(prompts, show_progress_bar=True, convert_to_numpy=True)


def _embed_bert(prompts: list[str], model_name: str, batch_size: int = 16) -> np.ndarray:
    """CLS-token embedding from a BERT-style model (e.g. CodeBERT)."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise ImportError("pip install transformers torch") from exc

    print(f"[features] Encoding {len(prompts)} prompts with {model_name} (CLS token)…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    all_embeddings = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                           max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_emb)
        if (i // batch_size) % 5 == 0:
            print(f"  [{i + len(batch)}/{len(prompts)}]", end="\r")

    print()
    return np.vstack(all_embeddings)


def _embed_t5_encoder(prompts: list[str], model_name: str, batch_size: int = 16) -> np.ndarray:
    """Mean-pool over the T5 encoder's last hidden state (e.g. CodeT5)."""
    try:
        import torch
        from transformers import AutoTokenizer, T5EncoderModel  # type: ignore
    except ImportError as exc:
        raise ImportError("pip install transformers torch") from exc

    print(f"[features] Encoding {len(prompts)} prompts with {model_name} (T5 encoder, mean pool)…")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = T5EncoderModel.from_pretrained(model_name)
    model.eval()

    all_embeddings = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                           max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pool over non-padding tokens
            hidden = outputs.last_hidden_state          # (B, seq, dim)
            mask   = inputs["attention_mask"].unsqueeze(-1).float()
            mean_emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            all_embeddings.append(mean_emb.cpu().numpy())
        if (i // batch_size) % 5 == 0:
            print(f"  [{i + len(batch)}/{len(prompts)}]", end="\r")

    print()
    return np.vstack(all_embeddings)
