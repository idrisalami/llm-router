"""Embedding enhancement techniques.

Each function takes raw MiniLM embeddings (n, 384) and returns a
transformed array (n, d).  Use compare_spread() from embedding_analysis.py
to see whether the transformation improves spread metrics.

Techniques
----------
whiten_zca        : ZCA whitening — decorrelates + equalises all dims
whiten_pca        : PCA-whitening — decorrelate + project to top-k dims
augment_code      : Append code-derived scalar features (length, keyword
                    counts, punctuation density) to the embedding vector
augment_tfidf     : Append a truncated TF-IDF vector computed over the
                    prompt corpus; captures lexical diversity not captured
                    by MiniLM's pooled representation
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import normalize

# ──────────────────────────────────────────────────────────────────────────────
# ZCA whitening
# ──────────────────────────────────────────────────────────────────────────────

def whiten_zca(
    embeddings: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """ZCA (Mahalanobis) whitening.

    Transforms embeddings so that the empirical covariance ≈ I.
    Every dimension gets unit variance and cross-dimension correlations
    are removed — vectors that were clustered due to shared directions
    get pushed further apart.

    Parameters
    ----------
    eps : regularisation added to eigenvalues before inversion (avoids
          division by near-zero).
    """
    X = embeddings.astype(np.float64)
    X -= X.mean(axis=0)

    cov = X.T @ X / (len(X) - 1)          # (d, d)
    U, S, Vt = np.linalg.svd(cov)
    W = U @ np.diag(1.0 / np.sqrt(S + eps)) @ Vt   # ZCA matrix
    out = (X @ W.T).astype(np.float32)
    # Re-normalise to unit sphere so cosine metrics are meaningful
    return normalize(out, norm="l2")


# ──────────────────────────────────────────────────────────────────────────────
# PCA whitening (dimensionality-reducing)
# ──────────────────────────────────────────────────────────────────────────────

def whiten_pca(
    embeddings: np.ndarray,
    n_components: int = 128,
    eps: float = 1e-5,
) -> np.ndarray:
    """Project to the top-k PCA components then whiten.

    Removes low-variance noise dimensions that contribute nothing to
    routing signal.  n_components=128 keeps ~93-95% of MiniLM variance
    while discarding the tail.
    """
    X = embeddings.astype(np.float64)
    X -= X.mean(axis=0)

    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X)                         # (n, k)

    # Whiten in PCA space (divide each PC by its std)
    stds = Z.std(axis=0) + eps
    Z_w = (Z / stds).astype(np.float32)
    return normalize(Z_w, norm="l2")


# ──────────────────────────────────────────────────────────────────────────────
# Code-feature augmentation
# ──────────────────────────────────────────────────────────────────────────────

_PYTHON_KEYWORDS = [
    "def ", "class ", "return ", "import ", "for ", "while ",
    "if ", "else ", "elif ", "try ", "except ", "lambda ",
    "yield ", "with ", "assert ", "raise ",
]

def _code_features(prompt: str) -> list[float]:
    """~20 scalar code-complexity features extracted from the prompt text."""
    p = prompt.lower()
    tokens = re.split(r"\W+", p)

    features = [
        len(prompt),                              # total chars
        len(tokens),                              # rough word count
        prompt.count("\n"),                       # newlines
        prompt.count("("),                        # parentheses
        prompt.count("["),                        # brackets
        prompt.count("{"),                        # braces
        prompt.count(":"),                        # colon count
        prompt.count(","),                        # comma count
        p.count("return"),                        # 'return' mentions
        p.count("list"),
        p.count("dict"),
        p.count("string"),
        p.count("sort"),
        p.count("sum"),
        p.count("max"),
        p.count("min"),
        p.count("index"),
        sum(p.count(kw) for kw in _PYTHON_KEYWORDS),   # keyword density
        float(len(prompt)) / max(len(tokens), 1),       # avg word length
        float(prompt.count(" ")) / max(len(prompt), 1), # space density
    ]
    return features


def augment_code(
    embeddings: np.ndarray,
    prompts: list[str],
    weight: float = 1.0,
) -> np.ndarray:
    """Append normalised code-derived scalar features to MiniLM embeddings.

    The scalar block is z-score normalised then scaled by `weight` so it
    doesn't dominate the 384-dim embedding part.  Both halves are then
    L2-normalised as a whole.

    Parameters
    ----------
    weight : relative weight of the scalar feature block (1.0 = equal
             contribution after standardisation, >1 = more weight).
    """
    feat_matrix = np.array([_code_features(p) for p in prompts], dtype=np.float32)
    # z-score normalise each feature column
    mu  = feat_matrix.mean(axis=0)
    std = feat_matrix.std(axis=0) + 1e-8
    feat_matrix = (feat_matrix - mu) / std * weight

    combined = np.hstack([embeddings, feat_matrix])
    return normalize(combined, norm="l2")


# ──────────────────────────────────────────────────────────────────────────────
# TF-IDF augmentation
# ──────────────────────────────────────────────────────────────────────────────

def augment_tfidf(
    embeddings: np.ndarray,
    prompts: list[str],
    max_features: int = 200,
    weight: float = 0.5,
) -> np.ndarray:
    """Append a truncated TF-IDF vector to MiniLM embeddings.

    MiniLM pools tokens into a single vector; rare but discriminative
    words may be washed out.  TF-IDF preserves lexical spread.
    max_features keeps dimensionality manageable.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as sk_normalize

    tfidf = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
    )
    T = tfidf.fit_transform(prompts).toarray().astype(np.float32)
    # Reduce TF-IDF to 64 dims via SVD to avoid high-dim noise
    n_svd = min(64, T.shape[1], T.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_svd, random_state=42)
    T_reduced = svd.fit_transform(T).astype(np.float32)
    T_norm = sk_normalize(T_reduced, norm="l2") * weight

    combined = np.hstack([embeddings, T_norm])
    return normalize(combined, norm="l2")


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: build all variants at once
# ──────────────────────────────────────────────────────────────────────────────

def augment_llm_expansion(
    prompts: list[str],
    force_recompute: bool = False,
) -> np.ndarray:
    """Embed original-prompt + LLM expansion with MiniLM, then ZCA-whiten.

    The expansion (structured difficulty profile from Claude Haiku) adds
    per-prompt signal about algorithm type, complexity, and edge cases that
    the terse original prompt lacks.  ZCA whitening is applied on top to
    maximise geometric spread.
    """
    from analysis.llm_expander import load_or_expand
    from router.features import _embed_minilm

    expanded = load_or_expand(prompts, force_recompute=force_recompute)
    raw_emb = _embed_minilm(expanded)
    return whiten_zca(raw_emb)


def build_all_variants(
    embeddings: np.ndarray,
    prompts: list[str],
    include_llm: bool = False,
    recompute_expansions: bool = False,
) -> list[tuple[str, np.ndarray]]:
    """Return [(name, embeddings)] for MiniLM + all enhancement variants.

    include_llm=False by default because it requires the Anthropic API and
    cached expansions.  Pass include_llm=True (or --llm-expand CLI flag)
    to include the LLM-expansion variant.
    """
    variants = [
        ("MiniLM (raw)",        normalize(embeddings, norm="l2")),
        ("ZCA-whitened",        whiten_zca(embeddings)),
        ("PCA-128 whitened",    whiten_pca(embeddings, n_components=128)),
        ("+ code features",     augment_code(embeddings, prompts)),
        ("+ TF-IDF (SVD-64)",   augment_tfidf(embeddings, prompts)),
    ]
    if include_llm:
        variants.append((
            "LLM-expanded + ZCA",
            augment_llm_expansion(prompts, force_recompute=recompute_expansions),
        ))
    return variants
