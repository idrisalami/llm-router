"""Embedding spread analysis.

Measures and plots how spread out / clustered the prompt embeddings are.
Tight clusters → low routing signal. Good spread → embedding encodes
distinguishable features the router can exploit.

Metrics tracked
---------------
  mean_cos_sim       : mean pairwise cosine similarity (0 = diverse, 1 = identical)
  std_cos_sim        : std of pairwise cosine similarities
  effective_dim_90   : # PCA components to explain 90% of variance
  effective_dim_95   : # PCA components to explain 95% of variance
  pct_var_top2       : % variance explained by the first 2 PCs (PCA scatter quality)
  mean_l2_dist       : mean pairwise L2 distance (after unit-normalising embeddings)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


OUTPUT_DIR = Path(__file__).parent.parent / "output" / "embeddings"


def compute_spread_metrics(embeddings: np.ndarray) -> dict[str, float]:
    """Return all spread metrics for a set of embeddings."""
    # Unit-normalise so cosine similarity = dot product
    E = normalize(embeddings, norm="l2")
    n = len(E)

    # Pairwise cosine similarities (upper triangle, excluding diagonal)
    gram = E @ E.T
    idx  = np.triu_indices(n, k=1)
    cos_sims = gram[idx]

    # PCA
    pca = PCA().fit(E)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    eff_dim_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    eff_dim_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    pct_var_top2 = float(pca.explained_variance_ratio_[:2].sum() * 100)

    # L2 distances on unit-normalised vectors
    # ||u - v||^2 = 2 - 2*cos_sim  →  ||u - v|| = sqrt(2 - 2*cos_sim)
    l2_dists = np.sqrt(np.clip(2 - 2 * cos_sims, 0, None))

    return {
        "n_prompts":       n,
        "emb_dim":         embeddings.shape[1],
        "mean_cos_sim":    float(cos_sims.mean()),
        "std_cos_sim":     float(cos_sims.std()),
        "min_cos_sim":     float(cos_sims.min()),
        "max_cos_sim":     float(cos_sims.max()),
        "mean_l2_dist":    float(l2_dists.mean()),
        "std_l2_dist":     float(l2_dists.std()),
        "effective_dim_90": eff_dim_90,
        "effective_dim_95": eff_dim_95,
        "pct_var_top2":    pct_var_top2,
    }


def print_metrics(metrics: dict[str, float], label: str = "") -> None:
    header = f"  Embedding spread — {label}" if label else "  Embedding spread"
    print()
    print("=" * 60)
    print(header)
    print("=" * 60)
    print(f"  Prompts / dim          : {metrics['n_prompts']} × {metrics['emb_dim']}")
    print(f"  Mean cosine similarity : {metrics['mean_cos_sim']:.4f}  (0=diverse, 1=identical)")
    print(f"  Std  cosine similarity : {metrics['std_cos_sim']:.4f}")
    print(f"  Range                  : [{metrics['min_cos_sim']:.4f}, {metrics['max_cos_sim']:.4f}]")
    print(f"  Mean L2 distance       : {metrics['mean_l2_dist']:.4f}")
    print(f"  Effective dim (90% var): {metrics['effective_dim_90']}")
    print(f"  Effective dim (95% var): {metrics['effective_dim_95']}")
    print(f"  Var explained by PC1+2 : {metrics['pct_var_top2']:.1f}%")
    print("=" * 60)
    print()


def plot_embedding_spread(
    embeddings: np.ndarray,
    label: str,
    metrics: dict[str, float] | None = None,
    output_dir: Path = OUTPUT_DIR,
    filename: str | None = None,
) -> Path:
    """Four-panel embedding spread plot.

    Panel 1: pairwise cosine similarity distribution
    Panel 2: PCA scatter (PC1 vs PC2), coloured by PC3
    Panel 3: PCA scree plot (cumulative explained variance)
    Panel 4: L2 distance distribution
    """
    if metrics is None:
        metrics = compute_spread_metrics(embeddings)

    E    = normalize(embeddings, norm="l2")
    n    = len(E)
    pca  = PCA(n_components=min(50, n, E.shape[1])).fit(E)
    pcs  = pca.transform(E)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100

    gram = E @ E.T
    idx  = np.triu_indices(n, k=1)
    cos_sims = gram[idx]
    l2_dists = np.sqrt(np.clip(2 - 2 * cos_sims, 0, None))

    output_dir.mkdir(parents=True, exist_ok=True)
    slug = label.replace(" ", "_").replace("/", "-").lower()
    out_path = output_dir / (filename or f"embedding_spread_{slug}.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Embedding Spread Analysis — {label}\n"
        f"mean cos sim={metrics['mean_cos_sim']:.4f}  |  "
        f"eff. dim (90%)={metrics['effective_dim_90']}  |  "
        f"mean L2={metrics['mean_l2_dist']:.4f}",
        fontsize=12, fontweight="bold",
    )

    # ── Panel 1: cosine similarity distribution ───────────────────────────────
    ax = axes[0, 0]
    ax.hist(cos_sims, bins=60, color="#4C72B0", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(metrics["mean_cos_sim"], color="#C44E52", linewidth=2,
               label=f"mean = {metrics['mean_cos_sim']:.4f}")
    ax.set_xlabel("Pairwise cosine similarity")
    ax.set_ylabel("# pairs")
    ax.set_title("Pairwise cosine similarity\n(1 = identical, 0 = orthogonal)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 2: PCA scatter PC1 vs PC2 ──────────────────────────────────────
    ax = axes[0, 1]
    color_vals = pcs[:, 2] if pcs.shape[1] > 2 else np.zeros(n)
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=color_vals, cmap="coolwarm",
                    s=18, alpha=0.7, linewidths=0)
    plt.colorbar(sc, ax=ax, label="PC3 value")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title(f"PCA scatter (PC1 vs PC2)\n{metrics['pct_var_top2']:.1f}% variance explained")
    ax.grid(alpha=0.3)

    # ── Panel 3: scree / cumulative variance ──────────────────────────────────
    ax = axes[1, 0]
    n_shown = min(40, len(cumvar))
    ax.plot(range(1, n_shown + 1), cumvar[:n_shown], marker="o", markersize=3,
            color="#4C72B0", linewidth=1.5)
    ax.axhline(90, color="#C44E52", linewidth=1.2, linestyle="--",
               label=f"90% → {metrics['effective_dim_90']} dims")
    ax.axhline(95, color="#DD8452", linewidth=1.2, linestyle="--",
               label=f"95% → {metrics['effective_dim_95']} dims")
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative variance explained (%)")
    ax.set_title("PCA scree — how many dims carry signal?")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 102)

    # ── Panel 4: L2 distance distribution ────────────────────────────────────
    ax = axes[1, 1]
    ax.hist(l2_dists, bins=60, color="#55A868", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(metrics["mean_l2_dist"], color="#C44E52", linewidth=2,
               label=f"mean = {metrics['mean_l2_dist']:.4f}")
    ax.set_xlabel("Pairwise L2 distance (unit-normalised embeddings)")
    ax.set_ylabel("# pairs")
    ax.set_title("L2 distance distribution\n(higher = more diverse)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[embed] Plot saved → {out_path}")
    return out_path


def compare_spread(
    embedding_variants: list[tuple[str, np.ndarray]],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Bar chart comparing spread metrics across multiple embedding variants."""
    output_dir.mkdir(parents=True, exist_ok=True)

    labels  = [v[0] for v in embedding_variants]
    metrics = [compute_spread_metrics(v[1]) for v in embedding_variants]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Embedding Spread Comparison", fontsize=13, fontweight="bold")
    x = np.arange(len(labels))
    bar_kw = dict(edgecolor="black", linewidth=0.6)

    # Mean cosine similarity (lower is better for routing)
    ax = axes[0]
    vals = [m["mean_cos_sim"] for m in metrics]
    bars = ax.bar(x, vals, color="#4C72B0", **bar_kw)
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean pairwise cosine similarity")
    ax.set_title("Cosine similarity\n(↓ better — more diverse)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Effective dimensionality at 90% (higher is better)
    ax = axes[1]
    vals = [m["effective_dim_90"] for m in metrics]
    bars = ax.bar(x, vals, color="#55A868", **bar_kw)
    ax.bar_label(bars, fmt="%d", fontsize=8, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("# PCA components")
    ax.set_title("Effective dimensionality (90% var)\n(↑ better — signal spread wider)")
    ax.grid(axis="y", alpha=0.3)

    # Mean L2 distance (higher is better)
    ax = axes[2]
    vals = [m["mean_l2_dist"] for m in metrics]
    bars = ax.bar(x, vals, color="#DD8452", **bar_kw)
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean L2 distance")
    ax.set_title("Mean pairwise L2 distance\n(↑ better — more spread)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "embedding_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[embed] Comparison plot saved → {out_path}")
