"""Generate all plots for a single-prompt model sweep.

Outputs
-------
    accuracy_bar.png      — pass@1 per model
    cost_bar.png          — avg cost (USD) per model
    latency_bar.png       — avg latency (s) per model  [skipped if col missing]
    pareto_frontier.png   — matplotlib scatter + Pareto hull
    pareto_frontier.html  — plotly interactive version
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works in any environment
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Style constants
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]
PARETO_COLOR  = "#E63946"   # red — Pareto-optimal models
DEFAULT_COLOR = "#A8DADC"   # light blue — sub-optimal models
FIG_DPI = 150


# ──────────────────────────────────────────────────────────────────────────────
# Pareto frontier helper (NDCH — Non-Decreasing Convex Hull)
# ──────────────────────────────────────────────────────────────────────────────

def compute_pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Mark each model as Pareto-optimal or not.

    Methodology: sort by cost (ascending); a model is on the frontier if its
    accuracy is strictly greater than all cheaper models seen so far.
    This matches the NDCH method used in RouterBench (Hu et al., 2024).

    Returns the input df with an added boolean column 'pareto_optimal'.
    """
    df = df.copy().sort_values("cost").reset_index(drop=True)
    max_acc = -1.0
    flags: list[bool] = []
    for _, row in df.iterrows():
        if row["accuracy"] > max_acc:
            flags.append(True)
            max_acc = row["accuracy"]
        else:
            flags.append(False)
    df["pareto_optimal"] = flags
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Individual plot functions
# ──────────────────────────────────────────────────────────────────────────────

def _sorted_models(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.sort_values(col, ascending=False)


def plot_accuracy_bar(df: pd.DataFrame, out_path: Path, prompt_snippet: str = "") -> None:
    """Bar chart: pass@1 accuracy per model (sorted descending)."""
    sdf = _sorted_models(df, "accuracy")
    models = sdf["model"].tolist()
    values = sdf["accuracy"].tolist()
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(models))]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 0.9), 5))
    bars = ax.bar(models, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("pass@1 (accuracy)", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_title(f"Accuracy per Model\n{_snippet(prompt_snippet)}", fontsize=12)
    ax.tick_params(axis="x", rotation=35)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=9,
        )
    plt.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"[viz] Saved → {out_path}")


def plot_cost_bar(df: pd.DataFrame, out_path: Path, prompt_snippet: str = "") -> None:
    """Bar chart: avg cost per query (USD), sorted descending."""
    sdf = _sorted_models(df, "cost")
    models = sdf["model"].tolist()
    values = sdf["cost"].tolist()
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(models))]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 0.9), 5))
    bars = ax.bar(models, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Avg cost per query (USD)", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_title(f"Cost per Model\n{_snippet(prompt_snippet)}", fontsize=12)
    ax.tick_params(axis="x", rotation=35)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01 + max(values) * 0.005,
            f"${val:.4f}",
            ha="center", va="bottom", fontsize=8,
        )
    plt.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"[viz] Saved → {out_path}")


def plot_latency_bar(df: pd.DataFrame, out_path: Path, prompt_snippet: str = "") -> None:
    """Bar chart: avg latency (seconds) per model, sorted descending."""
    if "latency" not in df.columns:
        print("[viz] Skipping latency bar chart — 'latency' column not available.")
        return

    sdf = _sorted_models(df, "latency")
    models = sdf["model"].tolist()
    values = sdf["latency"].tolist()
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(models))]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 0.9), 5))
    bars = ax.bar(models, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Avg latency (seconds)", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_title(f"Latency per Model\n{_snippet(prompt_snippet)}", fontsize=12)
    ax.tick_params(axis="x", rotation=35)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:.2f}s",
            ha="center", va="bottom", fontsize=9,
        )
    plt.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"[viz] Saved → {out_path}")


def plot_pareto_frontier(
    df: pd.DataFrame,
    out_png: Path,
    out_html: Path,
    prompt_snippet: str = "",
) -> None:
    """Scatter plot of Cost vs Accuracy with Pareto frontier highlighted.

    Saves both a static PNG (matplotlib) and an interactive HTML (plotly).
    """
    pf = compute_pareto_frontier(df)
    _plot_pareto_matplotlib(pf, out_png, prompt_snippet)
    _plot_pareto_plotly(pf, out_html, prompt_snippet)


# ──────────────────────────────────────────────────────────────────────────────
# Pareto — matplotlib (PNG)
# ──────────────────────────────────────────────────────────────────────────────

def _plot_pareto_matplotlib(pf: pd.DataFrame, out_path: Path, prompt_snippet: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot all models
    for _, row in pf.iterrows():
        color = PARETO_COLOR if row["pareto_optimal"] else DEFAULT_COLOR
        ax.scatter(row["cost"], row["accuracy"], color=color, s=120, zorder=3,
                   edgecolors="black", linewidth=0.6)
        ax.annotate(
            row["model"],
            (row["cost"], row["accuracy"]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8, color="#333333",
        )

    # Draw Pareto frontier line connecting optimal models (sorted by cost)
    frontier = pf[pf["pareto_optimal"]].sort_values("cost")
    if len(frontier) > 1:
        ax.plot(frontier["cost"], frontier["accuracy"], color=PARETO_COLOR,
                linewidth=2, linestyle="--", label="Pareto frontier", zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("Avg cost per query (USD) — log scale", fontsize=11)
    ax.set_ylabel("Accuracy (pass@1)", fontsize=11)
    ax.set_title(f"Cost vs Accuracy — Pareto Frontier\n{_snippet(prompt_snippet)}", fontsize=12)
    ax.set_ylim(-0.05, 1.15)

    legend_handles = [
        mpatches.Patch(color=PARETO_COLOR, label="Pareto-optimal"),
        mpatches.Patch(color=DEFAULT_COLOR, label="Sub-optimal"),
    ]
    ax.legend(handles=legend_handles, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"[viz] Saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Pareto — plotly (HTML)
# ──────────────────────────────────────────────────────────────────────────────

def _plot_pareto_plotly(pf: pd.DataFrame, out_path: Path, prompt_snippet: str) -> None:
    try:
        import plotly.graph_objects as go  # type: ignore
    except ImportError:
        print("[viz] plotly not installed — skipping HTML Pareto plot.")
        return

    frontier = pf[pf["pareto_optimal"]].sort_values("cost")
    non_frontier = pf[~pf["pareto_optimal"]]

    fig = go.Figure()

    # Non-Pareto models
    if not non_frontier.empty:
        hover_text = _build_hover(non_frontier)
        fig.add_trace(go.Scatter(
            x=non_frontier["cost"],
            y=non_frontier["accuracy"],
            mode="markers+text",
            name="Sub-optimal",
            text=non_frontier["model"],
            textposition="top right",
            hovertext=hover_text,
            hoverinfo="text",
            marker=dict(color=DEFAULT_COLOR, size=12, line=dict(color="black", width=1)),
        ))

    # Pareto-optimal models
    if not frontier.empty:
        hover_text = _build_hover(frontier)
        fig.add_trace(go.Scatter(
            x=frontier["cost"],
            y=frontier["accuracy"],
            mode="markers+text",
            name="Pareto-optimal",
            text=frontier["model"],
            textposition="top right",
            hovertext=hover_text,
            hoverinfo="text",
            marker=dict(color=PARETO_COLOR, size=14, symbol="star",
                        line=dict(color="black", width=1)),
        ))

        # Frontier line connecting optimal models
        fig.add_trace(go.Scatter(
            x=frontier["cost"], y=frontier["accuracy"],
            mode="lines",
            name="Pareto frontier",
            line=dict(color=PARETO_COLOR, dash="dash", width=2),
            showlegend=True,
        ))

    title_text = (
        f"Cost vs Accuracy — Pareto Frontier<br>"
        f"<sub>{_snippet(prompt_snippet)}</sub>"
    )
    fig.update_layout(
        title=title_text,
        xaxis_title="Avg cost per query (USD) — log scale",
        yaxis_title="Accuracy (pass@1)",
        yaxis=dict(range=[-0.05, 1.2]),
        xaxis=dict(type="log"),
        template="plotly_white",
        font=dict(size=12),
        legend=dict(x=0.01, y=0.99),
        width=900,
        height=600,
    )
    fig.write_html(str(out_path))
    print(f"[viz] Saved → {out_path}")


def _build_hover(df: pd.DataFrame) -> list[str]:
    rows = []
    for _, r in df.iterrows():
        parts = [
            f"<b>{r['model']}</b>",
            f"Accuracy: {r['accuracy']:.3f}",
            f"Cost: ${r['cost']:.5f}",
        ]
        if "latency" in r and pd.notna(r.get("latency")):
            parts.append(f"Latency: {r['latency']:.2f}s")
        rows.append("<br>".join(parts))
    return rows


def _step_coords(xs: list, ys: list) -> tuple[list, list]:
    """Convert (x, y) pairs to step-function coordinates for plotly."""
    step_x, step_y = [], []
    for i in range(len(xs)):
        if i > 0:
            step_x.append(xs[i])
            step_y.append(ys[i - 1])
        step_x.append(xs[i])
        step_y.append(ys[i])
    return step_x, step_y


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def save_all_plots(
    df: pd.DataFrame,
    output_dir: Path,
    prompt_snippet: str = "",
) -> None:
    """Generate and save all 4 plots to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_bar(df, output_dir / "accuracy_bar.png", prompt_snippet)
    plot_cost_bar(df, output_dir / "cost_bar.png", prompt_snippet)
    plot_latency_bar(df, output_dir / "latency_bar.png", prompt_snippet)
    plot_pareto_frontier(
        df,
        out_png=output_dir / "pareto_frontier.png",
        out_html=output_dir / "pareto_frontier.html",
        prompt_snippet=prompt_snippet,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _snippet(text: str, maxlen: int = 80) -> str:
    if not text:
        return ""
    return f'"{text[:maxlen]}{"…" if len(text) > maxlen else ""}"'
