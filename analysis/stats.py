"""Summary statistics for a single-prompt model sweep.

Computes and displays:
    - pass@1 (accuracy)
    - avg cost per query (USD)
    - avg latency in seconds (if available)
    - token efficiency ratio = quality / cost  (higher = better value)
    - input/output token counts (if available)
    - Pareto-optimal flag

Output: printed Rich table (falls back to plain text) + CSV file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .visualize import compute_pareto_frontier


# ──────────────────────────────────────────────────────────────────────────────
# Core statistics function
# ──────────────────────────────────────────────────────────────────────────────

def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-model summary statistics DataFrame.

    Input df must have columns:  model, accuracy, cost
    Optional columns:            latency, input_tokens, output_tokens
    """
    pf = compute_pareto_frontier(df)  # adds 'pareto_optimal' flag

    stats_rows = []
    for _, row in pf.sort_values("accuracy", ascending=False).iterrows():
        r: dict = {
            "model":            row["model"],
            "accuracy (pass@1)": round(float(row["accuracy"]), 4),
            "cost_usd":          round(float(row["cost"]), 6),
        }
        if "latency" in row and pd.notna(row.get("latency")):
            r["latency_s"] = round(float(row["latency"]), 3)
        if "input_tokens" in row and pd.notna(row.get("input_tokens")):
            r["input_tokens"] = int(round(float(row["input_tokens"])))
        if "output_tokens" in row and pd.notna(row.get("output_tokens")):
            r["output_tokens"] = int(round(float(row["output_tokens"])))

        # Token efficiency: quality per dollar spent
        # (undefined if cost == 0; set to inf / 0 gracefully)
        cost = float(row["cost"])
        acc = float(row["accuracy"])
        r["quality_per_dollar"] = round(acc / cost, 2) if cost > 0 else float("inf")

        r["pareto_optimal"] = bool(row["pareto_optimal"])
        stats_rows.append(r)

    return pd.DataFrame(stats_rows)


# ──────────────────────────────────────────────────────────────────────────────
# Print + Save
# ──────────────────────────────────────────────────────────────────────────────

def print_and_save_stats(
    df: pd.DataFrame,
    output_dir: Path,
    prompt_snippet: str = "",
) -> pd.DataFrame:
    """Compute stats, print to console, and save as CSV.

    Returns the stats DataFrame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = compute_summary_stats(df)

    # ── Console output ────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    if prompt_snippet:
        print(f"  Prompt: \"{prompt_snippet[:65]}\"")
    print("  Summary Statistics — per Model")
    print("=" * 70)

    try:
        _print_rich(stats)
    except ImportError:
        _print_plain(stats)

    print("=" * 70)
    print()

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = output_dir / "summary_stats.csv"
    stats.to_csv(csv_path, index=False)
    print(f"[stats] Summary CSV saved → {csv_path}")

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────────────────────

def _print_rich(stats: pd.DataFrame) -> None:
    """Pretty-print using the 'rich' library (optional dependency)."""
    from rich.console import Console  # type: ignore
    from rich.table import Table      # type: ignore

    console = Console()
    table = Table(show_header=True, header_style="bold cyan", box=None)

    table.add_column("Model",              style="bold", no_wrap=True)
    table.add_column("Accuracy",           justify="right")
    table.add_column("Cost (USD)",         justify="right")
    if "latency_s" in stats.columns:
        table.add_column("Latency (s)",    justify="right")
    if "input_tokens" in stats.columns:
        table.add_column("Input tok.",     justify="right")
    if "output_tokens" in stats.columns:
        table.add_column("Output tok.",    justify="right")
    table.add_column("Qual/Dollar",        justify="right")
    table.add_column("Pareto?",            justify="center")

    for _, row in stats.iterrows():
        pareto_str = "[green]✓[/green]" if row["pareto_optimal"] else ""
        cells = [
            str(row["model"]),
            f"{row['accuracy (pass@1)']:.4f}",
            f"${row['cost_usd']:.6f}",
        ]
        if "latency_s" in stats.columns:
            cells.append(f"{row['latency_s']:.3f}")
        if "input_tokens" in stats.columns:
            cells.append(str(row["input_tokens"]))
        if "output_tokens" in stats.columns:
            cells.append(str(row["output_tokens"]))
        q_per_d = row["quality_per_dollar"]
        cells.append("∞" if q_per_d == float("inf") else f"{q_per_d:.2f}")
        cells.append(pareto_str)
        table.add_row(*cells)

    console.print(table)


def _print_plain(stats: pd.DataFrame) -> None:
    """Fallback plain-text table (no external dependencies)."""
    col_order = [c for c in [
        "model", "accuracy (pass@1)", "cost_usd", "latency_s",
        "input_tokens", "output_tokens", "quality_per_dollar", "pareto_optimal",
    ] if c in stats.columns]
    print(stats[col_order].to_string(index=False))
