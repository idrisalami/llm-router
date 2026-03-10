"""Cross-validated evaluation of the router against baselines.

Baselines
---------
  always-best    : always use the single model with the highest average accuracy (GPT-4)
  always-cheapest: always use the cheapest model (mistral-7b)
  random         : pick a random model per prompt
  oracle         : cheapest model that actually passes (theoretical ceiling)
  router (lr)    : logistic regression router (cross-validated)
  router (gb)    : gradient boosting router (cross-validated)

Metrics
-------
  accuracy : fraction of prompts where the chosen model passed
  avg_cost : average cost of the chosen model per prompt
  savings  : cost saved vs always-best (%)
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold  # type: ignore

from .train import (
    MODELS,
    build_training_matrix,
    train_classifiers,
)
from .predict import route_embedding

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "router"


# ──────────────────────────────────────────────────────────────────────────────
# Cross-validated router evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_router_cv(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    prompts: list[str],
    cv: int = 5,
    budget: float | None = None,
    score_tolerance: float = 0.0,
    clf_type: str = "lr",
) -> pd.DataFrame:
    """Run k-fold CV and return a DataFrame of per-prompt routing decisions.

    Columns: prompt, fold, routed_model, actual_quality, actual_cost
    """
    X, y_dict, model_costs, ordered_prompts = build_training_matrix(df, embeddings, prompts)
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train = X[train_idx]
        y_train = {m: y[train_idx] for m, y in y_dict.items()}

        fold_classifiers = train_classifiers(X_train, y_train, clf_type=clf_type)

        for j in test_idx:
            routed_model, scores = route_embedding(
                X[j], fold_classifiers, model_costs, budget, score_tolerance
            )
            rows.append({
                "prompt":         ordered_prompts[j],
                "fold":           fold_idx,
                "routed_model":   routed_model,
                "actual_quality": y_dict[routed_model][j],
                "actual_cost":    model_costs[routed_model],
                "scores":         scores,
            })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Baseline metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_baselines(
    y_dict: dict[str, np.ndarray],
    model_costs: dict[str, float],
) -> dict[str, dict]:
    """Compute accuracy and avg_cost for each baseline strategy."""
    n = len(next(iter(y_dict.values())))

    # always-best: model with highest mean accuracy
    best_model = max(y_dict, key=lambda m: y_dict[m].mean())
    always_best = {
        "accuracy": float(y_dict[best_model].mean()),
        "avg_cost": model_costs[best_model],
        "model":    best_model,
    }

    # always-cheapest: model with lowest avg cost
    cheapest_model = min(model_costs, key=lambda m: model_costs[m])
    always_cheapest = {
        "accuracy": float(y_dict[cheapest_model].mean()),
        "avg_cost": model_costs[cheapest_model],
        "model":    cheapest_model,
    }

    # random routing
    random.seed(42)
    model_list = list(y_dict.keys())
    random_acc  = np.mean([y_dict[random.choice(model_list)][i] for i in range(n)])
    random_cost = np.mean([model_costs[random.choice(model_list)] for _ in range(n)])
    rand_baseline = {"accuracy": float(random_acc), "avg_cost": float(random_cost), "model": "random"}

    # oracle: cheapest model that actually passes (per prompt)
    oracle_accs, oracle_costs = [], []
    for i in range(n):
        passing = [m for m in model_list if y_dict[m][i] == 1.0]
        if passing:
            cheapest_passing = min(passing, key=lambda m: model_costs[m])
            oracle_accs.append(1.0)
            oracle_costs.append(model_costs[cheapest_passing])
        else:
            oracle_accs.append(0.0)
            oracle_costs.append(0.0)  # no cost if nothing passes
    oracle = {
        "accuracy": float(np.mean(oracle_accs)),
        "avg_cost": float(np.mean(oracle_costs)),
        "model":    "oracle",
    }

    return {
        "always-best":     always_best,
        "always-cheapest": always_cheapest,
        "random":          rand_baseline,
        "oracle":          oracle,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def print_report(
    cv_results: pd.DataFrame,
    baselines: dict[str, dict],
    model_costs: dict[str, float],
) -> pd.DataFrame:
    """Print comparison table and return as DataFrame."""
    router_acc  = cv_results["actual_quality"].mean()
    router_cost = cv_results["actual_cost"].mean()

    best_cost = baselines["always-best"]["avg_cost"]

    all_results = {**baselines, "router (ours)": {"accuracy": router_acc, "avg_cost": router_cost, "model": "router"}}

    rows = []
    for name, stats in all_results.items():
        savings = (1 - stats["avg_cost"] / best_cost) * 100 if best_cost > 0 else 0
        rows.append({
            "strategy":   name,
            "accuracy":   round(stats["accuracy"], 4),
            "avg_cost":   round(stats["avg_cost"], 6),
            "savings_vs_best_%": round(savings, 1),
        })

    results_df = pd.DataFrame(rows)

    print()
    print("=" * 70)
    print("  Router Evaluation — Cross-Validated (5-fold)")
    print("=" * 70)

    try:
        from rich.console import Console  # type: ignore
        from rich.table import Table      # type: ignore
        console = Console()
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Strategy",            style="bold", min_width=20)
        table.add_column("Accuracy",            justify="right")
        table.add_column("Avg cost (USD)",      justify="right")
        table.add_column("Savings vs best (%)", justify="right")
        for _, row in results_df.iterrows():
            style = "bold green" if row["strategy"] == "router (ours)" else ""
            table.add_row(
                row["strategy"],
                f"{row['accuracy']:.4f}",
                f"${row['avg_cost']:.6f}",
                f"{row['savings_vs_best_%']:.1f}%",
                style=style,
            )
        console.print(table)
    except ImportError:
        print(results_df.to_string(index=False))

    print("=" * 70)
    print()

    # Model selection frequency
    freq = cv_results["routed_model"].value_counts()
    print("  Router model selection frequency:")
    for model, count in freq.items():
        pct = count / len(cv_results) * 100
        print(f"    {model:<42} {count:>4} prompts  ({pct:.1f}%)")
    print()

    return results_df


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_comparison(
    router_variants: list[tuple[str, pd.DataFrame]],
    baselines: dict[str, dict],
    model_costs: dict[str, float],
    y_dict: dict[str, np.ndarray],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Plot all strategies + individual models on the cost-accuracy plane.

    router_variants: list of (label, cv_results_df) — one entry per router config.
    Labels starting with "gb" or containing "gb" are drawn in blue; others in red.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7))

    # ── Individual model dots (grey, small) ───────────────────────────────────
    for model in y_dict:
        acc  = float(y_dict[model].mean())
        cost = model_costs[model]
        ax.scatter(cost, acc, color="#CCCCCC", s=60, zorder=2)
        ax.annotate(model.split("/")[-1], (cost, acc),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7, color="#999999")

    # ── Baseline dots ─────────────────────────────────────────────────────────
    baseline_colors = {
        "always-best":     "#4C72B0",
        "always-cheapest": "#DD8452",
        "random":          "#8172B3",
        "oracle":          "#55A868",
    }
    for name, stats in baselines.items():
        ax.scatter(stats["avg_cost"], stats["accuracy"],
                   color=baseline_colors[name], s=130, zorder=4,
                   edgecolors="black", linewidth=0.8, label=name)

    # ── Router variants ───────────────────────────────────────────────────────
    # LR variants in shades of red, GB variants in shades of blue
    lr_colors = ["#E63946", "#C1121F", "#6D1A2A"]
    gb_colors = ["#1D70B8", "#0A4A8A", "#062040"]
    lr_idx = gb_idx = 0

    lr_points: list[tuple[float, float]] = []
    gb_points: list[tuple[float, float]] = []

    for label, cv_results in router_variants:
        acc  = cv_results["actual_quality"].mean()
        cost = cv_results["actual_cost"].mean()
        is_gb = "gb" in label.lower()
        if is_gb:
            color = gb_colors[gb_idx % len(gb_colors)]
            gb_idx += 1
            gb_points.append((cost, acc))
        else:
            color = lr_colors[lr_idx % len(lr_colors)]
            lr_idx += 1
            lr_points.append((cost, acc))
        ax.scatter(cost, acc, color=color, s=240, marker="*", zorder=5,
                   edgecolors="black", linewidth=0.8, label=label)
        ax.annotate(label, (cost, acc),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=8, fontweight="bold", color=color)

    # Connect variants within each family with a dashed trade-off line
    if len(lr_points) > 1:
        xs, ys = zip(*lr_points)
        ax.plot(xs, ys, color="#E63946", linewidth=1.2, linestyle="--", alpha=0.5, zorder=4)
    if len(gb_points) > 1:
        xs, ys = zip(*gb_points)
        ax.plot(xs, ys, color="#1D70B8", linewidth=1.2, linestyle="--", alpha=0.5, zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("Avg cost per query (USD) — log scale", fontsize=11)
    ax.set_ylabel("Accuracy (pass@1)", fontsize=11)
    ax.set_title("Router vs Baselines — Cost-Accuracy Trade-off\n(★ red = LR, ★ blue = GradientBoosting)", fontsize=12)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "router_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[eval] Saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Detailed router analysis plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_router_analysis(
    cv_results: pd.DataFrame,
    y_dict: dict[str, np.ndarray],
    model_costs: dict[str, float],
    output_dir: Path = OUTPUT_DIR,
    filename: str = "router_analysis.png",
) -> None:
    """Four-panel diagnostic plot for the router's routing decisions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Short display names (strip org prefix)
    def short(m: str) -> str:
        return m.split("/")[-1]

    # Predicted confidence for the chosen model on each prompt
    cv_results = cv_results.copy()
    cv_results["chosen_score"] = cv_results.apply(
        lambda r: r["scores"].get(r["routed_model"], float("nan")), axis=1
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Router Decision Analysis (5-fold CV)", fontsize=13, fontweight="bold")

    # ── Panel 1: Selection frequency stacked by pass / fail ───────────────────
    ax = axes[0, 0]
    sel = cv_results.groupby("routed_model")["actual_quality"].agg(
        passed=lambda x: (x == 1).sum(),
        failed=lambda x: (x == 0).sum(),
    ).reset_index()
    sel["total"] = sel["passed"] + sel["failed"]
    sel = sel.sort_values("total", ascending=False)
    labels = [short(m) for m in sel["routed_model"]]
    x = np.arange(len(labels))
    ax.bar(x, sel["passed"], label="passed", color="#55A868")
    ax.bar(x, sel["failed"], bottom=sel["passed"], label="failed", color="#E63946")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("# prompts routed")
    ax.set_title("Selection frequency (pass vs fail)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 2: Pass rate for each routed model ───────────────────────────────
    ax = axes[0, 1]
    sel["pass_rate"] = sel["passed"] / sel["total"]
    # Also show overall accuracy of each model in the dataset (grey dashed)
    bar_colors = ["#55A868" if r >= 0.5 else "#E63946" for r in sel["pass_rate"]]
    ax.bar(x, sel["pass_rate"], color=bar_colors, edgecolor="black", linewidth=0.5)
    # Overlay dataset-wide accuracy for each model (dashed grey)
    for i, model in enumerate(sel["routed_model"]):
        if model in y_dict:
            overall = float(y_dict[model].mean())
            ax.plot([i - 0.4, i + 0.4], [overall, overall],
                    color="#888888", linewidth=1.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Pass rate when routed here")
    ax.set_ylim(0, 1.05)
    ax.set_title("Pass rate by routed model\n(dashed = dataset-wide accuracy)")
    ax.axhline(cv_results["actual_quality"].mean(), color="#E63946",
               linewidth=1.5, linestyle=":", label=f"router avg ({cv_results['actual_quality'].mean():.2f})")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 3: Per-fold router accuracy ─────────────────────────────────────
    ax = axes[1, 0]
    fold_acc = cv_results.groupby("fold")["actual_quality"].mean()
    fold_colors = ["#4C72B0"] * len(fold_acc)
    ax.bar(fold_acc.index, fold_acc.values, color=fold_colors, edgecolor="black", linewidth=0.5)
    ax.axhline(cv_results["actual_quality"].mean(), color="#E63946",
               linewidth=1.5, linestyle="--", label=f"mean ({cv_results['actual_quality'].mean():.3f})")
    ax.set_xticks(fold_acc.index)
    ax.set_xticklabels([f"Fold {i}" for i in fold_acc.index])
    ax.set_ylabel("Accuracy (pass rate)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Router accuracy per fold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 4: Confidence histogram (P(pass) of chosen model) ───────────────
    ax = axes[1, 1]
    passed = cv_results[cv_results["actual_quality"] == 1]["chosen_score"].dropna()
    failed = cv_results[cv_results["actual_quality"] == 0]["chosen_score"].dropna()
    bins = np.linspace(0, 1, 21)
    ax.hist(passed, bins=bins, alpha=0.65, color="#55A868", label=f"passed (n={len(passed)})")
    ax.hist(failed, bins=bins, alpha=0.65, color="#E63946", label=f"failed (n={len(failed)})")
    ax.set_xlabel("Predicted P(pass) for chosen model")
    ax.set_ylabel("# prompts")
    ax.set_title("Router confidence at routing time\n(did higher confidence → more passes?)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[eval] Saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

ROUTER_VARIANTS = [
    ("router tol=0.00", 0.00),
    ("router tol=0.02", 0.02),
    ("router tol=0.05", 0.05),
]


def run_full_evaluation(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    prompts: list[str],
    cv: int = 5,
    budget: float | None = None,
    clf_types: list[str] | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Run full evaluation for one or more classifier types.

    clf_types: list of "lr", "gb", or both (default: ["lr", "gb"]).
    """
    if clf_types is None:
        clf_types = ["lr", "gb"]

    X, y_dict, model_costs, ordered_prompts = build_training_matrix(df, embeddings, prompts)
    baselines = compute_baselines(y_dict, model_costs)
    best_cost = baselines["always-best"]["avg_cost"]

    # ── Run CV for all classifier × tolerance combinations ────────────────────
    router_variant_results: list[tuple[str, pd.DataFrame]] = []
    summary_rows: list[dict] = []

    for clf_type in clf_types:
        clf_label = "GB" if clf_type == "gb" else "LR"
        for tol_label, tol in ROUTER_VARIANTS:
            label = f"{clf_label} {tol_label}"
            print(f"\n[eval] Running {cv}-fold CV — {label}…")
            cv_res = evaluate_router_cv(
                df, embeddings, prompts, cv=cv,
                budget=budget, score_tolerance=tol, clf_type=clf_type,
            )
            router_variant_results.append((label, cv_res))
            acc  = cv_res["actual_quality"].mean()
            cost = cv_res["actual_cost"].mean()
            savings = (1 - cost / best_cost) * 100
            print(f"         accuracy={acc:.4f}  avg_cost=${cost:.6f}  savings={savings:.1f}%")
            summary_rows.append({
                "strategy":         label,
                "clf_type":         clf_type,
                "tolerance":        tol,
                "accuracy":         round(acc, 4),
                "avg_cost":         round(cost, 6),
                "savings_vs_best_%": round(savings, 1),
            })

    # ── Print combined comparison table ───────────────────────────────────────
    print()
    print("=" * 75)
    print("  LR vs GB — Combined Comparison (5-fold CV)")
    print("=" * 75)

    # Baselines first
    baseline_rows = []
    for name, stats in baselines.items():
        savings = (1 - stats["avg_cost"] / best_cost) * 100
        baseline_rows.append({
            "strategy": name, "clf_type": "-", "tolerance": "-",
            "accuracy": round(stats["accuracy"], 4),
            "avg_cost": round(stats["avg_cost"], 6),
            "savings_vs_best_%": round(savings, 1),
        })

    all_rows = baseline_rows + summary_rows
    all_df = pd.DataFrame(all_rows)

    try:
        from rich.console import Console  # type: ignore
        from rich.table import Table      # type: ignore
        console = Console()
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Strategy",            style="bold", min_width=24)
        table.add_column("Accuracy",            justify="right")
        table.add_column("Avg cost (USD)",      justify="right")
        table.add_column("Savings vs best (%)", justify="right")
        for _, row in all_df.iterrows():
            is_gb  = "GB" in str(row["strategy"])
            is_lr  = "LR" in str(row["strategy"])
            style  = "bold blue" if is_gb else ("bold red" if is_lr else "")
            table.add_row(
                row["strategy"],
                f"{row['accuracy']:.4f}",
                f"${row['avg_cost']:.6f}",
                f"{row['savings_vs_best_%']:.1f}%",
                style=style,
            )
        console.print(table)
    except ImportError:
        print(all_df.to_string(index=False))

    print("=" * 75)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Save per-variant CV results ────────────────────────────────────────────
    for label, cv_res in router_variant_results:
        slug = label.replace("=", "").replace(".", "").replace(" ", "_")
        csv_path = output_dir / f"cv_results_{slug}.csv"
        cv_res.drop(columns=["scores"]).to_csv(csv_path, index=False)
        print(f"[eval] CV results saved → {csv_path}")

    summary_path = output_dir / "comparison_summary.csv"
    all_df.to_csv(summary_path, index=False)
    print(f"[eval] Summary saved → {summary_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_comparison(router_variant_results, baselines, model_costs, y_dict, output_dir)

    # Analysis plot: use the LR quality variant if available, else first variant
    lr_quality = next(
        ((l, cv) for l, cv in router_variant_results if "LR" in l and "0.00" in l),
        router_variant_results[0],
    )
    plot_router_analysis(lr_quality[1], y_dict, model_costs, output_dir)

    # If GB also ran, plot its analysis separately
    gb_quality = next(
        ((l, cv) for l, cv in router_variant_results if "GB" in l and "0.00" in l),
        None,
    )
    if gb_quality:
        plot_router_analysis(gb_quality[1], y_dict, model_costs,
                             output_dir=output_dir, filename="router_analysis_gb.png")
