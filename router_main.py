"""LLM Router — Joint MLP

Usage
-----
    # Train MLP on full dataset and save weights
    python3 router_main.py --train

    # Route a single prompt (requires --train first)
    python3 router_main.py --route "Write a function to sort a list of tuples by the second element."

    # Route with a cost budget
    python3 router_main.py --route "..." --budget 0.001

    # Tolerance sweep: compare MLP across tolerance values vs baselines
    python3 router_main.py --compare

    # Analyse embedding spread (re-run after each representation change)
    python3 router_main.py --analyze-embeddings
"""

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="router_main.py",
        description="LLM Router — Joint MLP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--train", action="store_true",
                        help="Train MLP on full dataset and save weights.")
    parser.add_argument("--compare", action="store_true",
                        help="Tolerance sweep: evaluate MLP across tolerance values.")
    parser.add_argument("--route", type=str, default=None, metavar="PROMPT",
                        help="Route a single prompt to the best model.")
    parser.add_argument("--budget", type=float, default=None,
                        help="Max avg cost per query (USD).")
    parser.add_argument("--tolerance", type=float, default=0.10,
                        help="Score tolerance for routing (default: 0.10).")
    parser.add_argument("--recompute-embeddings", action="store_true",
                        help="Force re-computation of prompt embeddings.")
    parser.add_argument("--cv", type=int, default=5,
                        help="Number of cross-validation folds (default: 5).")
    parser.add_argument("--analyze-embeddings", action="store_true",
                        help="Plot embedding spread metrics (re-run after each representation change).")

    args = parser.parse_args(argv)

    if not args.train and not args.route and not args.compare and not args.analyze_embeddings:
        parser.print_help()
        print("\n[error] Provide --train, --route, --compare, or --analyze-embeddings.", file=sys.stderr)
        return 1

    from sweep.load_data import load_coding_data
    from router.features import load_or_compute_embeddings

    print("[router] Loading dataset…")
    df = load_coding_data()

    print("[router] Loading MiniLM embeddings…")
    embeddings, prompts = load_or_compute_embeddings(
        df, force_recompute=args.recompute_embeddings
    )
    print(f"[router] {len(prompts)} prompts, dim={embeddings.shape[1]}\n")

    # ── --train ───────────────────────────────────────────────────────────────
    if args.train:
        from router.mlp_router import build_training_matrix, train_mlp, save_mlp
        import numpy as np

        X, y_dict, model_costs, _ = build_training_matrix(df, embeddings, prompts)
        model_names = list(y_dict.keys())
        Y = np.stack([y_dict[m] for m in model_names], axis=1).astype(np.float32)

        print(f"[router] Training MLP on {X.shape[0]} prompts × {X.shape[1]} features, {len(model_names)} models…")
        mlp = train_mlp(X, Y)
        save_mlp(mlp, model_costs, model_names)

    # ── --analyze-embeddings ──────────────────────────────────────────────────
    if args.analyze_embeddings:
        from analysis.embedding_analysis import (
            compute_spread_metrics, print_metrics, plot_embedding_spread,
        )
        metrics = compute_spread_metrics(embeddings)
        print_metrics(metrics, label="MiniLM (all-MiniLM-L6-v2, 384-dim)")
        plot_embedding_spread(embeddings, label="MiniLM", metrics=metrics)

    # ── --compare ─────────────────────────────────────────────────────────────
    if args.compare:
        _run_tolerance_sweep(df, embeddings, prompts, args.cv)

    # ── --route ───────────────────────────────────────────────────────────────
    if args.route:
        from router.mlp_router import load_mlp, _scores_from_mlp, select_model

        try:
            mlp, model_costs, model_names = load_mlp(input_dim=embeddings.shape[1])
        except FileNotFoundError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1

        from router.features import embed_single
        embedding = embed_single(args.route)
        scores    = _scores_from_mlp(mlp, embedding, model_names)
        best      = select_model(scores, model_costs, args.budget, args.tolerance)

        print(f"[router] Routing: {args.route[:100]}")
        if args.budget:
            print(f"[router] Budget: ${args.budget:.4f} / query")
        print(f"[router] Tolerance: {args.tolerance}")
        print(f"\n  → Recommended model: {best}")
        print(f"     P(pass) = {scores[best]:.3f}   avg_cost = ${model_costs[best]:.6f}")
        print()
        print("  All scores:")
        for m, s in sorted(scores.items(), key=lambda x: -x[1]):
            within = "✓" if args.budget is None or model_costs.get(m, 999) <= args.budget else "✗"
            tag    = " ← selected" if m == best else ""
            print(f"    {within}  {m:<42}  P(pass)={s:.3f}  cost=${model_costs.get(m, 0):.6f}{tag}")
        print()

    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Tolerance sweep
# ──────────────────────────────────────────────────────────────────────────────

def _run_tolerance_sweep(df, embeddings, prompts, cv: int) -> None:
    import pandas as pd
    from pathlib import Path
    from router.mlp_router import build_training_matrix, compute_baselines, evaluate_mlp_router_cv, evaluate_prior_router_cv

    OUTPUT_DIR = Path("output/router")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    TOLERANCES = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]

    _, y_dict, model_costs, _ = build_training_matrix(df, embeddings, prompts)
    baselines = compute_baselines(y_dict, model_costs)
    best_cost = baselines["always-best"]["avg_cost"]

    rows = []
    for name, stats in baselines.items():
        sav = (1 - stats["avg_cost"] / best_cost) * 100
        rows.append({"model": name, "tolerance": "—",
                     "accuracy": round(stats["accuracy"], 4),
                     "avg_cost": round(stats["avg_cost"], 6),
                     "savings_%": round(sav, 1)})

    print("[compare] Running no-embedding baseline (prior) across tolerance values…")
    for tol in TOLERANCES:
        cv_res = evaluate_prior_router_cv(df, embeddings, prompts, cv=cv, score_tolerance=tol)
        acc  = cv_res["actual_quality"].mean()
        cost = cv_res["actual_cost"].mean()
        sav  = (1 - cost / best_cost) * 100
        print(f"  prior tol={tol:.2f}  acc={acc:.4f}  cost=${cost:.6f}  savings={sav:.1f}%")
        rows.append({"model": "Prior (no embedding)", "tolerance": tol,
                     "accuracy": round(acc, 4), "avg_cost": round(cost, 6),
                     "savings_%": round(sav, 1)})

    print("[compare] Running MLP tolerance sweep…")
    for tol in TOLERANCES:
        print(f"  tol={tol:.2f}", end=" ")
        cv_res = evaluate_mlp_router_cv(df, embeddings, prompts, cv=cv, score_tolerance=tol)
        acc  = cv_res["actual_quality"].mean()
        cost = cv_res["actual_cost"].mean()
        sav  = (1 - cost / best_cost) * 100
        print(f"  acc={acc:.4f}  cost=${cost:.6f}  savings={sav:.1f}%")
        rows.append({"model": "Joint MLP", "tolerance": tol,
                     "accuracy": round(acc, 4), "avg_cost": round(cost, 6),
                     "savings_%": round(sav, 1)})

    results_df = pd.DataFrame(rows)

    print()
    print("=" * 75)
    print("  Joint MLP — Tolerance Sweep (5-fold CV, MiniLM embeddings)")
    print("=" * 75)

    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Model",            style="bold", min_width=20)
        table.add_column("Tolerance",        justify="right")
        table.add_column("Accuracy",         justify="right")
        table.add_column("Avg cost (USD)",   justify="right")
        table.add_column("Savings vs GPT-4", justify="right")
        baseline_names = set(baselines.keys())
        for _, row in results_df.iterrows():
            style = "dim" if row["model"] in baseline_names else "bold magenta"
            table.add_row(str(row["model"]), str(row["tolerance"]),
                          f"{row['accuracy']:.4f}", f"${row['avg_cost']:.6f}",
                          f"{row['savings_%']:.1f}%", style=style)
        console.print(table)
    except ImportError:
        print(results_df.to_string(index=False))

    print("=" * 75)

    out_path = OUTPUT_DIR / "tolerance_sweep.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n[compare] Results saved → {out_path}")

    _plot_cost_accuracy(results_df, baselines, y_dict, model_costs, OUTPUT_DIR)


# ──────────────────────────────────────────────────────────────────────────────
# Cost-accuracy scatter plot
# ──────────────────────────────────────────────────────────────────────────────

def _plot_cost_accuracy(
    results_df,
    baselines: dict,
    y_dict: dict,
    model_costs: dict,
    output_dir,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    output_dir = Path(output_dir)
    fig, ax = plt.subplots(figsize=(13, 7))

    # ── Individual models (grey, small) ───────────────────────────────────────
    for model, y in y_dict.items():
        acc  = float(y.mean())
        cost = model_costs[model]
        ax.scatter(cost, acc, color="#CCCCCC", s=60, zorder=2)
        ax.annotate(model.split("/")[-1], (cost, acc),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7, color="#999999")

    # ── Baselines ─────────────────────────────────────────────────────────────
    baseline_styles = {
        "always-best":     ("#4C72B0", "Always GPT-4"),
        "always-cheapest": ("#DD8452", "Always cheapest"),
        "random":          ("#8172B3", "Random"),
        "oracle":          ("#55A868", "Oracle (ceiling)"),
    }
    for name, stats in baselines.items():
        color, label = baseline_styles[name]
        ax.scatter(stats["avg_cost"], stats["accuracy"],
                   color=color, s=160, zorder=4,
                   edgecolors="black", linewidth=0.8, label=label)

    # ── Prior (no-embedding) baseline across tolerances ───────────────────────
    prior_rows = results_df[results_df["model"] == "Prior (no embedding)"].copy()
    prior_rows = prior_rows.sort_values("avg_cost")
    ax.plot(prior_rows["avg_cost"].values, prior_rows["accuracy"].values,
            color="#888888", linewidth=1.2, linestyle=":", alpha=0.6, zorder=3)
    for _, row in prior_rows.iterrows():
        ax.scatter(row["avg_cost"], row["accuracy"],
                   color="#AAAAAA", s=120, marker="D", zorder=4,
                   edgecolors="#555555", linewidth=0.7)
    ax.scatter([], [], color="#AAAAAA", s=120, marker="D",
               edgecolors="#555555", linewidth=0.7, label="Prior/no-embed (↑tol → cheaper)")

    # ── MLP router across tolerances (connected trade-off curve) ──────────────
    mlp_rows = results_df[results_df["model"] == "Joint MLP"].copy()
    mlp_rows = mlp_rows.sort_values("avg_cost")  # left-to-right along cost axis

    tol_colors = plt.cm.plasma(
        np.linspace(0.15, 0.85, len(mlp_rows))
    )

    # Draw connecting line first (behind dots)
    ax.plot(mlp_rows["avg_cost"].values, mlp_rows["accuracy"].values,
            color="#C44E52", linewidth=1.5, linestyle="--", alpha=0.5, zorder=3)

    for (_, row), color in zip(mlp_rows.iterrows(), tol_colors):
        tol = row["tolerance"]
        ax.scatter(row["avg_cost"], row["accuracy"],
                   color=color, s=220, marker="*", zorder=5,
                   edgecolors="black", linewidth=0.7)
        ax.annotate(f"MLP tol={tol:.2f}", (row["avg_cost"], row["accuracy"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8, fontweight="bold", color="#C44E52")

    # Dummy entry for legend
    ax.scatter([], [], color="#C44E52", s=220, marker="*",
               edgecolors="black", linewidth=0.7, label="MLP router (↑tol → cheaper)")

    ax.set_xscale("log")
    ax.set_xlabel("Avg cost per query (USD) — log scale", fontsize=11)
    ax.set_ylabel("Accuracy (pass@1)", fontsize=11)
    ax.set_title(
        "Joint MLP Router — Cost vs Accuracy Trade-off\n"
        "(each ★ = one tolerance setting; right = expensive/accurate, left = cheap/lower accuracy)",
        fontsize=11,
    )
    ax.set_ylim(0.1, 0.95)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "router_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[compare] Plot saved → {out_path}")


if __name__ == "__main__":
    sys.exit(main())
