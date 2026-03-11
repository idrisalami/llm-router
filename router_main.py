"""Router CLI — LLM Router

Usage
-----
    # Train LR router + run cross-validated evaluation
    python3 router_main.py --train

    # Route a single prompt (requires --train to have been run first)
    python3 router_main.py --route "Write a function to sort a list of tuples by the second element."

    # Route with a cost budget (only consider models cheaper than $0.001/query)
    python3 router_main.py --route "..." --budget 0.001

    # Re-compute embeddings (e.g. after dataset update)
    python3 router_main.py --train --recompute-embeddings

    # Compare LR baseline vs joint MLP across tolerance values
    python3 router_main.py --compare
    python3 router_main.py --compare --cv 5
"""

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="router_main.py",
        description="LLM Router",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--train", action="store_true",
                        help="Train LR classifiers + run cross-validated evaluation.")
    parser.add_argument("--compare", action="store_true",
                        help="Compare LR baseline vs joint MLP across tolerance values.")
    parser.add_argument("--route", type=str, default=None, metavar="PROMPT",
                        help="Route a single prompt to the best model.")
    parser.add_argument("--budget", type=float, default=None,
                        help="Max avg cost per query (USD). Only considers models within budget.")
    parser.add_argument("--recompute-embeddings", action="store_true",
                        help="Force re-computation of prompt embeddings.")
    parser.add_argument("--cv", type=int, default=5,
                        help="Number of cross-validation folds (default: 5).")

    args = parser.parse_args(argv)

    if not args.train and not args.route and not args.compare:
        parser.print_help()
        print("\n[error] Provide --train, --route, or --compare.", file=sys.stderr)
        return 1

    from sweep.load_data import load_coding_data
    from router.features import load_or_compute_embeddings

    print("[router] Loading dataset…")
    df = load_coding_data()

    print("[router] Loading / computing MiniLM embeddings…")
    embeddings, prompts = load_or_compute_embeddings(
        df, force_recompute=args.recompute_embeddings
    )
    print(f"[router] {len(prompts)} prompts, embedding dim={embeddings.shape[1]}\n")

    # ── --compare ─────────────────────────────────────────────────────────────
    if args.compare:
        _run_comparison(df, embeddings, prompts, args.cv)
        return 0

    # ── --train ───────────────────────────────────────────────────────────────
    if args.train:
        from router.train import (
            build_training_matrix,
            cross_validate_classifiers,
            save_classifiers,
            train_classifiers,
        )
        from router.evaluate import run_full_evaluation

        X, y_dict, model_costs, ordered_prompts = build_training_matrix(df, embeddings, prompts)
        print(f"[router] Training matrix: {X.shape[0]} prompts × {X.shape[1]} features, {len(y_dict)} models\n")

        cross_validate_classifiers(X, y_dict, cv=args.cv, clf_type="lr")

        print("\n[router] Training final LR classifiers on full dataset…")
        classifiers = train_classifiers(X, y_dict, clf_type="lr")
        save_classifiers(classifiers, model_costs, clf_type="lr")

        run_full_evaluation(df, embeddings, prompts, cv=args.cv,
                            budget=args.budget, clf_types=["lr"])

    # ── --route ───────────────────────────────────────────────────────────────
    if args.route:
        from router.train import load_classifiers
        from router.predict import route

        try:
            classifiers, model_costs = load_classifiers(clf_type="lr")
        except FileNotFoundError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1

        prompt = args.route
        print(f"[router] Routing: {prompt[:100]}")
        if args.budget:
            print(f"[router] Budget: ${args.budget:.4f} / query")

        best_model, scores = route(prompt, classifiers, model_costs, budget=args.budget)

        print(f"\n  → Recommended model: {best_model}")
        print(f"     P(pass) = {scores[best_model]:.3f}   avg_cost = ${model_costs[best_model]:.6f}")
        print()
        print("  All model scores (P(pass) predicted):")
        for model, score in sorted(scores.items(), key=lambda x: -x[1]):
            within_budget = "✓" if args.budget is None or model_costs.get(model, 999) <= args.budget else "✗"
            selected = " ← selected" if model == best_model else ""
            print(f"    {within_budget}  {model:<42}  P(pass)={score:.3f}  cost=${model_costs.get(model, 0):.6f}{selected}")
        print()

    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Comparison: LR baseline vs joint MLP across score_tolerance values
# ──────────────────────────────────────────────────────────────────────────────

def _run_comparison(df, embeddings, prompts, cv: int) -> None:
    """Compare LR baseline and joint MLP across tolerance values."""
    import pandas as pd
    from pathlib import Path
    from router.train import build_training_matrix
    from router.evaluate import evaluate_router_cv, compute_baselines
    from router.mlp_router import evaluate_mlp_router_cv

    OUTPUT_DIR = Path("output/router")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    TOLERANCES = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]

    X, y_dict, model_costs, _ = build_training_matrix(df, embeddings, prompts)
    baselines = compute_baselines(y_dict, model_costs)
    best_cost = baselines["always-best"]["avg_cost"]

    rows = []

    # Baselines
    for name, stats in baselines.items():
        savings = (1 - stats["avg_cost"] / best_cost) * 100
        rows.append({
            "model": name,
            "tolerance": "—",
            "accuracy": round(stats["accuracy"], 4),
            "avg_cost": round(stats["avg_cost"], 6),
            "savings_%": round(savings, 1),
        })

    # LR baseline across tolerance values
    print("\n[compare] Running LR across tolerance values…")
    for tol in TOLERANCES:
        cv_res = evaluate_router_cv(df, embeddings, prompts, cv=cv,
                                    score_tolerance=tol, clf_type="lr")
        acc  = cv_res["actual_quality"].mean()
        cost = cv_res["actual_cost"].mean()
        sav  = (1 - cost / best_cost) * 100
        print(f"  LR  tol={tol:.2f}  acc={acc:.4f}  cost=${cost:.6f}  savings={sav:.1f}%")
        rows.append({"model": "LR (baseline)", "tolerance": tol,
                     "accuracy": round(acc, 4), "avg_cost": round(cost, 6),
                     "savings_%": round(sav, 1)})

    # Joint MLP across tolerance values
    print("\n[compare] Running joint MLP across tolerance values…")
    for tol in TOLERANCES:
        print(f"  MLP tol={tol:.2f}", end=" ")
        cv_res = evaluate_mlp_router_cv(df, embeddings, prompts, cv=cv,
                                        score_tolerance=tol)
        acc  = cv_res["actual_quality"].mean()
        cost = cv_res["actual_cost"].mean()
        sav  = (1 - cost / best_cost) * 100
        print(f" acc={acc:.4f}  cost=${cost:.6f}  savings={sav:.1f}%")
        rows.append({"model": "Joint MLP", "tolerance": tol,
                     "accuracy": round(acc, 4), "avg_cost": round(cost, 6),
                     "savings_%": round(sav, 1)})

    # Print table
    results_df = pd.DataFrame(rows)

    print()
    print("=" * 80)
    print("  LR Baseline vs Joint MLP — tolerance sweep (5-fold CV, MiniLM)")
    print("=" * 80)

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
            is_baseline = row["model"] in baseline_names
            is_mlp      = row["model"] == "Joint MLP"
            style = "dim" if is_baseline else "bold magenta" if is_mlp else ""
            table.add_row(
                str(row["model"]),
                str(row["tolerance"]),
                f"{row['accuracy']:.4f}",
                f"${row['avg_cost']:.6f}",
                f"{row['savings_%']:.1f}%",
                style=style,
            )
        console.print(table)
    except ImportError:
        print(results_df.to_string(index=False))

    print("=" * 80)

    out_path = OUTPUT_DIR / "tolerance_comparison.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n[compare] Results saved → {out_path}")


if __name__ == "__main__":
    sys.exit(main())
