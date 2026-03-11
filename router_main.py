"""Router CLI — Phase 3 & 4: LLM Router

Usage
-----
    # Train both LR and GB routers + run cross-validated evaluation (default)
    python3 router_main.py --train

    # Train only logistic regression
    python3 router_main.py --train --classifier lr

    # Train only gradient boosting
    python3 router_main.py --train --classifier gb

    # Route a single prompt (requires --train to have been run first)
    python3 router_main.py --route "Write a function to sort a list of tuples by the second element."

    # Route with a cost budget (only consider models cheaper than $0.001/query)
    python3 router_main.py --route "..." --budget 0.001

    # Route using a specific classifier
    python3 router_main.py --route "..." --classifier gb

    # Re-compute embeddings (e.g. after dataset update)
    python3 router_main.py --train --recompute-embeddings

    # Phase 4: Compare all embedding models + oracle multi-class approach
    python3 router_main.py --compare
    python3 router_main.py --compare --cv 5
"""

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="router_main.py",
        description="LLM Router — Phase 3 & 4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--train", action="store_true",
                        help="Train classifiers + run cross-validated evaluation.")
    parser.add_argument("--compare", action="store_true",
                        help="Run full embedding comparison: MiniLM vs CodeBERT vs CodeT5, "
                             "binary LR vs oracle multi-class. Prints summary table.")
    parser.add_argument("--route", type=str, default=None, metavar="PROMPT",
                        help="Route a single prompt to the best model.")
    parser.add_argument("--budget", type=float, default=None,
                        help="Max avg cost per query (USD). Only considers models within budget.")
    parser.add_argument("--recompute-embeddings", action="store_true",
                        help="Force re-computation of prompt embeddings.")
    parser.add_argument("--cv", type=int, default=5,
                        help="Number of cross-validation folds (default: 5).")
    parser.add_argument("--classifier", choices=["lr", "gb", "both"], default="both",
                        help="Classifier type: lr, gb, or both (default: both).")
    parser.add_argument("--embedding", choices=["minilm", "codebert", "graphcodebert"], default="minilm",
                        help="Embedding model for --train/--route (default: minilm).")

    args = parser.parse_args(argv)

    if not args.train and not args.route and not args.compare:
        parser.print_help()
        print("\n[error] Provide --train, --route, or --compare.", file=sys.stderr)
        return 1

    # Resolve classifier list
    if args.classifier == "both":
        clf_types = ["lr", "gb"]
    else:
        clf_types = [args.classifier]

    # ── Shared: load dataset ───────────────────────────────────────────────────
    from sweep.load_data import load_coding_data
    from router.features import load_or_compute_embeddings

    print("[router] Loading dataset…")
    df = load_coding_data()

    # ── --compare ─────────────────────────────────────────────────────────────
    if args.compare:
        _run_comparison(df, args.cv, args.recompute_embeddings)
        return 0

    # ── --train / --route: load selected embeddings ───────────────────────────
    print(f"[router] Loading / computing {args.embedding} embeddings…")
    embeddings, prompts = load_or_compute_embeddings(
        df, model=args.embedding, force_recompute=args.recompute_embeddings
    )
    print(f"[router] {len(prompts)} prompts, embedding dim={embeddings.shape[1]}\n")

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

        for clf_type in clf_types:
            cross_validate_classifiers(X, y_dict, cv=args.cv, clf_type=clf_type)

        for clf_type in clf_types:
            clf_label = "Gradient Boosting" if clf_type == "gb" else "Logistic Regression"
            print(f"\n[router] Training final {clf_label} classifiers on full dataset…")
            classifiers = train_classifiers(X, y_dict, clf_type=clf_type)
            save_classifiers(classifiers, model_costs, clf_type=clf_type)

        run_full_evaluation(df, embeddings, prompts, cv=args.cv,
                            budget=args.budget, clf_types=clf_types)

    # ── --route ───────────────────────────────────────────────────────────────
    if args.route:
        from router.train import load_classifiers
        from router.predict import route

        clf_type = clf_types[0]
        clf_label = "Gradient Boosting" if clf_type == "gb" else "Logistic Regression"

        try:
            classifiers, model_costs = load_classifiers(clf_type=clf_type)
        except FileNotFoundError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1

        prompt = args.route
        print(f"[router] Routing ({clf_label}): {prompt[:100]}")
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
# Comparison experiment
# ──────────────────────────────────────────────────────────────────────────────

def _run_comparison(df, cv: int, force_recompute: bool) -> None:
    """Run all embedding × classifier combinations and print a summary table."""
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from router.features import load_or_compute_embeddings
    from router.train import build_training_matrix
    from router.evaluate import evaluate_router_cv, compute_baselines
    from router.oracle_clf import evaluate_oracle_clf_cv, print_oracle_label_distribution

    OUTPUT_DIR = Path("output/router")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    EMBEDDING_MODELS = ["minilm", "codebert", "graphcodebert"]
    CLF_TYPES        = ["lr", "gb"]

    # ── Compute baselines once (from MiniLM data) ─────────────────────────────
    print("\n[compare] Computing baselines…")
    emb_base, prompts_base = load_or_compute_embeddings(df, model="minilm",
                                                        force_recompute=force_recompute)
    X_base, y_dict, model_costs, _ = build_training_matrix(df, emb_base, prompts_base)
    baselines = compute_baselines(y_dict, model_costs)
    best_cost = baselines["always-best"]["avg_cost"]

    print_oracle_label_distribution(y_dict, model_costs)

    rows = []

    # ── Add baselines to the table ─────────────────────────────────────────────
    for name, stats in baselines.items():
        savings = (1 - stats["avg_cost"] / best_cost) * 100
        rows.append({
            "experiment":        name,
            "embedding":         "—",
            "classifier":        "—",
            "accuracy":          round(stats["accuracy"], 4),
            "avg_cost":          round(stats["avg_cost"], 6),
            "savings_%":         round(savings, 1),
        })

    # ── Binary LR/GB × each embedding ─────────────────────────────────────────
    for emb_name in EMBEDDING_MODELS:
        print(f"\n[compare] Loading {emb_name} embeddings…")
        embeddings, prompts = load_or_compute_embeddings(df, model=emb_name,
                                                         force_recompute=force_recompute)
        print(f"          dim={embeddings.shape[1]}")

        for clf_type in CLF_TYPES:
            label = f"binary {clf_type.upper()} + {emb_name}"
            print(f"[compare] Running CV — {label}…")
            cv_res = evaluate_router_cv(
                df, embeddings, prompts, cv=cv,
                score_tolerance=0.0, clf_type=clf_type,
            )
            acc  = cv_res["actual_quality"].mean()
            cost = cv_res["actual_cost"].mean()
            savings = (1 - cost / best_cost) * 100
            print(f"          accuracy={acc:.4f}  cost=${cost:.6f}  savings={savings:.1f}%")
            rows.append({
                "experiment": label,
                "embedding":  emb_name,
                "classifier": clf_type,
                "accuracy":   round(acc, 4),
                "avg_cost":   round(cost, 6),
                "savings_%":  round(savings, 1),
            })

    # ── Oracle multi-class × each embedding ───────────────────────────────────
    for emb_name in EMBEDDING_MODELS:
        embeddings, prompts = load_or_compute_embeddings(df, model=emb_name,
                                                         force_recompute=False)
        for clf_type in CLF_TYPES:
            label = f"oracle {clf_type.upper()} + {emb_name}"
            print(f"[compare] Running CV — {label}…")
            cv_res = evaluate_oracle_clf_cv(
                df, embeddings, prompts, cv=cv, clf_type=clf_type,
            )
            acc  = cv_res["actual_quality"].mean()
            cost = cv_res["actual_cost"].mean()
            savings = (1 - cost / best_cost) * 100
            print(f"          accuracy={acc:.4f}  cost=${cost:.6f}  savings={savings:.1f}%")
            rows.append({
                "experiment": label,
                "embedding":  emb_name,
                "classifier": clf_type,
                "accuracy":   round(acc, 4),
                "avg_cost":   round(cost, 6),
                "savings_%":  round(savings, 1),
            })

    # ── Print table ────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(rows)

    print()
    print("=" * 85)
    print("  Embedding × Classifier Comparison (5-fold CV, tol=0.00)")
    print("=" * 85)

    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Experiment",     style="bold", min_width=30)
        table.add_column("Accuracy",       justify="right")
        table.add_column("Avg cost (USD)", justify="right")
        table.add_column("Savings vs GPT-4", justify="right")

        baseline_names = set(baselines.keys())
        for _, row in results_df.iterrows():
            is_baseline = row["experiment"] in baseline_names
            is_oracle   = "oracle" in str(row["experiment"])
            is_best     = row["accuracy"] == results_df[~results_df["experiment"].isin(baseline_names)]["accuracy"].max()
            style = ("dim" if is_baseline
                     else "bold green" if is_best
                     else "bold magenta" if is_oracle
                     else "")
            table.add_row(
                row["experiment"],
                f"{row['accuracy']:.4f}",
                f"${row['avg_cost']:.6f}",
                f"{row['savings_%']:.1f}%",
                style=style,
            )
        console.print(table)
    except ImportError:
        print(results_df.to_string(index=False))

    print("=" * 85)

    # Save
    out_path = OUTPUT_DIR / "embedding_comparison.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n[compare] Results saved → {out_path}")


if __name__ == "__main__":
    sys.exit(main())
