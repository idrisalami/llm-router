"""Router CLI — Phase 3: LLM Router

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
"""

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="router_main.py",
        description="LLM Router — Phase 3 (LR vs Gradient Boosting)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--train", action="store_true",
                        help="Train classifiers + run cross-validated evaluation.")
    parser.add_argument("--route", type=str, default=None, metavar="PROMPT",
                        help="Route a single prompt to the best model.")
    parser.add_argument("--budget", type=float, default=None,
                        help="Max avg cost per query (USD). Only considers models within budget.")
    parser.add_argument("--recompute-embeddings", action="store_true",
                        help="Force re-computation of prompt embeddings.")
    parser.add_argument("--cv", type=int, default=5,
                        help="Number of cross-validation folds (default: 5).")
    parser.add_argument("--classifier", choices=["lr", "gb", "both"], default="both",
                        help="Classifier type: lr (logistic regression), gb (gradient boosting), "
                             "or both (default: both — runs and compares all).")

    args = parser.parse_args(argv)

    if not args.train and not args.route:
        parser.print_help()
        print("\n[error] Provide --train or --route.", file=sys.stderr)
        return 1

    # Resolve classifier list
    if args.classifier == "both":
        clf_types = ["lr", "gb"]
    else:
        clf_types = [args.classifier]

    # ── Shared: load dataset + embeddings ─────────────────────────────────────
    from sweep.load_data import load_coding_data
    from router.features import load_or_compute_embeddings

    print("[router] Loading dataset…")
    df = load_coding_data()

    print("[router] Loading / computing embeddings…")
    embeddings, prompts = load_or_compute_embeddings(
        df, force_recompute=args.recompute_embeddings
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

        # Cross-validate each classifier type and print per-model scores
        for clf_type in clf_types:
            cross_validate_classifiers(X, y_dict, cv=args.cv, clf_type=clf_type)

        # Train final classifiers on full dataset and save
        for clf_type in clf_types:
            clf_label = "Gradient Boosting" if clf_type == "gb" else "Logistic Regression"
            print(f"\n[router] Training final {clf_label} classifiers on full dataset…")
            classifiers = train_classifiers(X, y_dict, clf_type=clf_type)
            save_classifiers(classifiers, model_costs, clf_type=clf_type)

        # Full routing evaluation (CV) — compares all clf_types
        run_full_evaluation(df, embeddings, prompts, cv=args.cv,
                            budget=args.budget, clf_types=clf_types)

    # ── --route ───────────────────────────────────────────────────────────────
    if args.route:
        from router.train import load_classifiers
        from router.predict import route

        # Use the first clf_type in the list for routing
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

        # Sort by predicted quality descending
        for model, score in sorted(scores.items(), key=lambda x: -x[1]):
            within_budget = "✓" if args.budget is None or model_costs.get(model, 999) <= args.budget else "✗"
            selected = " ← selected" if model == best_model else ""
            print(f"    {within_budget}  {model:<42}  P(pass)={score:.3f}  cost=${model_costs.get(model, 0):.6f}{selected}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
