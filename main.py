"""Capstone — Phase 1 & 2: Sweep Engine + Analysis

Usage
-----
    # Analyse a specific prompt
    python main.py --prompt "Write a function that returns the nth Fibonacci number."

    # List the first N available prompts in the dataset
    python main.py --list-prompts --top 20

    # Force re-download of the RouterBench dataset
    python main.py --prompt "..." --force-reload

    # Save outputs to a custom directory
    python main.py --prompt "..." --output-dir results/my_run
"""

import argparse
import sys
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sanitize_dirname(text: str, maxlen: int = 60) -> str:
    """Turn a prompt string into a safe directory name."""
    import re
    s = re.sub(r"[^\w\s-]", "", text)
    s = re.sub(r"\s+", "_", s.strip())
    return s[:maxlen] or "run"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="RouterBench Sweep Engine — Phase 1 & 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="Keyword(s) to search for a prompt (fuzzy matched against the dataset).",
    )
    parser.add_argument(
        "--id", "-i",
        type=int,
        default=None,
        metavar="N",
        help="Run analysis on prompt #N from --list-prompts (1-based index).",
    )
    parser.add_argument(
        "--list-prompts", "-l",
        action="store_true",
        help="Print all available prompts from the dataset and exit.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Limit --list-prompts to the first N entries.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory to save plots and CSV (default: output/<sanitized_prompt>/).",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Re-download and re-cache the RouterBench dataset.",
    )
    parser.add_argument(
        "--no-fuzzy",
        action="store_true",
        help="Disable fuzzy prompt matching (exact match only).",
    )

    args = parser.parse_args(argv)

    # ── Lazy imports (keep startup fast) ──────────────────────────────────────
    from sweep.load_data import load_coding_data
    from sweep.query import get_model_results, list_prompts

    # ── Load (or re-download) dataset ─────────────────────────────────────────
    try:
        df_full = load_coding_data(force_reload=args.force_reload)
    except Exception as exc:
        print(f"[error] Failed to load dataset: {exc}", file=sys.stderr)
        return 1

    # ── --list-prompts mode ───────────────────────────────────────────────────
    if args.list_prompts:
        prompts = list_prompts(df_full, n=args.top)
        print(f"\nAvailable prompts ({len(prompts)} total):\n")
        for i, p in enumerate(prompts, 1):
            print(f"  {i:>3}. {p[:120]}")
        print()
        print("Tip: run with  --id N  to analyse prompt #N directly.")
        print()
        return 0

    # ── Resolve prompt: --id takes precedence over --prompt ──────────────────
    if args.id is not None:
        all_prompts = list_prompts(df_full)
        if args.id < 1 or args.id > len(all_prompts):
            print(f"[error] --id {args.id} is out of range (1–{len(all_prompts)}).", file=sys.stderr)
            return 1
        exact_prompt = all_prompts[args.id - 1]
        print(f"\n[main] Selected prompt #{args.id}: {exact_prompt[:100]}")
        query_str = exact_prompt
        force_exact = True
    elif args.prompt:
        query_str = args.prompt
        force_exact = False
    else:
        parser.print_help()
        print("\n[error] Provide --prompt, --id, or --list-prompts.", file=sys.stderr)
        return 1

    # ── Query ─────────────────────────────────────────────────────────────────
    try:
        results, matched_prompt, match_type = get_model_results(
            query_str,
            df=df_full,
            fuzzy=(not args.no_fuzzy) and (not force_exact),
        )
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    # ── Warn clearly when the matched prompt differs from what was typed ───────
    if match_type != "exact":
        print()
        print("  ╔══ WARNING: fuzzy match ══════════════════════════════════════════╗")
        print(f"  ║  Your input  : {query_str[:60]!r}")
        print(f"  ║  Matched to  : {matched_prompt[:60]!r}")
        print(f"  ║  Match type  : {match_type}")
        print("  ║  Results below are for the MATCHED prompt, not your input.")
        print("  ║  Use --list-prompts to browse exact prompts, then --id N.")
        print("  ╚══════════════════════════════════════════════════════════════════╝")
        print()

    if results.empty:
        print("[error] No results returned for this prompt.", file=sys.stderr)
        return 1

    print(f"[main] {len(results)} models found: {sorted(results['model'].tolist())}\n")

    # ── Validate required columns ─────────────────────────────────────────────
    missing = [c for c in ("model", "accuracy", "cost") if c not in results.columns]
    if missing:
        print(f"[error] Missing required columns after aggregation: {missing}", file=sys.stderr)
        print(f"        Available columns: {list(results.columns)}", file=sys.stderr)
        return 1

    # ── Output directory ──────────────────────────────────────────────────────
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("output") / _sanitize_dirname(matched_prompt)
    print(f"[main] Output directory: {out_dir}\n")

    # ── Visualisations ────────────────────────────────────────────────────────
    from analysis.visualize import save_all_plots
    save_all_plots(results, out_dir, prompt_snippet=matched_prompt)

    # ── Statistics ────────────────────────────────────────────────────────────
    from analysis.stats import print_and_save_stats
    stats = print_and_save_stats(results, out_dir, prompt_snippet=matched_prompt)

    # ── Final summary ─────────────────────────────────────────────────────────
    pareto_models = stats.loc[stats["pareto_optimal"], "model"].tolist()
    best = stats.iloc[0]
    cheapest_pareto = stats[stats["pareto_optimal"]].sort_values("cost_usd").iloc[0]

    print("── Key Findings ─────────────────────────────────────────────────")
    print(f"  Highest accuracy : {best['model']}  ({best['accuracy (pass@1)']:.4f})")
    print(f"  Pareto-optimal   : {', '.join(pareto_models)}")
    print(
        f"  Best value       : {cheapest_pareto['model']}"
        f"  (acc={cheapest_pareto['accuracy (pass@1)']:.4f},"
        f" cost=${cheapest_pareto['cost_usd']:.5f})"
    )
    print()
    print(f"  Plots and CSV saved to: {out_dir.resolve()}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
