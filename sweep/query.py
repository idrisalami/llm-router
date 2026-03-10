"""Given a natural-language prompt, return per-model pre-computed results from RouterBench.

Each row in the returned DataFrame represents one model's result for the queried prompt.
Columns (canonical names after normalization):
    model         — model identifier string
    quality       — 0 or 1 (pass/fail on coding task)
    cost          — cost in USD
    latency       — response time in seconds (if available)
    input_tokens  — (if available)
    output_tokens — (if available)
"""

import re
from typing import Optional

import pandas as pd

from .load_data import find_col, load_coding_data, normalize_columns


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def get_model_results(
    prompt: str,
    df: Optional[pd.DataFrame] = None,
    fuzzy: bool = True,
    max_fuzzy_chars: int = 80,
) -> tuple[pd.DataFrame, str]:
    """Return (results_df, matched_prompt) for *prompt*.

    matched_prompt is the actual dataset prompt that was used — may differ from
    the user's input when fuzzy matching is applied.

    Matching strategy (in order):
        1. Exact string match on the 'input' column.
        2. Case-insensitive substring match (first max_fuzzy_chars chars).
        3. Token-overlap match — only accepted when score >= 0.8 AND the query
           is at least 4 tokens long, to avoid spurious matches on short inputs.

    Raises ValueError if no match is found.
    """
    if df is None:
        df = load_coding_data()

    input_col = find_col(df, "input")
    if input_col is None:
        raise ValueError("Dataset has no recognisable prompt/input column.")

    # ── 1. Exact match ────────────────────────────────────────────────────────
    mask = df[input_col] == prompt
    match_type = "exact"

    # ── 2. Case-insensitive substring ─────────────────────────────────────────
    if mask.sum() == 0 and fuzzy:
        snippet = prompt[:max_fuzzy_chars].lower()
        mask = df[input_col].str.lower().str.contains(re.escape(snippet), regex=True, na=False)
        match_type = "substring"

    # ── 3. Token-overlap — only for longer queries to avoid garbage matches ───
    if mask.sum() == 0 and fuzzy:
        prompt_tokens = set(prompt.lower().split())
        if len(prompt_tokens) >= 4:
            scores = df[input_col].fillna("").apply(
                lambda x: len(prompt_tokens & set(x.lower().split())) / max(len(prompt_tokens), 1)
            )
            best_score = scores.max()
            if best_score >= 0.8:
                mask = scores >= best_score * 0.95
                match_type = f"token-overlap (score={best_score:.2f})"
            else:
                mask = pd.Series([False] * len(df), index=df.index)
        else:
            mask = pd.Series([False] * len(df), index=df.index)

    if mask.sum() == 0:
        _raise_no_match(df, input_col, prompt)

    matched_inputs = df.loc[mask, input_col].unique()
    if len(matched_inputs) > 1:
        prompt_tokens = set(prompt.lower().split())
        best_input = max(
            matched_inputs,
            key=lambda x: len(prompt_tokens & set(x.lower().split())),
        )
        mask = df[input_col] == best_input

    selected_prompt = df.loc[mask, input_col].iloc[0]

    results = df[mask].copy()
    results = normalize_columns(results)
    return aggregate_by_model(results), selected_prompt, match_type


def aggregate_by_model(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multiple rows per model (e.g. from pass@k runs) into one row each."""
    model_col = find_col(results, "model") or "model"
    if model_col not in results.columns:
        return results

    numeric_cols = [
        c for c in ["quality", "cost", "latency", "input_tokens", "output_tokens"]
        if c in results.columns
    ]
    if not numeric_cols:
        return results

    agg: pd.DataFrame = (
        results.groupby(model_col)[numeric_cols]
        .mean()
        .reset_index()
    )
    # Rename 'quality' → 'accuracy' for downstream clarity
    if "quality" in agg.columns:
        agg = agg.rename(columns={"quality": "accuracy"})
    return agg


def list_prompts(df: Optional[pd.DataFrame] = None, n: Optional[int] = None) -> list[str]:
    """Return distinct prompt strings from the dataset (all by default)."""
    if df is None:
        df = load_coding_data()
    input_col = find_col(df, "input")
    if input_col is None:
        return []
    prompts = df[input_col].dropna().unique().tolist()
    return prompts if n is None else prompts[:n]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _raise_no_match(df: pd.DataFrame, input_col: str, prompt: str) -> None:
    samples = df[input_col].dropna().unique()[:5]
    sample_text = "\n".join(f"  • {p[:100]}" for p in samples)
    raise ValueError(
        f"No results found for the given prompt.\n\n"
        f"Your prompt (first 100 chars):\n  {prompt[:100]}\n\n"
        f"Sample prompts available in the dataset:\n{sample_text}\n\n"
        f"Tip: run  python main.py --list-prompts  to see all available prompts."
    )
