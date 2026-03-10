"""Load and cache the RouterBench coding subset.

RouterBench (Hu et al., 2024): https://arxiv.org/abs/2403.12031
HuggingFace dataset: withmartian/routerbench  (pickle files, not Datasets format)

Actual schema (routerbench_0shot.pkl):
    Wide format — one row per prompt, columns:
        sample_id, prompt (stringified list), eval_name,
        <model>          (quality: 0.0 / 1.0 per model),
        <model>|total_cost,
        <model>|model_response,
        oracle_model_to_route_to

This module downloads the pickle, melts it to long format (one row per
prompt×model), and caches the coding subset as a parquet file.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

CACHE_PATH = Path(__file__).parent.parent / "data" / "routerbench_coding.parquet"
HF_REPO    = "withmartian/routerbench"
HF_FILE    = "routerbench_0shot.pkl"

# Benchmark names in the dataset that count as "coding"
CODING_DATASETS = {"mbpp", "humaneval", "human_eval", "code"}

# All models present in the dataset
MODELS = [
    "WizardLM/WizardLM-13B-V1.2",
    "claude-instant-v1",
    "claude-v1",
    "claude-v2",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
    "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat",
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
]

# Canonical column aliases (kept for query.py compatibility)
COL_ALIASES: dict[str, list[str]] = {
    "input":   ["input", "prompt", "prompt_text"],
    "model":   ["model"],
    "quality": ["quality"],
    "cost":    ["cost"],
    "dataset": ["dataset", "eval_name"],
}


def find_col(df: pd.DataFrame, canonical: str) -> str | None:
    for alias in COL_ALIASES.get(canonical, [canonical]):
        if alias in df.columns:
            return alias
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """No-op here — columns are already canonical after _wide_to_long."""
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_coding_data(force_reload: bool = False) -> pd.DataFrame:
    """Return a long-format DataFrame of RouterBench coding rows.

    Columns: sample_id, input, dataset, model, quality, cost
    """
    if CACHE_PATH.exists() and not force_reload:
        print(f"[load] Reading cache: {CACHE_PATH}")
        return pd.read_parquet(CACHE_PATH)

    pkl_path = _download_pickle()
    print("[load] Parsing pickle…")
    raw: pd.DataFrame = pd.read_pickle(pkl_path)
    print(f"[load] Wide dataset: {raw.shape[0]:,} rows, {raw.shape[1]} cols")
    print(f"[load] Benchmarks present: {sorted(raw['eval_name'].unique())[:10]} …")

    long_df = _wide_to_long(raw)

    # Filter to coding benchmarks
    mask = long_df["dataset"].str.lower().isin(CODING_DATASETS)
    coding_df = long_df[mask].copy()
    if coding_df.empty:
        print("[load] WARNING: coding filter matched 0 rows — keeping all rows.")
        coding_df = long_df.copy()

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    coding_df.to_parquet(CACHE_PATH, index=False)
    print(f"[load] Cached {len(coding_df):,} coding rows → {CACHE_PATH}")
    print(f"[load] Models: {sorted(coding_df['model'].unique())}")
    print(f"[load] Benchmarks: {sorted(coding_df['dataset'].unique())}")
    return coding_df


# ──────────────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────────────

def _download_pickle() -> Path:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Install huggingface_hub:  pip install huggingface_hub"
        ) from exc

    print(f"[load] Downloading {HF_FILE} from {HF_REPO} (one-time ~100 MB)…")
    local = hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=HF_FILE)
    print(f"[load] Downloaded → {local}")
    return Path(local)


def _parse_prompt(raw: object) -> str:
    """Convert a stringified-list prompt to a plain string."""
    if isinstance(raw, list):
        return "\n".join(str(x) for x in raw)
    if isinstance(raw, str):
        try:
            parts = ast.literal_eval(raw)
            if isinstance(parts, list):
                return "\n".join(str(x) for x in parts)
        except (ValueError, SyntaxError):
            pass
        return raw
    return str(raw)


def _wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Melt RouterBench wide format → long format (one row per prompt×model)."""
    # Identify which model columns are actually present
    present_models = [m for m in MODELS if m in df.columns]
    missing = set(MODELS) - set(present_models)
    if missing:
        print(f"[load] Note: {len(missing)} model columns not found: {missing}")

    # Parse prompts (stringified Python lists → plain text)
    print("[load] Parsing prompt strings…")
    df = df.copy()
    df["input"] = df["prompt"].apply(_parse_prompt)

    # ── Quality (wide → long) ────────────────────────────────────────────────
    q_long = (
        df[["sample_id", "input", "eval_name"] + present_models]
        .melt(
            id_vars=["sample_id", "input", "eval_name"],
            value_vars=present_models,
            var_name="model",
            value_name="quality",
        )
    )

    # ── Cost (wide → long) ───────────────────────────────────────────────────
    cost_cols = [f"{m}|total_cost" for m in present_models if f"{m}|total_cost" in df.columns]
    c_long = (
        df[["sample_id"] + cost_cols]
        .melt(
            id_vars=["sample_id"],
            value_vars=cost_cols,
            var_name="_cost_col",
            value_name="cost",
        )
    )
    c_long["model"] = c_long["_cost_col"].str.replace("|total_cost", "", regex=False)
    c_long = c_long.drop(columns=["_cost_col"])

    # ── Merge ────────────────────────────────────────────────────────────────
    long_df = q_long.merge(c_long, on=["sample_id", "model"], how="left")
    long_df = long_df.rename(columns={"eval_name": "dataset"})
    long_df = long_df[["sample_id", "input", "dataset", "model", "quality", "cost"]]
    long_df["quality"] = pd.to_numeric(long_df["quality"], errors="coerce").fillna(0.0)
    long_df["cost"]    = pd.to_numeric(long_df["cost"],    errors="coerce").fillna(0.0)

    print(f"[load] Long format: {len(long_df):,} rows ({len(present_models)} models × {len(df):,} prompts)")
    return long_df
