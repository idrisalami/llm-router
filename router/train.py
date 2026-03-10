"""Train one classifier per model.

Each classifier predicts P(pass | prompt_embedding) for its model.
Supports logistic regression (lr) and gradient boosting (gb).
Classifiers are saved to data/router_models/ and reloaded at inference time.
"""

from __future__ import annotations

from pathlib import Path

import joblib  # type: ignore
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.model_selection import StratifiedKFold, cross_val_score  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

MODEL_DIR = Path(__file__).parent.parent / "data" / "router_models"

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


# ──────────────────────────────────────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────────────────────────────────────

def build_training_matrix(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    prompts: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], list[str]]:
    """Build X, per-model y labels, and per-model average costs.

    Returns
    -------
    X              : (n_prompts, 384) embedding matrix
    y_dict         : model → binary label array (n_prompts,), float 0/1
    model_costs    : model → average cost across all prompts (used for budget filtering)
    ordered_prompts: list of prompts in the same order as X rows
    """
    prompt_to_idx = {p: i for i, p in enumerate(prompts)}

    # One row per prompt
    pivot_q = df.pivot_table(index="input", columns="model", values="quality", aggfunc="mean")
    pivot_c = df.pivot_table(index="input", columns="model", values="cost",    aggfunc="mean")

    # Keep only prompts that appear in both embeddings and the pivot
    ordered_prompts = [p for p in prompts if p in pivot_q.index]
    X = np.array([embeddings[prompt_to_idx[p]] for p in ordered_prompts])

    y_dict:     dict[str, np.ndarray] = {}
    model_costs: dict[str, float]     = {}

    for model in MODELS:
        if model in pivot_q.columns:
            y_dict[model]      = pivot_q.loc[ordered_prompts, model].fillna(0).values.astype(float)
            model_costs[model] = float(pivot_c.loc[ordered_prompts, model].fillna(0).mean())

    return X, y_dict, model_costs, ordered_prompts


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def _make_pipeline(clf_type: str = "lr") -> Pipeline:
    """Return a sklearn Pipeline for the given classifier type.

    clf_type: "lr"  → StandardScaler + LogisticRegression
              "gb"  → HistGradientBoostingClassifier (no scaler needed)
    """
    if clf_type == "gb":
        return Pipeline([
            ("clf", HistGradientBoostingClassifier(
                max_iter=200, max_depth=3, learning_rate=0.1,
                min_samples_leaf=10, random_state=42,
            )),
        ])
    # default: logistic regression
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])


def train_classifiers(
    X: np.ndarray,
    y_dict: dict[str, np.ndarray],
    clf_type: str = "lr",
) -> dict[str, Pipeline]:
    """Fit one pipeline per model on the full training set."""
    classifiers: dict[str, Pipeline] = {}
    for model, y in y_dict.items():
        pipe = _make_pipeline(clf_type)
        pipe.fit(X, y.astype(int))
        classifiers[model] = pipe
    return classifiers


def cross_validate_classifiers(
    X: np.ndarray,
    y_dict: dict[str, np.ndarray],
    cv: int = 5,
    clf_type: str = "lr",
) -> dict[str, np.ndarray]:
    """Return per-model CV accuracy scores (n_folds,)."""
    cv_scores: dict[str, np.ndarray] = {}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    clf_label = "Gradient Boosting" if clf_type == "gb" else "Logistic Regression"
    print(f"\n[train] Per-model {cv}-fold CV accuracy — {clf_label} (vs majority-class baseline):")
    header = f"  {'Model':<42} {'CV acc':>8}  {'±':>6}  {'Baseline':>9}  {'Lift':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for model, y in y_dict.items():
        scores = cross_val_score(_make_pipeline(clf_type), X, y.astype(int), cv=skf, scoring="accuracy")
        cv_scores[model] = scores
        baseline = max(y.mean(), 1 - y.mean())
        lift = scores.mean() - baseline
        lift_str = f"+{lift:.3f}" if lift >= 0 else f"{lift:.3f}"
        print(
            f"  {model:<42} {scores.mean():>8.3f}  "
            f"{scores.std():>6.3f}  {baseline:>9.3f}  {lift_str:>6}"
        )

    return cv_scores


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_classifiers(
    classifiers: dict[str, Pipeline],
    model_costs: dict[str, float],
    path: Path = MODEL_DIR,
    clf_type: str = "lr",
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fname = f"classifiers_{clf_type}.joblib"
    joblib.dump(classifiers,  path / fname)
    joblib.dump(model_costs,  path / "model_costs.joblib")
    print(f"\n[train] Classifiers saved → {path / fname}")


def load_classifiers(
    path: Path = MODEL_DIR,
    clf_type: str = "lr",
) -> tuple[dict[str, Pipeline], dict[str, float]]:
    fname = f"classifiers_{clf_type}.joblib"
    clf_path  = path / fname
    cost_path = path / "model_costs.joblib"
    if not clf_path.exists():
        raise FileNotFoundError(
            f"No trained router found at {clf_path}.\n"
            f"Run:  python3 router_main.py --train --classifier {clf_type}"
        )
    return joblib.load(clf_path), joblib.load(cost_path)
