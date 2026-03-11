"""Multi-class oracle classifier.

Instead of 11 binary classifiers, trains ONE classifier whose target is the
oracle model — the cheapest model that actually passes for each prompt.

At inference time the predicted class is the model to route to directly.
This is better aligned with the actual routing objective and shares
information across all models in a single model.

Training target
---------------
y[i] = cheapest model that passes for prompt i.
If no model passes, y[i] = cheapest model overall (will fail, but we still
need a label; the router just happens to lose that prompt).

Label distribution
------------------
Reflects the oracle distribution — cheaper models appear more often as labels
when they are competitive (i.e. when they actually pass alongside expensive
ones). This makes the class frequencies informative about model complementarity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from .train import (
    MODELS,
    build_training_matrix,
    _make_pipeline,
)


# ──────────────────────────────────────────────────────────────────────────────
# Label construction
# ──────────────────────────────────────────────────────────────────────────────

def build_oracle_labels(
    y_dict: dict[str, np.ndarray],
    model_costs: dict[str, float],
) -> list[str]:
    """Return per-prompt oracle model label (cheapest passing model)."""
    n = len(next(iter(y_dict.values())))
    model_list = list(y_dict.keys())
    cheapest_overall = min(model_list, key=lambda m: model_costs.get(m, float("inf")))

    labels = []
    for i in range(n):
        passing = [m for m in model_list if y_dict[m][i] == 1.0]
        if passing:
            labels.append(min(passing, key=lambda m: model_costs.get(m, float("inf"))))
        else:
            labels.append(cheapest_overall)   # nothing passes; cheapest is least bad
    return labels


# ──────────────────────────────────────────────────────────────────────────────
# Cross-validated evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_oracle_clf_cv(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    prompts: list[str],
    cv: int = 5,
    clf_type: str = "lr",
) -> pd.DataFrame:
    """Run k-fold CV for the oracle multi-class classifier.

    Returns a DataFrame with columns:
      prompt, fold, routed_model, actual_quality, actual_cost
    """
    X, y_dict, model_costs, ordered_prompts = build_training_matrix(df, embeddings, prompts)

    oracle_labels = build_oracle_labels(y_dict, model_costs)
    le = LabelEncoder()
    y_oracle = le.fit_transform(oracle_labels)          # integer class ids

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        pipe = _make_pipeline(clf_type)
        pipe.fit(X[train_idx], y_oracle[train_idx])

        preds = pipe.predict(X[test_idx])
        pred_models = le.inverse_transform(preds)

        for j, model in zip(test_idx, pred_models):
            # If predicted model not in y_dict (shouldn't happen) fall back
            if model not in y_dict:
                model = min(model_costs, key=lambda m: model_costs[m])
            rows.append({
                "prompt":         ordered_prompts[j],
                "fold":           fold_idx,
                "routed_model":   model,
                "actual_quality": float(y_dict[model][j]),
                "actual_cost":    model_costs[model],
            })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Per-class accuracy (how often does the clf predict each oracle class?)
# ──────────────────────────────────────────────────────────────────────────────

def print_oracle_label_distribution(
    y_dict: dict[str, np.ndarray],
    model_costs: dict[str, float],
) -> None:
    labels = build_oracle_labels(y_dict, model_costs)
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    print("\n  Oracle label distribution (target class frequencies):")
    for model, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {model:<42}  {count:>4}  ({count/total*100:.1f}%)")
    print()
