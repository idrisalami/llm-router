"""Router prediction policy.

Given a prompt (or a pre-computed embedding) and an optional cost budget,
returns the model predicted to have the highest pass probability within budget.

Tie-breaking
------------
When multiple models have P(pass) scores within SCORE_TOLERANCE of the best
score, the cheapest model among that near-tied group is selected.  This avoids
routing to an expensive model when a much cheaper alternative is almost as
confident.  Default tolerance = 0.05 (5 percentage points).

Confidence signal
-----------------
The returned scores dict maps every model to its predicted P(pass).  A low
score for the chosen model (e.g. below 0.5) means the router is not confident
any model will succeed on this prompt — this can be surfaced to users or AI
agents as a reliability warning.
"""

from __future__ import annotations

import numpy as np

# Models whose P(pass) is within this margin of the best score are considered
# "tied" and the cheapest among them is selected.
SCORE_TOLERANCE = 0.05


def route(
    prompt: str,
    classifiers: dict,
    model_costs: dict[str, float],
    budget: float | None = None,
    score_tolerance: float = SCORE_TOLERANCE,
) -> tuple[str, dict[str, float]]:
    """Route a raw prompt string to the best model.

    Returns
    -------
    best_model : name of the selected model
    scores     : dict of model → predicted P(pass)
    """
    from .features import embed_single
    embedding = embed_single(prompt)
    return route_embedding(embedding, classifiers, model_costs, budget, score_tolerance)


def route_embedding(
    embedding: np.ndarray,
    classifiers: dict,
    model_costs: dict[str, float],
    budget: float | None = None,
    score_tolerance: float = SCORE_TOLERANCE,
) -> tuple[str, dict[str, float]]:
    """Route from a pre-computed embedding (used during batch evaluation).

    Budget is compared against each model's *average* cost across the training
    set — a simple proxy for expected cost on a new prompt.

    Among models whose P(pass) is within score_tolerance of the best score,
    the cheapest model is selected.
    """
    scores: dict[str, float] = {}
    for model, clf in classifiers.items():
        prob = clf.predict_proba(embedding.reshape(1, -1))[0]
        scores[model] = float(prob[1])   # P(pass = 1)

    # Apply budget filter
    eligible = {
        m: s for m, s in scores.items()
        if budget is None or model_costs.get(m, float("inf")) <= budget
    }

    if not eligible:
        # No model within budget — fall back to cheapest available
        cheapest = min(model_costs, key=lambda m: model_costs[m])
        return cheapest, scores

    best_score = max(eligible.values())

    # Among models within score_tolerance of the best, pick the cheapest
    near_best = {
        m: s for m, s in eligible.items()
        if best_score - s <= score_tolerance
    }
    best_model = min(near_best, key=lambda m: model_costs.get(m, float("inf")))
    return best_model, scores
