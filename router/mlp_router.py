"""Joint multi-output MLP router.

Takes X=(n_prompts, 384) and predicts Y=(n_prompts, 11),
where Y[i, j] = P(model_j passes on prompt_i).

At routing time the model with the highest predicted probability is selected.
score_tolerance widens the selection band so that models within N pp of the
best score are all eligible — the cheapest among them wins.

Architecture
------------
  Linear(384 → 128) → ReLU → Dropout(0.3)
  Linear(128 → 64)  → ReLU → Dropout(0.3)
  Linear(64 → 11)   → Sigmoid   (one probability per model)

Training
--------
  Loss    : BCELoss (binary cross-entropy, one output per model)
  Optim   : Adam, lr=1e-3, weight_decay=1e-4
  Stopping: early stopping on a 15%-held-out validation split (patience=15)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

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
    """Return X, per-model y labels, per-model avg costs, and ordered prompts."""
    prompt_to_idx = {p: i for i, p in enumerate(prompts)}

    pivot_q = df.pivot_table(index="input", columns="model", values="quality", aggfunc="mean")
    pivot_c = df.pivot_table(index="input", columns="model", values="cost",    aggfunc="mean")

    ordered_prompts = [p for p in prompts if p in pivot_q.index]
    X = np.array([embeddings[prompt_to_idx[p]] for p in ordered_prompts])

    y_dict:     dict[str, np.ndarray] = {}
    model_costs: dict[str, float]     = {}

    for model in MODELS:
        if model in pivot_q.columns:
            y_dict[model]      = pivot_q.loc[ordered_prompts, model].fillna(0).values.astype(float)
            model_costs[model] = float(pivot_c.loc[ordered_prompts, model].fillna(0).mean())

    return X, y_dict, model_costs, ordered_prompts


def compute_baselines(
    y_dict: dict[str, np.ndarray],
    model_costs: dict[str, float],
) -> dict[str, dict]:
    """Accuracy and avg_cost for always-best, always-cheapest, random, oracle."""
    n = len(next(iter(y_dict.values())))
    model_list = list(y_dict.keys())

    best_model     = max(y_dict, key=lambda m: y_dict[m].mean())
    cheapest_model = min(model_costs, key=lambda m: model_costs[m])

    random.seed(42)
    random_acc  = float(np.mean([y_dict[random.choice(model_list)][i] for i in range(n)]))
    random_cost = float(np.mean([model_costs[random.choice(model_list)] for _ in range(n)]))

    oracle_accs, oracle_costs = [], []
    for i in range(n):
        passing = [m for m in model_list if y_dict[m][i] == 1.0]
        if passing:
            cheapest_passing = min(passing, key=lambda m: model_costs[m])
            oracle_accs.append(1.0)
            oracle_costs.append(model_costs[cheapest_passing])
        else:
            oracle_accs.append(0.0)
            oracle_costs.append(0.0)

    return {
        "always-best":     {"accuracy": float(y_dict[best_model].mean()),     "avg_cost": model_costs[best_model]},
        "always-cheapest": {"accuracy": float(y_dict[cheapest_model].mean()), "avg_cost": model_costs[cheapest_model]},
        "random":          {"accuracy": random_acc,                            "avg_cost": random_cost},
        "oracle":          {"accuracy": float(np.mean(oracle_accs)),           "avg_cost": float(np.mean(oracle_costs))},
    }


# ──────────────────────────────────────────────────────────────────────────────
# MLP definition
# ──────────────────────────────────────────────────────────────────────────────

class RouterMLP(nn.Module):
    def __init__(self, input_dim: int, n_models: int = 11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_models),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_mlp(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    epochs: int = 150,
    lr: float = 1e-3,
    patience: int = 15,
    val_frac: float = 0.15,
    seed: int = 42,
) -> RouterMLP:
    """Train MLP with early stopping on a held-out validation split."""
    torch.manual_seed(seed)
    n = len(X_train)
    n_val = max(1, int(n * val_frac))
    idx = np.random.RandomState(seed).permutation(n)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    X_tr  = torch.tensor(X_train[tr_idx],  dtype=torch.float32)
    Y_tr  = torch.tensor(Y_train[tr_idx],  dtype=torch.float32)
    X_val = torch.tensor(X_train[val_idx], dtype=torch.float32)
    Y_val = torch.tensor(Y_train[val_idx], dtype=torch.float32)

    model   = RouterMLP(input_dim=X_train.shape[1], n_models=Y_train.shape[1])
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCELoss()

    best_val, best_state, wait = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss_fn(model(X_tr), Y_tr).backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), Y_val).item()

        if val_loss < best_val - 1e-5:
            best_val  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def save_mlp(model: RouterMLP, model_costs: dict[str, float], model_names: list[str]) -> None:
    """Persist model weights and metadata to data/router_models/."""
    import joblib
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "mlp_weights.pt")
    joblib.dump({"model_costs": model_costs, "model_names": model_names},
                MODEL_DIR / "mlp_meta.joblib")
    print(f"[mlp] Model saved → {MODEL_DIR}")


def load_mlp(input_dim: int = 384) -> tuple[RouterMLP, dict[str, float], list[str]]:
    """Load saved MLP and metadata."""
    import joblib
    weights_path = MODEL_DIR / "mlp_weights.pt"
    meta_path    = MODEL_DIR / "mlp_meta.joblib"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"No trained MLP found at {weights_path}.\n"
            "Run:  python3 router_main.py --train"
        )
    meta = joblib.load(meta_path)
    model = RouterMLP(input_dim=input_dim, n_models=len(meta["model_names"]))
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    return model, meta["model_costs"], meta["model_names"]


# ──────────────────────────────────────────────────────────────────────────────
# Routing policy
# ──────────────────────────────────────────────────────────────────────────────

def _scores_from_mlp(model: RouterMLP, x: np.ndarray, model_names: list[str]) -> dict[str, float]:
    with torch.no_grad():
        probs = model(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    return {m: float(p) for m, p in zip(model_names, probs)}


def select_model(
    scores: dict[str, float],
    model_costs: dict[str, float],
    budget: float | None = None,
    score_tolerance: float = 0.0,
) -> str:
    """Apply budget + tolerance policy to a scores dict, return model name."""
    eligible = {m: s for m, s in scores.items()
                if budget is None or model_costs.get(m, float("inf")) <= budget}
    if not eligible:
        eligible = scores
    best_score = max(eligible.values())
    near_best  = {m: s for m, s in eligible.items() if best_score - s <= score_tolerance}
    return min(near_best, key=lambda m: model_costs.get(m, float("inf")))


# ──────────────────────────────────────────────────────────────────────────────
# Cross-validated evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_mlp_router_cv(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    prompts: list[str],
    cv: int = 5,
    budget: float | None = None,
    score_tolerance: float = 0.0,
) -> pd.DataFrame:
    """Run k-fold CV for the joint MLP router.

    Returns DataFrame: prompt, fold, routed_model, actual_quality, actual_cost
    """
    X, y_dict, model_costs, ordered_prompts = build_training_matrix(df, embeddings, prompts)
    model_names = list(y_dict.keys())
    Y = np.stack([y_dict[m] for m in model_names], axis=1).astype(np.float32)

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"  fold {fold_idx + 1}/{cv}…", end="\r")
        trained = train_mlp(X[train_idx], Y[train_idx])

        for j in test_idx:
            scores = _scores_from_mlp(trained, X[j], model_names)
            routed = select_model(scores, model_costs, budget, score_tolerance)
            rows.append({
                "prompt":         ordered_prompts[j],
                "fold":           fold_idx,
                "routed_model":   routed,
                "actual_quality": float(y_dict[routed][j]),
                "actual_cost":    model_costs[routed],
            })

    print()
    return pd.DataFrame(rows)


def evaluate_prior_router_cv(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    prompts: list[str],
    cv: int = 5,
    budget: float | None = None,
    score_tolerance: float = 0.0,
) -> pd.DataFrame:
    """No-embedding baseline: scores = per-model average pass rate on training fold.

    The same prompt-independent score is applied to every test prompt.
    This isolates whether the MLP's prompt embeddings add value beyond
    simply knowing each model's global pass rate.

    Difference from 'random': random picks a uniformly random model per prompt
    (ignores model quality entirely). This baseline knows model quality but
    ignores the specific prompt.
    """
    X, y_dict, model_costs, ordered_prompts = build_training_matrix(df, embeddings, prompts)
    model_names = list(y_dict.keys())

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Scores = average pass rate on the training fold (prompt-independent)
        train_pass_rates = {m: float(y_dict[m][train_idx].mean()) for m in model_names}

        for j in test_idx:
            routed = select_model(train_pass_rates, model_costs, budget, score_tolerance)
            rows.append({
                "prompt":         ordered_prompts[j],
                "fold":           fold_idx,
                "routed_model":   routed,
                "actual_quality": float(y_dict[routed][j]),
                "actual_cost":    model_costs[routed],
            })

    return pd.DataFrame(rows)
