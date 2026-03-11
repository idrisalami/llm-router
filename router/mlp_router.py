"""Joint multi-output router: RF and MLP.

Both models take X=(n_prompts, emb_dim) and predict Y=(n_prompts, 11),
where Y[i, j] = P(model_j passes on prompt_i).

At routing time the model with the highest predicted probability is selected
(same policy as the binary LR baseline, so results are directly comparable).

Models
------
  rf  : RandomForestRegressor (multi-output, shared trees)
  mlp : Small PyTorch MLP (shared hidden layers, BCELoss)

Architecture (MLP)
------------------
  Linear(emb_dim → 128) → ReLU → Dropout(0.3)
  Linear(128 → 64)      → ReLU → Dropout(0.3)
  Linear(64 → 11)       → Sigmoid   (one probability per model)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.model_selection import KFold

from .train import MODELS, build_training_matrix
from .predict import route_embedding as _route_embedding_dict


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
# Training helpers
# ──────────────────────────────────────────────────────────────────────────────

def _train_mlp(
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

    X_tr = torch.tensor(X_train[tr_idx], dtype=torch.float32)
    Y_tr = torch.tensor(Y_train[tr_idx], dtype=torch.float32)
    X_val = torch.tensor(X_train[val_idx], dtype=torch.float32)
    Y_val = torch.tensor(Y_train[val_idx], dtype=torch.float32)

    model = RouterMLP(input_dim=X_train.shape[1], n_models=Y_train.shape[1])
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCELoss()

    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss_fn(model(X_tr), Y_tr).backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), Y_val).item()

        if val_loss < best_val - 1e-5:
            best_val, best_state, wait = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def _train_rf(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    n_estimators: int = 200,
    seed: int = 42,
) -> RandomForestRegressor:
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=8,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(X_train, Y_train)
    return rf


# ──────────────────────────────────────────────────────────────────────────────
# Unified predict interface → dict[model_name → P(pass)]
# ──────────────────────────────────────────────────────────────────────────────

def _predict_scores(
    model,
    x: np.ndarray,
    model_names: list[str],
    model_type: str,
) -> dict[str, float]:
    """Return P(pass) dict for one prompt embedding."""
    if model_type == "mlp":
        with torch.no_grad():
            probs = model(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
            probs = probs.squeeze(0).numpy()
    else:  # rf
        probs = model.predict(x.reshape(1, -1))[0]
        probs = np.clip(probs, 0.0, 1.0)

    return {m: float(p) for m, p in zip(model_names, probs)}


# ──────────────────────────────────────────────────────────────────────────────
# Cross-validated evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_joint_router_cv(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    prompts: list[str],
    model_type: str = "mlp",   # "mlp" or "rf"
    cv: int = 5,
    budget: float | None = None,
    score_tolerance: float = 0.0,
) -> pd.DataFrame:
    """Run k-fold CV for the joint (multi-output) router.

    Returns DataFrame with columns:
      prompt, fold, routed_model, actual_quality, actual_cost
    """
    X, y_dict, model_costs, ordered_prompts = build_training_matrix(df, embeddings, prompts)
    model_names = list(y_dict.keys())

    # Y matrix: (n_prompts, n_models)
    Y = np.stack([y_dict[m] for m in model_names], axis=1).astype(np.float32)

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"  fold {fold_idx + 1}/{cv}…", end="\r")

        if model_type == "mlp":
            trained = _train_mlp(X[train_idx], Y[train_idx])
        else:
            trained = _train_rf(X[train_idx], Y[train_idx])

        for j in test_idx:
            scores = _predict_scores(trained, X[j], model_names, model_type)

            # Reuse existing routing policy (budget + tolerance)
            eligible = {
                m: s for m, s in scores.items()
                if budget is None or model_costs.get(m, float("inf")) <= budget
            }
            if not eligible:
                eligible = scores

            best_score = max(eligible.values())
            near_best  = {m: s for m, s in eligible.items()
                          if best_score - s <= score_tolerance}
            routed = min(near_best, key=lambda m: model_costs.get(m, float("inf")))

            rows.append({
                "prompt":         ordered_prompts[j],
                "fold":           fold_idx,
                "routed_model":   routed,
                "actual_quality": float(y_dict[routed][j]),
                "actual_cost":    model_costs[routed],
                "chosen_score":   scores[routed],
            })

    print()
    return pd.DataFrame(rows)
