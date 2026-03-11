"""Joint multi-output MLP router.

Takes X=(n_prompts, emb_dim) and predicts Y=(n_prompts, 11),
where Y[i, j] = P(model_j passes on prompt_i).

At routing time the model with the highest predicted probability is selected,
with optional score_tolerance to prefer cheaper models when scores are close.

Architecture
------------
  Linear(emb_dim → 128) → ReLU → Dropout(0.3)
  Linear(128 → 64)      → ReLU → Dropout(0.3)
  Linear(64 → 11)       → Sigmoid   (one probability per model)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

from .train import build_training_matrix


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

    Returns DataFrame with columns:
      prompt, fold, routed_model, actual_quality, actual_cost, chosen_score
    """
    X, y_dict, model_costs, ordered_prompts = build_training_matrix(df, embeddings, prompts)
    model_names = list(y_dict.keys())

    # Y matrix: (n_prompts, n_models)
    Y = np.stack([y_dict[m] for m in model_names], axis=1).astype(np.float32)

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"  fold {fold_idx + 1}/{cv}…", end="\r")
        trained = _train_mlp(X[train_idx], Y[train_idx])

        for j in test_idx:
            with torch.no_grad():
                probs = trained(torch.tensor(X[j], dtype=torch.float32).unsqueeze(0))
                probs = probs.squeeze(0).numpy()
            scores = {m: float(p) for m, p in zip(model_names, probs)}

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
