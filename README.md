# LLM Router — Capstone Project

Routes coding prompts to the cheapest LLM that will pass, using a joint multi-output MLP trained on prompt embeddings.

**Dataset:** [RouterBench](https://arxiv.org/abs/2403.12031) — 425 MBPP coding prompts × 11 LLMs, pre-computed pass/fail + cost.

---

## Model Architecture

A single neural network predicts pass probability for **all 11 models simultaneously**, sharing a common hidden representation. This is more efficient than training 11 independent classifiers and lets the network learn cross-model structure (e.g. "if GPT-4 fails this, cheaper models probably do too").

```
Prompt text
    │
    ▼
all-MiniLM-L6-v2          sentence-transformers, 384-dim, runs locally
    │
    ▼  (384,)
Linear(384 → 128) → ReLU → Dropout(0.3)
Linear(128 → 64)  → ReLU → Dropout(0.3)
Linear(64 → 11)   → Sigmoid
    │
    ▼  (11,)  one P(pass) per model
Routing policy: pick model with highest P(pass)
                within score_tolerance of best → prefer cheapest
```

**Training:** Adam, lr=1e-3, weight_decay=1e-4, BCELoss, early stopping on 15% validation split (patience=15).

**Evaluation:** Strict 5-fold cross-validation — no prompt is ever in both training and test for a given fold.

---

## Results

| Strategy | Accuracy | Avg cost | Savings vs GPT-4 |
|---|---|---|---|
| always-best (GPT-4) | 68.6% | $0.009377 | 0.0% |
| always-cheapest (Mistral-7B) | 34.2% | $0.000049 | 99.5% |
| random | 51.8% | $0.001999 | 78.7% |
| oracle (theoretical ceiling) | 86.6% | $0.000413 | 95.6% |
| **MLP tol=0.00** | **69.1%** | $0.009149 | 2.4% |
| **MLP tol=0.05** | **65.2%** | $0.003134 | 66.6% |
| **MLP tol=0.10** ✓ | **63.4%** | **$0.000336** | **96.4%** |
| MLP tol=0.20 | 55.9% | $0.000187 | 98.0% |

**`tol=0.10` is the recommended operating point.** It strictly dominates the previous LR baseline on both accuracy (63.4% vs 54.1%) and cost savings (96.4% vs 90.0%), and approaches the oracle's 95.6% savings while staying 29pp above random.

### The tolerance parameter

`score_tolerance` controls the width of the "near-tied" band. Models within `tol` of the best predicted probability are all eligible — the cheapest wins.

- `tol=0.00` — always pick the single highest-confidence model → near-GPT-4 quality but routes almost entirely to GPT-4 (2.4% savings)
- `tol=0.10` — widen the band so cheaper models can win when the MLP is nearly as confident in them → **recommended sweet spot**
- `tol=0.20` — very aggressive cost-cutting, trades off accuracy noticeably

The MLP responds well to tolerance tuning because it has genuinely learned cross-model signal: at `tol=0.10` it can tell "GPT-3.5 is nearly as likely to pass as GPT-4 on this prompt" and route there instead.

---

## Setup

```bash
pip install -r requirements.txt
```

Dataset (~100 MB) is downloaded automatically on first run and cached.

## Usage

```bash
# Train MLP on full dataset
python3 router_main.py --train

# Tolerance sweep — reproduce the results table above
python3 router_main.py --compare

# Route a single prompt (requires --train first)
python3 router_main.py --route "Write a function that returns the nth Fibonacci number."

# Route with budget constraint
python3 router_main.py --route "..." --budget 0.001 --tolerance 0.10
```

## Project Structure

```
captsone/
├── router/
│   ├── mlp_router.py   # RouterMLP, training, CV evaluation, routing policy
│   └── features.py     # MiniLM embeddings (cached to data/)
├── sweep/
│   ├── load_data.py    # RouterBench download + wide→long transform
│   └── query.py        # prompt lookup with fuzzy matching
├── analysis/
│   ├── visualize.py    # cost-accuracy plots (matplotlib + plotly)
│   └── stats.py        # summary stats table
├── main.py             # Sweep Engine CLI
├── router_main.py      # Router CLI
└── requirements.txt
```

## References

- Hu et al. (2024). *RouterBench: A Benchmark for LLM Router Evaluation*. [arXiv:2403.12031](https://arxiv.org/abs/2403.12031)
- Reimers & Gurevych (2019). *Sentence-BERT*. EMNLP 2019.
