# LLM Router

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

All metrics from strict 5-fold cross-validation (no prompt seen in both train and test).

| Strategy | tol | Accuracy | Avg cost | Savings vs GPT-4 |
|---|---|---|---|---|
| always-best (GPT-4) | — | 68.6% | $0.009377 | 0.0% |
| always-cheapest (Mistral-7B) | — | 34.2% | $0.000049 | 99.5% |
| random | — | 51.8% | $0.001999 | 78.7% |
| oracle (theoretical ceiling) | — | 86.6% | $0.000413 | 95.6% |
| Prior (no embedding) | 0.00 | 68.6% | $0.009377 | 0.0% |
| Prior (no embedding) | 0.05 | 65.2% | $0.000335 | 96.4% |
| Prior (no embedding) | 0.10 | 65.2% | $0.000335 | 96.4% |
| Prior (no embedding) | 0.20 | 51.5% | $0.000156 | 98.3% |
| MLP | 0.00 | **69.1%** | $0.009149 | 2.4% |
| MLP | 0.05 | 65.2% | $0.003134 | 66.6% |
| **MLP** | **0.10** | 63.4% | $0.000336 | **96.4%** |
| MLP | 0.20 | 55.9% | $0.000187 | 98.0% |

### The tolerance parameter

`score_tolerance` widens the selection band: all models within `tol` of the best predicted P(pass) are eligible — the cheapest wins.

- `tol=0.00` — always pick the single highest-confidence model → routes almost entirely to GPT-4
- `tol=0.10` — recommended sweet spot: large cost savings while staying well above random
- `tol=0.20` — very aggressive, trades off accuracy noticeably

### Sanity check: does the embedding help?

The **Prior (no embedding)** baseline uses only each model's average pass rate from the training fold as a prompt-independent score, then applies the same tolerance policy — no neural network, no embeddings.

Comparing Prior vs MLP at the same tolerance reveals the embedding adds very little for MBPP:
- At `tol=0.05`: both achieve **65.2% accuracy**, but prior costs $0.000335 vs MLP's $0.003134 — the prior is 10× cheaper for identical accuracy
- At `tol=0.10`: both reach the same cost (~$0.000336), but prior is actually *more accurate* (65.2% vs 63.4%)

**Conclusion:** for MBPP, the MLP is essentially learning the same global pass-rate ranking the prior uses directly. All 425 prompts are semantically near-identical ("write a Python function…"), so MiniLM embeddings carry almost no per-prompt difficulty signal. The router would need richer features (code complexity, AST structure, etc.) to genuinely outperform the prior.

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
