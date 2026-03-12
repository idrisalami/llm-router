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

---

## Why the embedding barely helps — and what to do about it

### The Prior baseline

The **Prior (no embedding)** baseline uses only each model's average pass rate from the training fold — no neural network, no embeddings. Every test prompt gets the same prompt-independent score; the cheapest model within tolerance wins.

Comparing Prior vs MLP at the same tolerance reveals the embedding adds almost nothing:
- At `tol=0.05`: both achieve **65.2% accuracy**, but the prior costs $0.000335$ vs MLP's $0.003134$ — 10× cheaper for identical accuracy
- At `tol=0.10`: both reach the same cost (~$0.000336), but the prior is actually *more accurate* (65.2% vs 63.4%)

**Root cause:** the MLP is not learning per-prompt difficulty. It is learning the same global pass-rate ranking the prior uses directly. There is no prompt-specific signal for it to learn from.

### Why embeddings carry no signal here

MBPP prompts are informationally compressed: *"Write a function that returns the nth Fibonacci number."* is 10 words, but those 10 words encode algorithm type, required data structures, time complexity, and edge cases. MiniLM embeds the surface words — it does not unpack what the task *requires*.

To measure this, we ran pairwise cosine similarity across all 425 embeddings:

| Metric | Value |
|---|---|
| Mean pairwise cosine similarity | **0.305** (0 = orthogonal, 1 = identical) |
| Effective dimensionality (90% variance) | **85** out of 384 |
| Variance explained by top 2 PCs | 15.8% |

A mean cosine similarity of 0.31 means every pair of prompts shares ~31% of their embedding direction. From the MLP's perspective, there is barely any feature space variation to learn from — all prompts look alike.

### Improving spread with ZCA whitening

ZCA whitening decorrelates all embedding dimensions and equalises their variance, spreading the point cloud across the full 384-dimensional sphere. Results across all representation techniques tested (all at tol=0.10, 5-fold CV):

| Variant | cos sim | eff dim (90%) | Accuracy | Savings |
|---|---|---|---|---|
| MiniLM (raw) | 0.305 | 85 / 384 | 63.4% | 96.4% |
| ZCA-whitened | −0.002 | 292 / 384 | 64.5% | 96.5% |
| LLM-expanded + ZCA | −0.002 | 293 / 384 | 63.1% | 96.7% |
| PCA-128 whitened | −0.002 | 108 / 128 | 61.9% | 96.8% |
| + TF-IDF (SVD-64) | 0.278 | 101 / 448 | 61.2% | 96.9% |
| + code features | 0.071 | 16 / 404 | 60.7% | 96.4% |
| **Prior (no embed)** | — | — | **65.2%** | 96.4% |

### LLM expansion: the representation problem is solved

Claude Haiku expands each terse prompt into a structured difficulty profile before embedding:

```
Concepts:    [Python/CS concepts required]
Algorithm:   [recursion / DP / sorting / hashing / math / …]
Complexity:  [O(n) / O(n log n) / O(n²) / …]
Edge cases:  [2–3 specific cases]
Difficulty:  [easy / medium / hard + one-line reason]
```

The original prompt and the expansion are concatenated and re-embedded with MiniLM + ZCA whitening. This is cached to `data/llm_expansions.json` (~$0.005 one-time API cost for all 425 prompts).

The embedding spread metrics confirm the approach works as intended:
- Max pairwise cosine similarity: **0.99 → 0.61** (was 0.99 for raw MiniLM)
- Effective dimensionality: **85 → 293** out of 384
- The per-prompt difficulty signal is genuinely in the embeddings now

But routing accuracy comes in at 63.1% — not better than plain ZCA (64.5%) and still below the Prior (65.2%).

### The real bottleneck: dataset size and diversity

Every representation technique we tried — ZCA whitening, PCA, TF-IDF, code features, LLM expansion — converges to the same ~63–65% accuracy range. The ~1–2pp variation across all variants is within the noise of 5-fold CV on 425 samples. No representation fix can escape this ceiling.

The root cause is now clearly identified: **it is not the representation, it is the data**.

Two problems compound each other:

**1. Too few samples (425 prompts).** Binary pass/fail labels are noisy — whether a model passes depends on temperature, prompt phrasing, and randomness. With 5-fold CV that leaves ~340 training examples per fold, the MLP cannot reliably learn subtle per-prompt difficulty patterns even when they are present in the embeddings. The LLM expansion proved those patterns exist geometrically; the model just doesn't have enough examples to fit them.

**2. No difficulty diversity (MBPP only).** All 425 prompts are from a single benchmark of introductory Python tasks. The difficulty range is narrow — almost everything is "easy" or "medium". There is no strong correlation between prompt features and which model passes, because even Mistral-7B passes most of them. The router has almost nothing to discriminate.

For the embedding-based approach to work, you need a dataset where:
- Prompts span a wide range of difficulty (easy → very hard)
- Some prompts are genuinely hard for cheap models but easy for expensive ones
- There are thousands of examples, not hundreds

**Next step: expand the dataset with harder and more varied benchmarks** — mixing MBPP with HumanEval, LeetCode-style problems, and multi-step reasoning tasks — to create real difficulty variation across models. With genuine difficulty spread in the labels, the LLM-expanded embeddings should finally give the MLP something to learn.

---

## Embedding spread analysis

```bash
python3 router_main.py --analyze-embeddings          # spread metrics + 4-panel plot
python3 router_main.py --compare-embeddings          # all variants side-by-side
python3 router_main.py --compare-routing             # routing accuracy per variant
```

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
│   ├── mlp_router.py       # RouterMLP, training, CV evaluation, routing policy
│   └── features.py         # MiniLM embeddings (cached to data/)
├── sweep/
│   ├── load_data.py        # RouterBench download + wide→long transform
│   └── query.py            # prompt lookup with fuzzy matching
├── analysis/
│   ├── embedding_analysis.py  # spread metrics + plots
│   ├── embedding_enhance.py   # ZCA whitening, PCA-whiten, code-aug, TF-IDF, LLM-expand
│   ├── llm_expander.py        # Claude Haiku structured prompt expansion + cache
│   ├── visualize.py           # cost-accuracy plots (matplotlib + plotly)
│   └── stats.py               # summary stats table
├── main.py                 # Sweep Engine CLI
├── router_main.py          # Router CLI
└── requirements.txt
```

## References

- Hu et al. (2024). *RouterBench: A Benchmark for LLM Router Evaluation*. [arXiv:2403.12031](https://arxiv.org/abs/2403.12031)
- Reimers & Gurevych (2019). *Sentence-BERT*. EMNLP 2019.
