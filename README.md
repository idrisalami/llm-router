# LLM Router — Capstone Project

A capstone project on **LLM routing intelligence** — the problem of choosing the right language model for each task to maximise accuracy while minimising cost.

---

## Overview

Modern AI applications face a trade-off: large, expensive models (e.g. GPT-4) produce the best results, but are overkill for simpler tasks. Smaller models can handle many tasks at a fraction of the cost. A **router** sits in front of a pool of models and decides, per prompt, which model to call.

This project builds a complete routing pipeline on coding tasks using the [RouterBench](https://arxiv.org/abs/2403.12031) dataset (Hu et al., 2024):

- **Phase 1 & 2** — Sweep Engine: given a prompt, look up ground-truth performance across 11 LLMs and visualise the cost-accuracy trade-off.
- **Phase 3** — Router: train a logistic regression classifier on prompt embeddings to predict which model to route each new prompt to.

---

## Dataset

- Source: [`withmartian/routerbench`](https://huggingface.co/withmartian/routerbench), file `routerbench_0shot.pkl`
- **427 MBPP coding prompts** evaluated across **11 LLMs** (0-shot, no in-context examples)
- Per (prompt, model) entry: pass/fail quality score + total cost (USD, input + output tokens × API price at collection time)
- No latency data in the 0-shot file

### Models

| Model | Provider |
|---|---|
| `gpt-4-1106-preview` | OpenAI |
| `gpt-3.5-turbo-1106` | OpenAI |
| `claude-v2`, `claude-v1`, `claude-instant-v1` | Anthropic |
| `mistralai/mixtral-8x7b-chat` | Mistral |
| `mistralai/mistral-7b-chat` | Mistral |
| `meta/code-llama-instruct-34b-chat` | Meta |
| `meta/llama-2-70b-chat` | Meta |
| `WizardLM/WizardLM-13B-V1.2` | WizardLM |
| `zero-one-ai/Yi-34B-Chat` | 01.AI |

---

## Methodology

### Phase 1 & 2 — Sweep Engine

The sweep engine answers the question: *"For this prompt, how did each model actually perform?"*

**Data pipeline:**
```
RouterBench (HuggingFace, ~100 MB)
        │
        ▼
sweep/load_data.py     Download routerbench_0shot.pkl (one-time), melt wide format
                       → long format (one row per prompt×model), filter to MBPP,
                       cache as data/routerbench_coding.parquet
        │
        ▼
sweep/query.py         Match user query to dataset prompt via exact → substring →
                       token-overlap fallback. Return per-model accuracy + cost.
        │
        ▼
analysis/visualize.py  Accuracy bar, cost bar, Pareto frontier (PNG + interactive HTML).
                       Pareto frontier: sort models by cost ascending, keep a model
                       only if its accuracy strictly exceeds all cheaper models so far
                       (Non-Decreasing Convex Hull method from the RouterBench paper).
        │
        ▼
analysis/stats.py      Summary table: accuracy, cost, quality/dollar ratio, Pareto flag.
                       Saved as CSV + printed via Rich.
```

**Prompt matching** uses a three-tier strategy to handle partial or abbreviated user queries:
1. Exact string match
2. Case-insensitive substring match
3. Token-overlap match (only accepted with ≥4 tokens and ≥0.8 overlap score, to avoid spurious matches)

### Phase 3 — Logistic Regression Router

The router answers the question: *"For a new, unseen prompt, which model should I call?"*

**Architecture:** One binary logistic regression classifier per model, each predicting P(pass | prompt). At inference time, all 11 classifiers vote and the model with the highest predicted pass probability (within budget, if specified) is selected.

**Prompt representation:** Prompts are embedded with `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions). This model runs locally, requires no API key, and captures semantic meaning — so prompts about similar topics yield similar embeddings. Embeddings are cached to `data/embeddings_mbpp.npy` after the first computation.

**Training:**
- Feature matrix X: shape (427, 384) — one embedding per prompt
- Labels y: per-model binary array — 1 if the model passed, 0 otherwise
- Classifier: `StandardScaler → LogisticRegression(C=1.0, max_iter=1000)`
- Final classifiers are saved to `data/router_models/` via joblib

**Budget-aware routing:**
- Each model has a pre-computed average cost (training-set mean) as a proxy for expected cost
- If `--budget X` is set, only models with `avg_cost ≤ X` are eligible
- Among eligible models, the one with the highest P(pass) is selected
- If no model is within budget, fall back to the cheapest available

**Evaluation:** 5-fold cross-validation with `StratifiedKFold`. For each fold, classifiers are trained on 80% of prompts and evaluated on the held-out 20%. This ensures all reported metrics are **strictly out-of-sample** — no prompt is ever in both the training set and the test set for a given fold.

**Baselines compared:**
| Strategy | Description |
|---|---|
| always-best | Always use GPT-4 (highest avg accuracy) |
| always-cheapest | Always use Mistral-7B (lowest avg cost) |
| random | Pick a random model per prompt |
| oracle | Cheapest model that actually passes (theoretical ceiling, requires knowing the answer) |
| router (ours) | Logistic regression on prompt embeddings |

---

## Results

### Data landscape (from Phase 1 & 2)

On 0-shot MBPP, the **oracle** (always picking the cheapest model that actually passes) achieves **86.7% accuracy** at an average cost of **~$0.0004/query**. GPT-4, the single best model, achieves **68.6% accuracy** at **~$0.0094/query**. This 18-point accuracy gap between GPT-4 and the oracle reveals that no single model dominates — models fail and succeed on different prompts. This complementarity is what makes routing worthwhile.

### Router evaluation (Phase 3)

Results from 5-fold cross-validation on 425 MBPP prompts:

| Strategy | Accuracy | Avg cost | Savings vs GPT-4 |
|---|---|---|---|
| always-best (GPT-4) | 68.6% | $0.009377 | 0.0% |
| always-cheapest (Mistral-7B) | 34.2% | $0.000049 | 99.5% |
| random | 51.8% | $0.001999 | 78.7% |
| oracle (ceiling) | 86.6% | $0.000413 | 95.6% |
| **router — quality** (tol=0.00) | **64.9%** | **$0.003094** | **67.0%** |
| **router — balanced** (tol=0.02) | **56.5%** | **$0.001566** | **83.3%** |
| **router — economy** (tol=0.05) | **55.3%** | **$0.001272** | **86.4%** |

**What the numbers mean:**
- **Accuracy**: fraction of prompts where the model chosen by the router actually passed (ground-truth, held-out test set — strictly out-of-sample)
- **Avg cost**: average cost of the selected model per query (per-model training-set mean used as a proxy)
- **Savings**: cost reduction vs always-GPT-4

### The score tolerance parameter

The router exposes a `score_tolerance` knob (default `0.05`) that controls aggressiveness of cost-saving:

- **`tol=0.00` (quality)**: pick the single model with the highest predicted P(pass). Only exact ties are broken by cost. → 64.9% accuracy, $0.0031, **67% savings**.
- **`tol=0.02` (balanced)**: models within 2pp of the best are near-tied; cheapest wins. → 56.5% accuracy, $0.0016, **83.3% savings**.
- **`tol=0.05` (economy)**: models within 5pp of the best are near-tied; cheapest wins. → 55.3% accuracy, $0.0013, **86.4% savings**.

The three operating points are visible as a connected trade-off curve in `router_comparison.png`. Going from tol=0.00 to tol=0.02 costs ~8pp accuracy but saves an additional 16pp in cost. Going further to tol=0.05 yields only marginal additional savings (+3pp) for a small extra accuracy cost (−1pp) — meaning tol=0.02 is close to the "knee" of the trade-off curve.

```bash
# quality mode (default)
python3 router_main.py --route "Write a function to sort a list of tuples."

# economy mode (override tolerance)
# (set SCORE_TOLERANCE in router/predict.py, or pass score_tolerance= at call time)
```

### Model selection distribution

With the default `tol=0.05` (economy mode), routing is heavily shifted toward cheap models:

| Model | Prompts routed | Share |
|---|---|---|
| meta/code-llama-instruct-34b-chat | 81 | 19.1% |
| mistralai/mistral-7b-chat | 64 | 15.1% |
| mistralai/mixtral-8x7b-chat | 58 | 13.6% |
| gpt-3.5-turbo-1106 | 51 | 12.0% |
| WizardLM/WizardLM-13B-V1.2 | 38 | 8.9% |
| gpt-4-1106-preview | 31 | 7.3% |
| zero-one-ai/Yi-34B-Chat | 28 | 6.6% |
| claude-instant-v1 | 23 | 5.4% |
| claude-v2 | 18 | 4.2% |
| claude-v1 | 18 | 4.2% |
| meta/llama-2-70b-chat | 15 | 3.5% |

GPT-4 drops from 20.5% of traffic (quality mode) to 7.3% (economy mode). The router genuinely diversifies — no single model dominates.

### Confidence signal for downstream agents

Beyond the routing decision itself, the `scores` dict returned by the router gives a predicted P(pass) for every model. The **chosen model's score** is a direct confidence signal:

- **High confidence (P ≥ 0.7)**: the router is reasonably sure the chosen model will pass — proceed normally.
- **Low confidence (P < 0.5)**: the router is uncertain any model will handle this prompt well. This can be surfaced to users or upstream AI agents as a reliability warning, e.g.:
  - "This question may be difficult — consider verifying the output manually."
  - Trigger a retry with a stronger model.
  - Flag the result for human review in an agentic pipeline.

This confidence signal costs nothing extra — it is a by-product of the probability outputs already computed during routing.

### Diagnostic plots (`output/router/router_analysis.png`)

The four-panel analysis plot reveals the router's behaviour in detail:

**Panel 1 — Selection frequency (pass vs fail):** Stacked bar showing how many times each model was picked and how many of those passed. GPT-4 and GPT-3.5 dominate in volume, but models like mistral-7b are used sparingly (8 times total) and with low pass rate — indicating the router mostly avoids them but occasionally mis-routes there.

**Panel 2 — Pass rate by routed model:** For each model the router routed to, what fraction actually passed (bars), compared to that model's overall dataset accuracy (dashed grey line). If the bars were consistently *above* the dashed line, the router would be smartly routing only the prompts each model can handle. In practice the bars are near or below the dashed line — the router is not yet reliably identifying each model's strengths.

**Panel 3 — Per-fold accuracy:** Router accuracy across each of the 5 folds. Stable folds suggest the signal is consistent; high variance would indicate the router is sensitive to which prompts are held out.

**Panel 4 — Confidence histogram:** Distribution of the router's predicted P(pass) for the chosen model, split by whether the model actually passed (green) or failed (red). A well-calibrated router would show higher confidence on successes. Overlap between the distributions reflects the difficulty of the discrimination task.

### Interpretation

The router achieves a strong cost-accuracy trade-off: it captures most of GPT-4's quality (64.9% vs 68.6%) at one-third of the cost, and it genuinely diversifies across models rather than collapsing to a single choice.

That said, there is a large gap between the router (64.9%) and the oracle ceiling (86.6%). The main reasons:

1. **Hard discrimination task.** MBPP prompts are semantically similar coding problems. Sentence embeddings do not carry enough signal to reliably separate easy from hard problems per model — every per-model classifier sits at or below its majority-class baseline in CV.

2. **Limited data.** 425 examples is a small training set for 11 binary classifiers. More data, especially with richer prompt features (problem type, complexity, constraints), would help.

3. **Linear model.** Logistic regression is a linear classifier in embedding space. Non-linear models (gradient boosting, small neural networks) could capture more complex decision boundaries.

4. **Cost proxy.** The router uses per-model average cost as a budget proxy, not actual per-prompt token counts. A model with variable output length would be better served by prompt-conditioned cost estimates.

### Key takeaway

Even a simple logistic regression on 384-dim sentence embeddings delivers **67% cost savings at near-GPT-4 quality**, with routing genuinely spread across the model pool. The path from 64.9% to the 86.6% oracle ceiling is the main opportunity for future work.

---

## Setup

```bash
pip install -r requirements.txt
```

On first run, the dataset (~100 MB) is downloaded automatically from HuggingFace and cached to `data/routerbench_coding.parquet`. Subsequent runs load from cache instantly.

---

## Usage

### Phase 1 & 2 — Sweep Engine

```bash
# Browse available prompts
python3 main.py --list-prompts
python3 main.py --list-prompts --top 20

# Run analysis by prompt number (recommended)
python3 main.py --id 1
python3 main.py --id 42

# Run analysis by keyword search (fuzzy matching)
python3 main.py --prompt "fibonacci"
python3 main.py --prompt "sort a list of tuples"

# Options
python3 main.py --id 5 --output-dir results/my_run   # custom output directory
python3 main.py --id 5 --no-fuzzy                    # exact match only
python3 main.py --id 5 --force-reload                # re-download dataset
```

> If the prompt is not found exactly, fuzzy matching is attempted and a warning is shown. Use `--id` to avoid ambiguity.

### Phase 3 — Router

```bash
# Train classifiers + run cross-validated evaluation
python3 router_main.py --train

# Route a single prompt (requires --train first)
python3 router_main.py --route "Write a function to sort a list of tuples by the second element."

# Route with a cost budget (only consider models cheaper than $0.001/query)
python3 router_main.py --route "..." --budget 0.001

# Re-compute embeddings (e.g. after dataset update)
python3 router_main.py --train --recompute-embeddings
```

---

## Output files

### Phase 1 & 2 (`output/<prompt_slug>/`)

| File | Description |
|---|---|
| `accuracy_bar.png` | Pass@1 accuracy per model, sorted descending |
| `cost_bar.png` | Average cost per query (USD) per model |
| `pareto_frontier.png` | Cost vs. accuracy scatter with Pareto frontier (log x-axis) |
| `pareto_frontier.html` | Interactive version (plotly) |
| `summary_stats.csv` | Per-model: accuracy, cost, quality/dollar, Pareto flag |

### Phase 3 (`output/router/`)

| File | Description |
|---|---|
| `router_comparison.png` | Cost-accuracy plane: router vs all baselines + individual models |
| `router_analysis.png` | 4-panel diagnostic: selection frequency, pass rate by model, per-fold accuracy, confidence histogram |
| `cv_results.csv` | Per-prompt CV routing decisions (prompt, fold, routed model, quality, cost) |
| `comparison_summary.csv` | Aggregate metrics for all strategies |

---

## Project structure

```
captsone/
├── data/
│   ├── routerbench_coding.parquet      # auto-generated on first run
│   ├── embeddings_mbpp.npy             # prompt embeddings cache (384-dim)
│   ├── embeddings_mbpp_prompts.txt     # prompt list aligned with embeddings
│   └── router_models/
│       ├── classifiers.joblib          # trained logistic regression pipelines
│       └── model_costs.joblib          # per-model average costs
├── sweep/
│   ├── load_data.py                    # dataset download, wide→long, caching
│   └── query.py                        # prompt matching, per-model lookup
├── analysis/
│   ├── visualize.py                    # plots (matplotlib + plotly)
│   └── stats.py                        # summary stats table + CSV
├── router/
│   ├── features.py                     # sentence-transformer embeddings
│   ├── train.py                        # classifier training + CV
│   ├── predict.py                      # routing policy (budget-aware)
│   └── evaluate.py                     # CV evaluation + baselines + plots
├── main.py                             # Phase 1 & 2 CLI
├── router_main.py                      # Phase 3 CLI
└── requirements.txt
```

---

## References

- Hu et al. (2024). *RouterBench: A Benchmark for LLM Router Evaluation*. [arXiv:2403.12031](https://arxiv.org/abs/2403.12031)
- Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019.
