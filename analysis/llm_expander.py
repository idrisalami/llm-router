"""LLM-based prompt expansion for richer embeddings.

Each terse MBPP prompt ("Write a function that returns the nth Fibonacci number.")
is expanded into a structured difficulty profile by Claude Haiku.  The expanded
text has genuine lexical and semantic variance across difficulty dimensions
(algorithm type, complexity, edge cases) that the original prompt lacks.

The expanded text is concatenated with the original prompt before embedding,
so the MiniLM vector carries both the original meaning and the unpacked signal.

Expansions are cached to data/llm_expansions.json — the API is called only once
per unique prompt.  Re-run with force_recompute=True to refresh.

Usage
-----
    from analysis.llm_expander import load_or_expand

    expanded_prompts = load_or_expand(prompts)   # list[str], same order as input
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CACHE_PATH = Path(__file__).parent.parent / "data" / "llm_expansions.json"

_SYSTEM_PROMPT = """\
You are a code difficulty analyzer. Given a Python programming task, output a \
concise structured analysis using exactly these fields on separate lines:

Concepts: [comma-separated list of Python/CS concepts required]
Algorithm: [primary algorithm or technique, e.g. recursion, DP, sorting, hashing, \
two-pointer, math, string manipulation]
Complexity: [time complexity, e.g. O(n), O(n log n), O(n²)]
Edge cases: [2–3 specific edge cases separated by commas]
Difficulty: [easy / medium / hard — one short reason]

Be specific and technical. Keep the total response under 80 words.\
"""


def _prompt_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def _call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def _call_llm(prompt: str) -> str:
    return _call_anthropic(prompt)


def load_or_expand(
    prompts: list[str],
    force_recompute: bool = False,
) -> list[str]:
    """Return expanded prompt strings (original + structured analysis).

    Loads cached expansions where available; calls Haiku for new prompts.
    Returns strings in the same order as `prompts`.
    """
    cache: dict[str, str] = {}
    if CACHE_PATH.exists() and not force_recompute:
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    missing = [p for p in prompts if _prompt_key(p) not in cache]

    if missing:
        print(f"[expander] Expanding {len(missing)} prompts via Claude Haiku…")
        for i, p in enumerate(missing, 1):
            key = _prompt_key(p)
            try:
                expansion = _call_llm(p)
                cache[key] = expansion
            except Exception as exc:
                print(f"  [warn] prompt {i} failed: {exc} — using empty expansion")
                cache[key] = ""
            if i % 50 == 0 or i == len(missing):
                print(f"  {i}/{len(missing)} done")
                _save_cache(cache)

        _save_cache(cache)
        print(f"[expander] Expansions cached → {CACHE_PATH}")
    else:
        print(f"[expander] All {len(prompts)} expansions loaded from cache.")

    return [_combine(p, cache[_prompt_key(p)]) for p in prompts]


def _combine(original: str, expansion: str) -> str:
    if expansion:
        return f"{original}\n\n{expansion}"
    return original


def _save_cache(cache: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
