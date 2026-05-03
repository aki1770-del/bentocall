"""aggregate_counts task — Paper Appendix A worked-example shape.

Document: many records, each with a category label drawn from a small set.
Task: emit `{category: total_count}` over the whole document.

Plan: Split → Map(M) → MergeCounts. Different from ool_pairs in three ways:
  - Output is a dict, not a list of pairs.
  - Reducer is dict-sum-merge, not list-flatten + Cross.
  - No post-recursion symbolic step — the merged dict IS the answer.
"""
from __future__ import annotations

import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

from bentocall.combinators import token_count
from bentocall.llm import M
from bentocall.planner import plan as make_plan
from bentocall.runtime import run as rlm_run

CATEGORIES = (
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
)


@dataclass
class AggregateCase:
    document: str
    gold_counts: Dict[str, int]
    n_items: int
    n_categories: int
    n_tokens: int


def generate(
    n_items: int = 60,
    n_categories: int = 6,
    seed: int = 0,
) -> AggregateCase:
    """Synthetic doc of `record N: type=<cat>` lines, with deterministic gold."""
    rng = random.Random(seed)
    cats = list(CATEGORIES[:n_categories])
    counter: Counter = Counter()
    lines: List[str] = []
    for i in range(1, n_items + 1):
        c = rng.choice(cats)
        counter[c] += 1
        filler = rng.choice(["normal", "high-priority", "stale", "fresh", "deferred"])
        lines.append(f"record {i:04d}: type={c} status={filler}")
    document = "\n".join(lines)
    return AggregateCase(
        document=document,
        gold_counts=dict(counter),
        n_items=n_items,
        n_categories=n_categories,
        n_tokens=token_count(document),
    )


# ---------- Leaf + reducer ----------

LEAF_PROMPT = (
    "Count records by category in this chunk. Each line has the form\n"
    "  record <id>: type=<category> status=<word>\n"
    "Categories are drawn from this fixed set: alpha, beta, gamma, delta, "
    "epsilon, zeta.\n\n"
    "Emit ONE JSON object on a single line that maps each present category "
    "to its count in this chunk. Skip categories with zero count. "
    "Output nothing else. Example:\n"
    '  {"alpha": 12, "gamma": 7}\n\n'
    "Chunk:\n"
)

LINE_RE = re.compile(r"^record\s+\d+\s*:\s*type=(\w+)\s")


def _leaf_count(chunk: str, model: str) -> Dict[str, int]:
    """Run M on a chunk; parse a single JSON-dict line into {cat: count}."""
    out = M(
        prompt=chunk,
        system=LEAF_PROMPT,
        model=model,
        tag="leaf",
        max_tokens=512,
        cache_system=True,
    )
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            return {str(k): int(v) for k, v in obj.items() if int(v) > 0}
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    counts: Counter = Counter()
    for raw in chunk.splitlines():
        m = LINE_RE.match(raw)
        if m:
            counts[m.group(1)] += 1
    return dict(counts)


def _merge_counts(parts: List[Dict[str, int]]) -> Dict[str, int]:
    """Dict-sum reducer. Commutative + associative."""
    merged: Counter = Counter()
    for p in parts:
        merged.update(p)
    return dict(merged)


def flat_baseline(case: AggregateCase, model: str = "sonnet") -> Tuple[Dict[str, int], dict]:
    """One LLM call asking for the count dict. Used as the cost-inversion-zone
    fallback by `api.solve()` for small aggregate_counts inputs."""
    prompt = (
        "Count records by category in this document. Each line has the form\n"
        "  record <id>: type=<category> status=<word>\n"
        "Categories may include: alpha, beta, gamma, delta, epsilon, zeta.\n\n"
        "Emit ONE JSON object on a single line that maps each present category "
        "to its count. Output nothing else. Example: {\"alpha\": 12, \"gamma\": 7}\n\n"
        f"Document:\n{case.document}\n"
    )
    t0 = time.time()
    raw = M(prompt, model=model, tag="flat", max_tokens=512)
    pred: Dict[str, int] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            pred = {str(k): int(v) for k, v in obj.items() if int(v) > 0}
            break
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return pred, {"latency_s": round(time.time() - t0, 3)}


def lambda_rlm(case: AggregateCase, model: str, K: int = 2000) -> Tuple[Dict[str, int], dict]:
    """Recursive solver. Split → Map(M) → MergeCounts."""
    pl = make_plan(case.n_tokens, K)
    t0 = time.time()
    result = rlm_run(
        case.document,
        plan=pl,
        leaf_fn=lambda chunk: _leaf_count(chunk, model),
        reduce_fn=_merge_counts,
    )
    return result.answer, {
        "plan": vars(pl),
        "leaf_calls": result.leaf_count,
        "latency_s": round(time.time() - t0, 3),
    }


def score(predicted: Dict[str, int], gold: Dict[str, int]) -> dict:
    """Per-category L1 error; exact-match if total error == 0."""
    cats = set(predicted) | set(gold)
    abs_err = sum(abs(predicted.get(c, 0) - gold.get(c, 0)) for c in cats)
    total_gold = sum(gold.values()) or 1
    return {
        "abs_error": abs_err,
        "rel_error": round(abs_err / total_gold, 4),
        "exact_match": abs_err == 0,
        "n_categories_gold": len(gold),
        "n_categories_pred": len(predicted),
    }
