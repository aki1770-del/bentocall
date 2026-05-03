"""Single-call entry point.

    from bentocall import solve
    result = solve(document_text, task="ool_pairs")
    # result["routing"] reveals which path ran: "lambda-rlm" or "flat-sonnet"

Routing rules (defaults — re-derive on your workload before relying on them):
- `ool_pairs`        : ≥4K tokens → lambda-rlm; <4K → Sonnet flat
- `aggregate_counts` : ≥8K tokens → lambda-rlm; <8K → Sonnet flat

Below threshold, Sonnet flat is both as accurate and cheaper than recursion,
so bentocall transparently falls through to flat. Same JSON schema either way.
"""
from __future__ import annotations

from typing import Optional, Tuple

from bentocall.combinators import token_count
from bentocall.tasks.aggregate_counts import (
    AggregateCase,
    flat_baseline as agg_flat,
    lambda_rlm as agg_lambda_rlm,
)
from bentocall.tasks.oolong_pairs import (
    OOLPairsCase,
    flat_baseline as ool_flat,
    lambda_rlm as ool_lambda_rlm,
    LINE_RE,
)

SUPPORTED_TASKS = ("ool_pairs", "aggregate_counts")
DEFAULT_TARGET_LABELS = ("abbreviation", "description and abstract concept")

# Minimum doc tokens at which lambda-rlm beats Sonnet flat in the validation
# batch (2026-05-03). Below the threshold, route to Sonnet flat.
ROUTE_THRESHOLDS = {
    "ool_pairs": 4000,
    "aggregate_counts": 8000,
}


def _detect_irregular(doc: str) -> bool:
    lines = [l for l in doc.splitlines() if l.strip()][:50]
    if not lines:
        return True
    matches = sum(1 for l in lines if LINE_RE.match(l))
    return matches < len(lines) * 0.5


def _auto_K(n_tokens: int) -> int:
    if n_tokens <= 4000:
        return 800
    if n_tokens <= 16000:
        return 2000
    if n_tokens <= 64000:
        return 5000
    return 8000


def _should_route_flat(task: str, n_tokens: int) -> bool:
    return n_tokens < ROUTE_THRESHOLDS.get(task, 0)


def _solve_ool_pairs_lrlm(document, target_labels, K, model: str):
    n_tok = token_count(document)
    K = K or _auto_K(n_tok)
    irregular = _detect_irregular(document)
    case = OOLPairsCase(
        document=document, gold_pairs=set(),
        target_labels=set(target_labels),
        n_items=0, n_users=0, n_tokens=n_tok, irregular=irregular,
    )
    pred, meta = ool_lambda_rlm(case, model=model, K=K)
    pl = meta.get("plan", {})
    return {
        "answer": sorted([list(p) for p in pred]),
        "trace": {
            "leaves": meta.get("leaf_calls"),
            "depth": pl.get("depth"),
            "k_star": pl.get("k_star"),
            "tau_star": pl.get("tau_star"),
            "wall_s": meta.get("latency_s"),
        },
        "format_detected": "irregular" if irregular else "regular",
        "n_tokens": n_tok,
        "K": K,
        "target_labels": list(target_labels),
        "routing": "lambda-rlm",
        "model": model,
    }


def _solve_aggregate_counts_lrlm(document, K, model):
    n_tok = token_count(document)
    K = K or _auto_K(n_tok)
    case = AggregateCase(
        document=document, gold_counts={},
        n_items=0, n_categories=0, n_tokens=n_tok,
    )
    pred, meta = agg_lambda_rlm(case, model=model, K=K)
    pl = meta.get("plan", {})
    return {
        "answer": dict(sorted(pred.items())),
        "trace": {
            "leaves": meta.get("leaf_calls"),
            "depth": pl.get("depth"),
            "k_star": pl.get("k_star"),
            "tau_star": pl.get("tau_star"),
            "wall_s": meta.get("latency_s"),
        },
        "n_tokens": n_tok,
        "K": K,
        "routing": "lambda-rlm",
        "model": model,
    }


def _solve_ool_pairs_flat(document, target_labels):
    n_tok = token_count(document)
    irregular = _detect_irregular(document)
    case = OOLPairsCase(
        document=document, gold_pairs=set(),
        target_labels=set(target_labels),
        n_items=0, n_users=0, n_tokens=n_tok, irregular=irregular,
    )
    pred, meta = ool_flat(case, model="sonnet")
    return {
        "answer": sorted([list(p) for p in pred]),
        "trace": {"leaves": 0, "depth": 0, "wall_s": meta.get("latency_s")},
        "format_detected": "irregular" if irregular else "regular",
        "n_tokens": n_tok,
        "target_labels": list(target_labels),
        "routing": "flat-sonnet",
        "model": "sonnet",
    }


def _solve_aggregate_counts_flat(document):
    n_tok = token_count(document)
    case = AggregateCase(
        document=document, gold_counts={},
        n_items=0, n_categories=0, n_tokens=n_tok,
    )
    pred, meta = agg_flat(case, model="sonnet")
    return {
        "answer": dict(sorted(pred.items())),
        "trace": {"leaves": 0, "depth": 0, "wall_s": meta.get("latency_s")},
        "n_tokens": n_tok,
        "routing": "flat-sonnet",
        "model": "sonnet",
    }


def solve(
    document: str,
    *,
    task: str = "ool_pairs",
    target_labels: Tuple[str, ...] = DEFAULT_TARGET_LABELS,
    K: Optional[int] = None,
    model: str = "haiku",
    force_lambda: bool = False,
) -> dict:
    """Solve a long-context task. Auto-routes between lambda-rlm and Sonnet-flat
    based on token count + task type. `force_lambda=True` skips the routing
    decision (useful for benchmarking)."""
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"task '{task}' not yet supported. Available: {SUPPORTED_TASKS}"
        )

    n_tok = token_count(document)
    route_flat = (not force_lambda) and _should_route_flat(task, n_tok)

    if route_flat:
        if task == "ool_pairs":
            result = _solve_ool_pairs_flat(document, target_labels)
        else:
            result = _solve_aggregate_counts_flat(document)
    else:
        if task == "ool_pairs":
            result = _solve_ool_pairs_lrlm(document, target_labels, K, model)
        else:
            result = _solve_aggregate_counts_lrlm(document, K, model)

    result["task"] = task
    return result
