"""Drift-rate measurement across seeds.

Definition:
    rel_err = sum_c |pred[c] - gold[c]| / sum_c gold[c]

A single-seed rel_err is a sample. Real drift is a distribution; this module
runs N seeds against the configured backend and reports mean / median / p90 /
p95 / max so you can pick a model that holds at the tail, not just the mean.

Usage (CLI):
    python -m bentocall.research.drift --seeds 5 --task aggregate_counts
    BENTOCALL_BASE_URL=http://127.0.0.1:8765/v1/chat/completions \\
    BENTOCALL_MODEL_HAIKU=hermes-3-llama-3.1-8b \\
    python -m bentocall.research.drift --seeds 10 --task aggregate_counts

Usage (library):
    from bentocall.research.drift import measure
    stats = measure(task="aggregate_counts", n_seeds=10, model="haiku")
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import List

from bentocall.tasks.aggregate_counts import (
    generate as gen_agg, lambda_rlm as agg_lambda, score as agg_score,
)
from bentocall.tasks.oolong_pairs import (
    generate as gen_pairs, lambda_rlm as pairs_lambda, f1 as pairs_f1,
)


@dataclass
class SeedRun:
    seed: int
    rel_err: float
    abs_err: int
    capability: float    # F1 for ool_pairs, 1.0/0.0 for exact-match on aggregate
    wall_s: float


def _run_one(task: str, seed: int, model: str, K: int) -> SeedRun:
    t0 = time.time()
    if task == "aggregate_counts":
        case = gen_agg(n_items=60, n_categories=4, seed=seed)
        pred, _ = agg_lambda(case, model=model, K=K)
        s = agg_score(pred, case.gold_counts)
        return SeedRun(
            seed=seed, rel_err=s["rel_error"], abs_err=s["abs_error"],
            capability=1.0 if s["exact_match"] else 0.0,
            wall_s=round(time.time() - t0, 2),
        )
    elif task == "ool_pairs":
        case = gen_pairs(n_items=30, n_users=10, target_hit_rate=0.5, seed=seed)
        pred, _ = pairs_lambda(case, model=model, K=K)
        s = pairs_f1(pred, case.gold_pairs)
        # Treat (1 - F1) as the "drift" analogue for ool_pairs.
        return SeedRun(
            seed=seed, rel_err=round(1.0 - s["f1"], 4),
            abs_err=s["fp"] + s["fn"], capability=s["f1"],
            wall_s=round(time.time() - t0, 2),
        )
    else:
        raise ValueError(f"unsupported task: {task}")


def measure(task: str, n_seeds: int = 5, model: str = "haiku", K: int = 400,
            seed_start: int = 100) -> dict:
    runs: List[SeedRun] = []
    for i in range(n_seeds):
        runs.append(_run_one(task, seed_start + i, model, K))
    rels = [r.rel_err for r in runs]
    caps = [r.capability for r in runs]
    return {
        "task": task,
        "model": model,
        "n_seeds": n_seeds,
        "rel_err": {
            "mean":   round(statistics.fmean(rels), 4),
            "median": round(statistics.median(rels), 4),
            "p90":    round(statistics.quantiles(rels, n=10)[8] if len(rels) >= 10 else max(rels), 4),
            "max":    round(max(rels), 4),
            "min":    round(min(rels), 4),
        },
        "capability": {
            "mean":   round(statistics.fmean(caps), 3),
            "min":    round(min(caps), 3),
            "n_holding_>=0.95": sum(1 for c in caps if c >= 0.95),
        },
        "per_seed": [asdict(r) for r in runs],
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="bentocall.research.drift")
    ap.add_argument("--task", required=True, choices=["ool_pairs", "aggregate_counts"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--K", type=int, default=400)
    ap.add_argument("--seed-start", type=int, default=100)
    args = ap.parse_args()
    stats = measure(args.task, n_seeds=args.seeds, model=args.model,
                    K=args.K, seed_start=args.seed_start)
    json.dump(stats, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
