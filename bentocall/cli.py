"""bentocall CLI — recursive long-context solver.

Reads a document from stdin or --file, runs bentocall, prints JSON to stdout.
Stderr carries progress + errors.

Exit codes:
  0  success
  1  cost-cap hit ($BENTOCALL_DAILY_CAP exceeded)
  2  task-spec error (bad flag, unknown task, empty input)
  3  upstream API error after retries
  4  internal/parse error
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

from bentocall.api import solve, SUPPORTED_TASKS
from bentocall.llm import CALL_LOG


def _today_spend_usd() -> float:
    if not CALL_LOG.exists():
        return 0.0
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    total = 0.0
    for line in CALL_LOG.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = r.get("ts", 0)
        if datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d") == today:
            total += r.get("cost_usd", 0.0)
    return total


def _check_cap() -> None:
    cap = os.environ.get("BENTOCALL_DAILY_CAP") or os.environ.get("LRLM_DAILY_CAP")
    if not cap:
        return
    try:
        cap_usd = float(cap)
    except ValueError:
        print(f"warn: BENTOCALL_DAILY_CAP={cap!r} is not a number; ignoring", file=sys.stderr)
        return
    spent = _today_spend_usd()
    if spent >= cap_usd:
        print(
            f"bentocall: daily cost cap hit ({spent:.4f} ≥ {cap_usd:.4f} USD). "
            f"Unset BENTOCALL_DAILY_CAP or wait until UTC midnight.",
            file=sys.stderr,
        )
        sys.exit(1)


def _self_test_ool_pairs() -> int:
    from bentocall.tasks.oolong_pairs import generate, lambda_rlm, f1
    case = generate(n_items=30, n_users=10, target_hit_rate=0.5, seed=42)
    pred, meta = lambda_rlm(case, model="haiku", K=400)
    s = f1(pred, case.gold_pairs)
    ok = s["f1"] == 1.0
    msg_kind = "OK" if ok else "FAIL"
    print(
        f"{msg_kind} task=ool_pairs F1={s['f1']:.3f} "
        f"P={s['precision']:.3f} R={s['recall']:.3f} leaves={meta['leaf_calls']}",
        file=sys.stderr,
    )
    return 0 if ok else 4


def _self_test_aggregate_counts() -> int:
    from bentocall.tasks.aggregate_counts import generate, lambda_rlm, score as agg_score
    case = generate(n_items=60, n_categories=4, seed=42)
    pred, meta = lambda_rlm(case, model="haiku", K=400)
    s = agg_score(pred, case.gold_counts)
    ok = s["rel_error"] < 0.05
    msg_kind = "OK" if ok else "FAIL"
    note = "" if s["exact_match"] else " (within tolerance — see docstring)"
    print(
        f"{msg_kind} task=aggregate_counts rel_err={s['rel_error']:.3f} "
        f"abs_err={s['abs_error']}{note} "
        f"gold={case.gold_counts} pred={pred} leaves={meta['leaf_calls']}",
        file=sys.stderr,
    )
    return 0 if ok else 4


def _self_test(task: str) -> int:
    if task == "ool_pairs":
        return _self_test_ool_pairs()
    if task == "aggregate_counts":
        return _self_test_aggregate_counts()
    print(f"bentocall: no self-test for task '{task}'", file=sys.stderr)
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(prog="bentocall")
    ap.add_argument("--task", default="ool_pairs", choices=list(SUPPORTED_TASKS))
    ap.add_argument("--file", help="read document from path (else stdin)")
    ap.add_argument("--K", type=int, help="leaf token budget; auto if omitted")
    ap.add_argument(
        "--target",
        default="abbreviation,description and abstract concept",
        help="comma-separated target labels for ool_pairs",
    )
    ap.add_argument("--model", default="haiku", choices=["haiku", "sonnet"])
    ap.add_argument("--force-lambda", action="store_true",
                    help="skip the auto-routing decision; always run lambda-rlm")
    ap.add_argument("--self-test", action="store_true",
                    help="run canned self-test for the chosen --task")
    args = ap.parse_args()

    _check_cap()

    if args.self_test:
        return _self_test(args.task)

    if args.file:
        try:
            from pathlib import Path
            document = Path(args.file).read_text()
        except OSError as e:
            print(f"bentocall: cannot read --file: {e}", file=sys.stderr)
            return 2
    else:
        if sys.stdin.isatty():
            print("bentocall: no document provided (stdin is a TTY and --file not given)",
                  file=sys.stderr)
            return 2
        document = sys.stdin.read()

    if not document.strip():
        print("bentocall: empty document", file=sys.stderr)
        return 2

    target_labels = tuple(t.strip() for t in args.target.split(",") if t.strip())

    t0 = time.time()
    try:
        result = solve(
            document,
            task=args.task,
            target_labels=target_labels,
            K=args.K,
            model=args.model,
            force_lambda=args.force_lambda,
        )
    except RuntimeError as e:
        print(f"bentocall: upstream error: {e}", file=sys.stderr)
        return 3
    except Exception as e:  # noqa: BLE001
        print(f"bentocall: internal error: {e!r}", file=sys.stderr)
        return 4

    result["wall_s"] = round(time.time() - t0, 2)
    result["today_spend_usd"] = round(_today_spend_usd(), 4)
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
