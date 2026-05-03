"""bentocall-usage — roll up calls.jsonl into spend windows + savings estimate."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from bentocall.llm import CALL_LOG

SONNET_IN_PER_MTOK = 3.0
SONNET_OUT_PER_MTOK = 15.0


def _sonnet_equiv_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * SONNET_IN_PER_MTOK + output_tokens * SONNET_OUT_PER_MTOK
    ) / 1_000_000


def main() -> int:
    ap = argparse.ArgumentParser(prog="bentocall-usage")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--today", action="store_true", help="(default) just today UTC")
    g.add_argument("--week", action="store_true", help="last 7 days")
    g.add_argument("--month", action="store_true", help="last 30 days")
    g.add_argument("--all", action="store_true", help="all-time")
    ap.add_argument("--by-tag", action="store_true",
                    help="break down by call tag (leaf, flat, probe, ...)")
    ap.add_argument("--savings", action="store_true",
                    help="estimate savings vs all-Sonnet counterfactual")
    args = ap.parse_args()

    if not CALL_LOG.exists():
        print("(no calls logged yet — run bentocall first)")
        return 0

    now = datetime.now(timezone.utc)
    if args.week:
        cutoff, label = now - timedelta(days=7), "last 7 days"
    elif args.month:
        cutoff, label = now - timedelta(days=30), "last 30 days"
    elif args.all:
        cutoff, label = datetime.fromtimestamp(0, tz=timezone.utc), "all-time"
    else:
        cutoff, label = now.replace(hour=0, minute=0, second=0, microsecond=0), "today (UTC)"

    by_model: dict = defaultdict(lambda: {"calls": 0, "cost": 0.0,
                                           "in_tok": 0, "out_tok": 0,
                                           "sonnet_equiv": 0.0})
    by_tag: dict = defaultdict(lambda: {"calls": 0, "cost": 0.0})
    total_calls = 0
    total_cost = 0.0
    total_sonnet_equiv = 0.0
    for line in CALL_LOG.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = r.get("ts", 0)
        when = datetime.fromtimestamp(ts, tz=timezone.utc)
        if when < cutoff:
            continue
        cost = r.get("cost_usd", 0.0)
        in_tok = r.get("input_tokens", 0)
        out_tok = r.get("output_tokens", 0)
        m = r.get("model", "?")
        by_model[m]["calls"] += 1
        by_model[m]["cost"] += cost
        by_model[m]["in_tok"] += in_tok
        by_model[m]["out_tok"] += out_tok
        if "sonnet" in m.lower():
            sonnet_equiv = cost
        else:
            sonnet_equiv = _sonnet_equiv_cost(in_tok, out_tok)
        by_model[m]["sonnet_equiv"] += sonnet_equiv
        total_sonnet_equiv += sonnet_equiv
        if args.by_tag:
            tag = r.get("tag") or "(none)"
            by_tag[tag]["calls"] += 1
            by_tag[tag]["cost"] += cost
        total_calls += 1
        total_cost += cost

    print(f"bentocall usage — {label}")
    print(f"  total: {total_calls} calls, ${total_cost:.4f}")
    for m, d in sorted(by_model.items()):
        print(f"  {m}: {d['calls']} calls, ${d['cost']:.4f} "
              f"(in={d['in_tok']:,} out={d['out_tok']:,})")
    if args.by_tag:
        print("  by tag:")
        for tag, d in sorted(by_tag.items(), key=lambda x: -x[1]["cost"]):
            print(f"    {tag:<10} {d['calls']} calls, ${d['cost']:.4f}")
    if args.savings:
        delta = total_sonnet_equiv - total_cost
        pct = (delta / total_sonnet_equiv * 100) if total_sonnet_equiv > 0 else 0.0
        print()
        print(f"  savings vs all-Sonnet counterfactual:")
        print(f"    actual:           ${total_cost:.4f}")
        print(f"    sonnet-equiv:     ${total_sonnet_equiv:.4f}")
        print(f"    saved:            ${delta:.4f}  ({pct:+.1f}%)")
        print()
        print("  note: per-call substitution counterfactual (same prompt,")
        print("  Sonnet model). Not a flat-vs-recursion comparison.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
