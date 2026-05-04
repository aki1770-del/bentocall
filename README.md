# bentocall

> Recursive long-context LLM calls, in lunchbox shape.

A bento box partitions a meal into compartments. A bigger box can nest a smaller one inside it, and a smaller compartment can be carved out of a bigger space. **bentocall** does the same thing to long-context LLM tasks: split a long document into smaller compartments, run a cheap-and-fast model on each, fold the results back together with deterministic Python. When you'd otherwise pay for one big frontier-model call that drifts on long context, you instead pay for many small specialist calls that don't.

This is a working implementation of the **Lambda-RLM** algorithm ([arXiv:2603.20105](https://arxiv.org/abs/2603.20105)) wrapped around the OpenRouter API, with task adapters for two extract-and-aggregate shapes that come up a lot.

## What it actually saves you

Empirically, on a 20-task batch validation (2026-05-03):

| Path | Capability hold (≥0.95 / exact) | Total cost | Δ vs Sonnet flat |
|---|---|---|---|
| Sonnet 4.6 flat baseline | 17/20 | $0.94 | — |
| **bentocall** (Haiku λ-RLM + auto-routed flat fallback) | **20/20** | **$0.50** | **−47%** |

Two task shapes shipped:
- **`ool_pairs`** — pairwise relation extraction over labelled records (e.g. "users who both posted in target categories"). Routes to Haiku-recursion at ≥4K tokens.
- **`aggregate_counts`** — per-key counts across a long document. Routes to Sonnet-recursion at ≥8K tokens (Haiku has ±1 counting drift on structured records, Sonnet doesn't).

Below the per-task threshold, bentocall transparently calls Sonnet 4.6 flat — same answer, cheaper than recursion overhead.

## Install

```sh
pip install bentocall
export OPENROUTER_API_KEY=sk-or-v1-...
```

### Local-inference backend (v0.1.1+)

Point bentocall at any OpenAI-compatible local server (llama.cpp `llama-server`, vLLM, LM Studio, …) instead of OpenRouter:

```sh
# example: llama.cpp serving Hermes-3-Llama-3.1-8B at port 8765
export BENTOCALL_BASE_URL="http://127.0.0.1:8765/v1/chat/completions"
export BENTOCALL_MODEL_HAIKU="hermes-3-llama-3.1-8b"
# OPENROUTER_API_KEY is optional when using a local server
unset OPENROUTER_API_KEY

bentocall --task ool_pairs --self-test
```

Validated on `ool_pairs`: Hermes-3-8B Q4 hits 10/10 perfect F1, identical to cloud Haiku. **Not** validated for `aggregate_counts` — small local models drift ~4× more than Haiku on counting; route those to cloud. See `bentocall.research.drift` for measuring this on your own setup.

## Use

CLI:
```sh
echo "User 1: \"What does NASA stand for?\" [label: abbreviation]
User 2: \"What is freedom?\" [label: description and abstract concept]
User 1: \"What is justice?\" [label: description and abstract concept]" | \
  bentocall --task ool_pairs --target "abbreviation,description and abstract concept"
```

Or as a library:
```python
from bentocall import solve

result = solve(open("long_doc.txt").read(), task="ool_pairs")
print(result["answer"])              # → [[1, 2], ...]
print(result["routing"])             # → "lambda-rlm" or "flat-sonnet"
print(result["trace"]["leaves"])     # → number of LLM calls made
```

Self-test the install:
```sh
bentocall --task ool_pairs --self-test         # canned 30-item, expects F1=1.0
bentocall --task aggregate_counts --self-test  # canned 60-item, expects rel_err < 0.05
```

Watch your spend:
```sh
bentocall-usage              # today
bentocall-usage --week --savings   # 7-day with all-Sonnet counterfactual
```

## ⚠ The thresholds you're inheriting are NOT yours

The `ROUTE_THRESHOLDS = {"ool_pairs": 4000, "aggregate_counts": 8000}` defaults were measured on **one specific workload distribution** at OpenRouter pricing on 2026-05-03. Your task shapes, your input sizes, and current model pricing all shift the cost-inversion point. **Re-derive on your data before relying on these in production:**

```sh
# Run the autoresearch sweep on your workload (~$3, ~30 min)
python -m bentocall.research.sweep
# Inspect the per-cell winner table
cat runs/frontier.json
# Update bentocall/api.py: ROUTE_THRESHOLDS = {...}
```

If you skip this step and your task mix doesn't look like ours, you may save 0% or even pay 17% more. The 47% number is the *upper bound on a specific synthetic*; treat it as "the algorithm works" rather than "this is your savings."

## Add a new task adapter

Two of the six task shapes from the original paper are implemented. To add a third (`search`, `multi_hop`, `summarise`, `s_niah`):

1. Read `bentocall/tasks/aggregate_counts.py` end to end — it's the template.
2. New file `bentocall/tasks/<your_task>.py` exposing exactly: `generate(...)`, `lambda_rlm(case, model, K)`, `flat_baseline(case, model)`, `score(pred, gold)`.
3. Register in `bentocall/api.py`: add to `SUPPORTED_TASKS`, dispatch in `solve()`, add a `ROUTE_THRESHOLDS` entry (set conservatively until you've measured).
4. Add a CLI choice in `bentocall/cli.py` and a self-test branch.

PRs that follow this template will be reviewed quickly. PRs that don't will get a one-line redirect.

## Maintenance posture

This is a **reference implementation**, maintained by one person on weekends. Issues and PRs get best-effort responses, no SLA. Adapter PRs that follow the template in `CONTRIBUTING.md` get merged fast; everything else may not.

If you depend on this in production, **fork it.** That's the right relationship for a reference impl, and you'll thank yourself when an upstream change you didn't want lands at a bad time.

For questions and "how do I…" use [Discussions](https://github.com/aki1770-del/bentocall/discussions). Issues are reserved for reproducible bugs and security reports.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Acknowledgements

The recursion shape comes from the **Lambda-RLM** paper (Roy et al., 2026, [arXiv:2603.20105](https://arxiv.org/abs/2603.20105)). bentocall is one cloud-backed implementation of that algorithm, plus task adapters and routing logic that are this project's own work.
