# Contributing to bentocall

Thanks for your interest. This is a one-person reference implementation; the contribution surface is intentionally narrow so the project stays maintainable. Please read this whole document before opening a PR.

## What gets merged quickly

**New task adapters that follow the template.** This is the highest-leverage contribution. Each adapter is one self-contained Python file. Follow this checklist exactly:

1. Copy `bentocall/tasks/aggregate_counts.py` as your starting point.
2. Your file must export exactly four public functions:
   - `generate(...) -> YourCase` — synthetic generator with deterministic gold
   - `lambda_rlm(case, model, K) -> (pred, meta)` — recursive solver
   - `flat_baseline(case, model) -> (pred, meta)` — single-call baseline
   - `score(pred, gold) -> dict` — capability metric
3. Register in `bentocall/api.py`:
   - Add to `SUPPORTED_TASKS`
   - Add a `ROUTE_THRESHOLDS` entry (start conservative — `999999` is fine until you've measured)
   - Add `_solve_<task>_lrlm` and `_solve_<task>_flat` dispatch in `solve()`
4. Register in `bentocall/cli.py`:
   - Add to `--task` choices
   - Add a `_self_test_<task>()` branch
5. Self-test must pass: `bentocall --task <yours> --self-test` exits 0.
6. Brief PR description: what shape it solves, what `ROUTE_THRESHOLDS` you'd suggest, and how you measured.

PRs matching this template are usually reviewed within a week.

## What may not get merged

- **Bug fixes for narrow setups** (specific OpenRouter routing configs, alternative quantizations, etc.) — fork and apply your own patch is often the right call.
- **New CLI flags for one-off use cases** — usually better to consume the library API directly.
- **Refactors that change file layout or import paths** — breaks downstream forks. Convince me first via Discussions before opening a refactor PR.
- **CI matrix expansions** (multiple Python versions, multiple OSes) — current matrix is intentionally minimal to keep maintenance bounded.
- **Documentation rewrites** unless they're fixing actual errors.

## Bug reports

Use the bug template. **Required:** reproducer (smallest possible), bentocall version, Python version, OpenRouter model used, sample input that triggers the bug. Reports without these fields will be closed.

## Feature requests

Use the feature template. **Required:** what you tried, what didn't work, what specifically would fix it. "It would be nice if X" without context is closed.

## Questions

Use [GitHub Discussions](https://github.com/aki1770-del/bentocall/discussions), not Issues. Questions in Issues get redirected.

## Local development

```sh
git clone https://github.com/aki1770-del/bentocall.git
cd bentocall
pip install -e ".[dev]"
export OPENROUTER_API_KEY=sk-or-v1-...
bentocall --task ool_pairs --self-test
```

## License

By contributing, you agree your contributions will be licensed under Apache-2.0 (the project license).
