# Changelog

## 0.1.1 — 2026-05-04

### Added
- **Local-server backend** via `BENTOCALL_BASE_URL` env var. Point at any
  OpenAI-compatible endpoint (llama.cpp `llama-server`, vLLM, LM Studio, etc.)
  to run leaves on local hardware instead of OpenRouter.
- `BENTOCALL_MODEL_HAIKU` and `BENTOCALL_MODEL_SONNET` env vars to remap the
  internal aliases to whatever model name the local server uses.
- `bentocall.research.drift` — multi-seed drift measurement module for
  comparing model accuracy distribution (mean / median / p90 / max). Run
  `python -m bentocall.research.drift --task <task> --seeds 10 --model haiku`.

### Changed
- `OPENROUTER_API_KEY` is now **optional** when `BENTOCALL_BASE_URL` points
  at a non-OpenRouter URL. The cloud path (default URL) still requires it.
- The `Authorization` header is omitted from local-server requests.

### Empirical findings (Hermes-3-Llama-3.1-8B Q4 vs cloud Haiku 4.5)

10-seed drift sweep (see `bentocall.research.drift`):
- `ool_pairs` (set/pair extraction): both models 10/10 perfect F1=1.0.
  Hermes-3-8B is a **true drop-in for Haiku** on this task shape.
- `aggregate_counts` (exact counts): Hermes-3-8B drifts ~4× more than Haiku
  (mean 11.3% vs 3.0%, p90 21% vs 5%). **Don't route counting workloads to
  small local models.** Cloud Sonnet 4.6 remains the only model that holds
  exact_match reliably for this shape.

### Routing recommendation update

```
ool_pairs         ≥4K tokens → local Hermes-3-8B (or any solid 7-8B)
aggregate_counts  any size   → cloud Sonnet flat (do NOT route to local)
```

## 0.1.0 — 2026-05-03

Initial release. Lambda-RLM cloud-port with two task adapters
(`ool_pairs`, `aggregate_counts`), auto-routing between recursion and
flat-Sonnet baseline based on per-task token thresholds, validated 47%
saved vs Sonnet flat on a 20-task batch.
