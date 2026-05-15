# Changelog

All notable changes to depth-lens will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — v1.0 milestones in progress
- `parallel_map` thread-pool helper in `depth_lens/adapters/_concurrency.py`.
- API adapters (Anthropic / OpenAI / Gemini) now accept a `max_concurrent`
  kwarg and fan requests across a thread pool. A 1000-prompt probe drops
  from sequential hours to minutes.
- `CONTRIBUTING.md` covering how to add a Task and how to add an Adapter,
  with the conventions used in the bundled implementations.
- Standalone `depth-lens:gpu` and `depth-lens:api` Docker images
  (multi-stage Dockerfile); no longer piggy-backs on the OpenMythos image.
- **`CustomTask`**: bring your own JSONL of `{prompt, target, depth?}` and
  probe it with any adapter. Pluggable scorers: `exact`, `first_int`,
  `last_int`, `yes_no`, `contains`, `regex:<pattern>`. Addressable as
  `custom:<path>:<scorer>` everywhere. This is the headline practical
  feature for API users tuning thinking budgets on their own workload.
- **Cost & latency tracking**: `probe()` now records per-cell median wall-clock
  latency and aggregates `input` / `output` / `thinking` token usage from
  adapter metadata. `ProbeResult.cost_per_cell(pricing)` returns $/prediction.
- README rewritten with sharper positioning: depth-lens is *the* tool that
  puts looped transformers + Claude/o3/Gemini extended thinking on the same
  axis, with your own data. Explicit comparison table vs LLMThinkBench,
  usail-hkust benchmark, and o1 scaling laws.

## [0.5.0] — 2026-05-15

First public-facing release. Adds API adapters, more tasks, statistical
rigor, persistence, and an interactive dashboard.

### Added
- **Adapters**: `anthropic:<model>` (Claude extended thinking),
  `openai:<model>` (o-series reasoning_effort), `gemini:<model>`
  (Gemini 2.5 thinking_budget), `vllm:<model>` (OpenAI-compatible local
  server like vLLM / SGLang / TGI).
- **Tasks**: `parity` (XOR over n bits), `graph-reach` (multi-hop
  reachability with balanced positives/negatives), `state-tracking`
  (two-counter register machine with swap and add).
- **Wilson 95% CIs** computed for every probe cell and drawn as bands
  on accuracy curves.
- **Probe result cache** on disk, JSON, keyed by inputs — repeated
  `depth-lens probe ...` calls return instantly.
- **`depth-lens dashboard`**: Streamlit explorer over the cache with
  per-probe drill-downs, curves, heatmaps, and overthinking reports.
- **`depth-lens compare --models ...`**: multi-model overlay plots,
  one panel per task depth.
- `Task.vocab_seed()` API so tasks with predictable target ranges can
  declare canonical tokens; the OpenMythos adapter consults it when
  building its vocab.

### Changed
- README rewritten with the full v0.5 surface area and quick-start
  examples per adapter family.
- ROADMAP updated to reflect v0.5 mid-point findings (qualitatively
  different compute-scaling profiles across four tasks).
- OpenMythos adapter's training helper now uses a larger
  `max_seq_len` (auto-sized to handle depth extrapolation).

### Empirical findings from this release
- K-hop: heavy overthinking; peak at training depth (loops=4), decay
  thereafter.
- Parity: clean monotonic improvement up to training depth, then flat.
- Graph-reach: zero compute benefit; model saturates at ~0.70 heuristic.
- State-tracking: overthinking like K-hop (peak at training depth, loops=4)
  but **best extrapolation** of the four tasks — K=8 still 0.61 at peak
  compute, suggesting the vector-valued state forces compositional learning
  rather than memorisation.

## [0.1.0] — 2026-05-15

Initial MVP. Single-day shipped scope.

### Added
- `Task` ABC and `KHopTask` (modular composition on Z/23Z).
- `ModelAdapter` ABC and `OpenMythosAdapter` (n_loops as compute knob),
  plus a `train_for_task()` helper.
- `HuggingFaceAdapter` (`hf:<model-id>`) with max_thinking_tokens.
- `probe()` core: depth × compute sweep returning `ProbeResult` with
  `effective_depth()` and `overthinking()` diagnostics.
- `depth-lens probe` and `depth-lens compare` CLI subcommands.
- Static matplotlib plots: accuracy curves, heatmaps, multi-model overlays.
- `examples/quickstart.ipynb`.
