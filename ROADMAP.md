# depth-lens — Roadmap

> A measurement and visualization toolkit for *reasoning depth* across model families.
> Started 2026-05-15 in conversation with Claude (Opus 4.7), after running the
> OpenMythos depth-extrapolation experiment in `/c/Users/burger/claude/openmythos`.

---

## Vision (one paragraph)

Modern reasoning models — looped transformers (OpenMythos, Parcae), extended-thinking
APIs (Claude, o-series, Gemini), agentic loops — all share a hidden axis: how much
inference-time compute do they spend, and what does that buy them? Standard
benchmarks (MMLU, GSM8K) collapse this axis. **depth-lens makes that axis legible.**
Given any reasoning system, it produces an accuracy-vs-compute curve, detects
overthinking, estimates effective reasoning depth, and lets practitioners compare
models on a depth-aware basis.

---

## North Star (success looks like)

A developer / researcher can:

```bash
$ depth-lens probe --model claude-opus-4-7-thinking --task k-hop --K-max 12
[task] k-hop modular composition  K_test=2..12
[model] claude-opus-4-7 (thinking_budget grid: 1k/4k/16k tokens)
  → effective_depth = 7.2 hops
  → overthinking_threshold = 12k tokens (accuracy peaks then drops)
  → cost_optimal_budget = 4k tokens
Plot saved: probe.png
```

…across a half-dozen tasks and a half-dozen model families, with curves
overlaid on the same axes.

---

## Empirical findings that motivated this project

From the 2026-05-15 OpenMythos experiment (see `/openmythos/results/`):

1. **Looped transformer extrapolates 1–2 hops beyond K_train**, baseline does not.
2. **"More loops = deeper reasoning" is false through the standard ACT-weighted
   readout** (overthinking dominates), but **true through the raw recurrent state**.
   → ACT halting becomes the bottleneck on OOD depths.
3. **Effective reasoning depth ≠ max loops × hops per loop**. There is a real,
   measurable saturation point.

These three observations are exactly the kinds of facts depth-lens should
automate the discovery of, for any model.

---

## Scope: what depth-lens IS and ISN'T

| IS | ISN'T |
|---|---|
| A measurement framework that produces accuracy-vs-compute curves | A model zoo |
| A library of bounded-depth reasoning probe tasks | A general LLM benchmark (HELM/lm-eval scope) |
| Adapters into common model APIs and architectures | A training framework |
| Visualization + diagnostic reports | A serving system |
| Comparable across model families | A leaderboard |

---

## Phase Roadmap

### v0.1 — MVP ✅ COMPLETE (2026-05-15, ¥0)

**Goal**: end-to-end probe of OpenMythos + 1 HuggingFace model on K-hop, with static plot.

- [x] Repo skeleton, pyproject.toml, MIT license, .gitignore, README
- [x] `depth_lens.tasks.Task` ABC + scoring contract
- [x] `k_hop` task (port from `openmythos/experiments/depth_extrapolation.py`)
- [x] `depth_lens.adapters.ModelAdapter` ABC with `compute_knob` parameter
- [x] `openmythos` adapter (n_loops as knob) + bundled `train_for_task` helper
- [x] `hf:<model-id>` adapter (max_thinking_tokens as knob)
- [x] `probe()` core: sweep compute × tasks, return curves
- [x] Static matplotlib plot: per-model curve, heatmap, and overlay (compare)
- [x] `depth-lens probe` and `depth-lens compare` CLI
- [x] Quick-start notebook (`examples/quickstart.ipynb`) + README example
- [x] Smoke test: full openmythos training + Qwen2.5-1.5B-Instruct comparison ran end-to-end on RTX 4080 SUPER in ~12 minutes total

**v0.1 release headline finding**: a 925K-parameter task-trained OpenMythos
extrapolated to effective_depth=6, beating a parameter-matched non-looped
baseline and a 1.5B general-purpose instruct LM (effective_depth=4) on the
same K-hop task. Overthinking was detected at every depth via the bundled
detector — peak n_loops always matched training depth.

### v0.5 — public release (in progress, +6 weeks target)

- [x] Adapter: `anthropic` (Claude with extended thinking, thinking_budget grid)
- [x] Adapter: `openai` (o-series effort levels)
- [x] Adapter: `gemini` (thinking mode)
- [x] Adapter: `vllm` / local-server backend (OpenAI-compatible)
- [x] Tasks: `parity`, `graph_reach`, `state_tracking` (+3). Still planned: `mini_csp`.
- [x] Wilson 95% CIs on accuracy estimates (binomial intervals)
- [x] Overthinking detector (peak vs last accuracy on the compute axis)
- [x] Effective-depth estimator (largest depth clearing an accuracy threshold)
- [x] Streamlit interactive dashboard (`depth-lens dashboard`)
- [x] Cache layer (probe results JSON-cached, resumable runs)
- [x] Blog post draft (`docs/blog/intro.md`)
- [x] PyPI release prep (version 0.5.0, CHANGELOG, MANIFEST) — actual `twine upload` deferred to release day
- [ ] Reproducible cross-vendor benchmark figure for the README (needs API budget)

**v0.5 findings** (4 tasks × OpenMythos adapter, 2026-05-15):
- K-hop: heavy overthinking — peak at training depth (loops=4), degrades after.
- Parity: monotonic improvement up to loops=4, then flat — no overthinking.
- Graph-reach: loops produce no improvement; model saturates at ~0.70 heuristic.
- State-tracking: overthinking like K-hop, but the vector-valued state forces compositional learning — best extrapolation of the four (K=8 still 0.61 at peak compute).

These four qualitatively different compute-scaling profiles, surfaced automatically by `depth-lens probe`, demonstrate why a depth-aware benchmark exists.

### v1.0 — community-ready ✅ COMPLETE (2026-05-16)

- [x] Batched / concurrent API eval (`parallel_map` thread-pool helper; per-adapter `max_concurrent` kwarg)
- [x] Full cross-vendor benchmark (Anthropic / OpenAI / Gemini, current + 2025 prior gen, 5 tasks each, ~$14 spend)
- [x] Contributor docs (`CONTRIBUTING.md`): how to add a Task and an Adapter
- [x] Docker image (`depth-lens:gpu`, plus an `api` slim-image target)
- [x] Public findings page (`docs/findings/v0.5-openmythos.md` + 5 v1.0 finding docs with committed plots and data)
- [x] Multi-language report (`README.ja.md`, `docs/blog/intro.ja.md`)
- [x] GitHub Actions CI (lint + tests on push)

### v1.1 — architecture comparison (2026-05-17)

- [x] OpenMythos vs Claude head-to-head on built-in K-hop ([finding](docs/findings/v1.1-architecture-comparison.md))
- [x] Per-vendor cost-vs-latency plots (Anthropic / OpenAI / Gemini) ([finding](docs/findings/v1.1-cost-vs-latency-per-vendor.md))
- [x] OpenMythos `n_loops` sweep w/ latency capture — saturation finding measured directly
- [x] README two-layer positioning: production cost CI (surface) + inference-time-compute scaling meter (deep)

### v1.2 — self-hosted vLLM on the same Pareto (2026-05-17)

- [x] vLLM adapter with two compute axes: `reasoning_effort` (thinking models) and `max_tokens` (instruct-only models like Llama-3-8B-Instruct)
- [x] GPU-hour pricing schema (`{gpu_hourly, gpus}`) in `cost_per_cell` — self-hosted and API points land on the same cost axis
- [x] `--gpu-hourly-rate` / `--compute-axis` flags on `recommend` + `probe`
- [x] Docker compose recipes for Llama-3-8B-Instruct AWQ and DeepSeek-R1-Distill-Qwen-1.5B that fit a 16 GB consumer GPU
- [x] 4-way Pareto finding doc ([v1.2-self-hosted-vs-api.md](docs/findings/v1.2-self-hosted-vs-api.md))
- [x] User-perspective audit + first-time-user friction fixes (Windows UTF-8 stdout, click UsageError wrapping, README quickstart targets, plot-script `/work` fallbacks, reproducibility data files force-added)

### v2.0 — "Inference-time compute scaling laws" (planned 1-2 session sprint)

**Strategic pivot**: instead of broadening (more adapters, more API
features), v2.0 focuses on *one shareable artifact* that doubles as
the OSS's recognizable visual: **a 3-paradigm scaling-law plot
that no other tool currently produces.**

Goal: ~50-100 GitHub stars from one blog post + tweet thread that
points at this plot, with depth-lens visibly the toolkit that
produced it.

**Scope:**

- **3 paradigms on one chart**:
  (a) Looped (OpenMythos), (b) Self-hosted vLLM
  (Llama-3-8B-AWQ + DeepSeek-R1-Distill-Qwen-1.5B),
  (c) Token-CoT API (Anthropic Haiku/Sonnet, OpenAI o4-mini/gpt-5-mini,
  Gemini 3.1 Flash-Lite).
- **3 OpenMythos sizes**: 1M / 10M / 100M params
  (925K was the v1.1 baseline; this extends the curve).
- **5 tasks**: K-hop, parity, state-tracking, mini-CSP, NEW
  structured-output.
- **FLOPs axis** (lightweight): accurate for OpenMythos/vLLM (counted
  from architecture); estimated for API (published param × token
  count, with explicit caveat).

**3 deliverable plots:**

1. **Scaling-law plot**: x = log(FLOPs/inference), y = accuracy. One
   curve per paradigm, per task. Shows where each paradigm wins.
2. **Looped size scaling curve**: at what OpenMythos parameter count
   does looped catch a frontier API on each task.
3. **TCO crossover plot**: $/correct-answer × latency, with the
   crossover region (where self-hosted starts beating API) marked.

**Writeup**: blog post + tweet thread + README v2.0 section.
No arXiv unless community response signals it merits one.

**Sessions:**

- **Session 1** (~4-5 hr live): implement FLOPs counters + structured-
  output task; kick off 1M and 10M training across all 5 tasks
  (~4 hr GPU); kick off 100M × K-hop in background; run all API and
  vLLM probes; first-pass plots.
- **Overnight**: 100M × remaining 4 tasks training continues in
  background (~12 hr GPU).
- **Session 2** (~3-4 hr live): collect 100M results; finalize three
  plots; write blog post; update README + ROADMAP; tag v2.0.

**Honest bottleneck**: 100M × 5 tasks = ~15 GPU-hours. Wall-clock
gap between sessions is structural, not laziness.

### v2.1 — LLM-as-judge scorer (in progress, 2026-05-18)

- [x] `llm:<judge-model>:<criterion>` scorer — open-ended task scoring
  via separate judge LLM, with 6 built-in criteria (correct/faithful/
  helpful/concise/format/polite) plus free-form rubric form
- [x] 19 new tests (parser, builtin criteria, rubric form with internal
  colons, score parsing, custom-task integration, regression tests for
  the existing path:scorer parser)
- [x] Smoke-tested on a summary task (o4-mini judged by gpt-5-mini for
  faithfulness; verified that semantically equivalent reword scores 1
  and unrelated text scores 0)
- [ ] Cost accounting: judge usage should be folded into recommend's
  $/k-pred output. Currently judge cost is real but not surfaced in
  the cost projection — track separately for now.
- [ ] Sample case study with an open-ended task that v1.0-v1.2 couldn't
  measure (forthcoming, e.g., customer-support reply quality).

### v2.2+ — backlog (after v2.1 ships and we see community signal)

- [ ] PyPI publish (`pip install depth-lens` from anywhere) — pull
      forward if HN/X traffic warrants it
- [ ] Code-generation task (Python single-function, exec + unit-test
      scoring) — defer from v2.0 due to scorer complexity
- [ ] RAG re-ranking task + scorer
- [ ] Long-form generation evaluation (LLM-as-judge or ROUGE-class)
- [ ] More adapters: AWS Bedrock, Groq, Together, Azure OpenAI,
      Mistral La Plateforme
- [ ] Continuous regression CI: GitHub Action template that re-runs
      `recommend` on a saved JSONL and fails the build when accuracy
      drops or cost rises
- [ ] Production-traffic sampler: collect a fraction of live prompts
      into a rolling bench (with PII redaction helpers)
- [ ] Safe-direction diagnostic: `.misclassification_direction()` on
      ProbeResult, surfacing whether errors trend toward the "safe"
      class (insight surfaced by the v1.2 tenant-urgency case study)
- [ ] arXiv preprint or workshop submission — only if v2.0 traction
      makes it worth the extra ~1-2 weeks of formal writeup

---

## Estimated effort & cost

| Version | Claude-Code session hours | Calendar (focused / weekends) | Cash cost |
|---|---|---|---|
| v0.1 MVP | 20–40h | 2-3 wk / 1-2 mo | **¥0** (local GPU + open weights only) |
| v0.5 public | +30–50h | 1-2 mo / 2-3 mo | ¥5k–30k (Claude/o3 API) |
| v1.0 community | +20–40h | 1-2 mo / 2-3 mo | ¥30k–150k (full benchmark API run) |
| **Total** | **70–130h** | **4-7 months** | **¥35k–180k** |

User's RTX 4080 SUPER 16GB covers all local-model compute. Cash budget is API-only.

---

## Open design questions (decide as we go)

| Question | Default I'll take if unanswered | When it matters |
|---|---|---|
| Package name on PyPI | `depth-lens` | v0.5 release |
| Repo host | github.com/yutoTachibana/depth-lens | v0.1 push |
| Python min version | 3.11 | v0.1 |
| Tooling | `uv` + `pyproject.toml` | v0.1 |
| License | MIT | v0.1 |
| CI | GitHub Actions, lint + smoke test on CPU | v0.1 |
| How to render thinking-budget axis next to n_loops axis when both probe "compute" | normalize to "log compute units" with per-model conversion | v0.5 |
| Whether to bundle small trained OpenMythos checkpoint or download on first use | download from HF Hub on first use | v0.5 |
| Whether to support remote eval workers (probe on cloud GPU) | not in scope for v1.0 | future |

---

## Reference materials

OpenMythos and the looped-transformer literature this builds on:

- OpenMythos repo: https://github.com/kyegomez/OpenMythos
- Saunshi et al. 2025 — Reasoning with Latent Thoughts: https://arxiv.org/abs/2502.17416
- Parcae (Prairie et al. 2026): https://arxiv.org/abs/2604.12946
- Coconut (Continuous Latent CoT, Meta): https://arxiv.org/abs/2412.06769
- Universal Transformers (Dehghani 2018): https://arxiv.org/pdf/1807.03819
- Anthropic extended thinking docs
- OpenAI o-series reasoning effort docs

Empirical seed data: `/c/Users/burger/claude/openmythos/results/` (full JSON dumps + plots from the 2026-05-15 K-hop experiment).
