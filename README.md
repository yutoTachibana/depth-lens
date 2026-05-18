# depth-lens

> **Pick the cheapest LLM config that meets your accuracy bar — for your data, not somebody else's benchmark.**
>
> Sweep every (model, knob) on your prompts in one CLI call. Wilson 95% CIs, per-call cost, latency p50, cross-vendor. Roughly **$1 and ten minutes** per audit.
>
> [日本語版](./README.ja.md)

[![tests](https://github.com/yutoTachibana/depth-lens/actions/workflows/test.yml/badge.svg)](https://github.com/yutoTachibana/depth-lens/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Status: v2.1](https://img.shields.io/badge/status-v2.1-green.svg)](#status)

![Switching from Opus 4.7 to Haiku 4.5 saves $123k/year on a 10k-calls/day workload — same accuracy](docs/findings/figures/hero-cost-savings.png)

**The plot above is a real `depth-lens recommend` output.** Four Anthropic configurations *all* score 1.00 accuracy on a K-hop tier-4 prompt set — and span **~35× in cost**. That's the gap the *"use the latest / biggest"* instinct burns through silently. depth-lens finds the cheapest passing tier on **your** prompts in under 10 minutes. The 30-second install is right below.

## 30 seconds: install, run, decide

```bash
git clone https://github.com/yutoTachibana/depth-lens.git
cd depth-lens
pip install -e .[anthropic,openai,gemini]

export OPENAI_API_KEY=...     # plus ANTHROPIC_API_KEY / GOOGLE_API_KEY as needed

# Your production prompts → one JSONL line each
cat > my_eval.jsonl <<'EOF'
{"prompt": "Compute (47 * 23 + 19) mod 31.", "target": "15", "depth": 1}
{"prompt": "Compute ((11 * 7 - 4) * 3 + 2) mod 41.", "target": "16", "depth": 1}
{"prompt": "Compute (13 * 17 + 8) mod 29.", "target": "26", "depth": 1}
{"prompt": "Compute ((5 * 9 + 7) * 4 - 3) mod 23.", "target": "21", "depth": 1}
{"prompt": "Compute (100 - 7 * 11) mod 19.", "target": "4", "depth": 1}
EOF

depth-lens recommend \
    --models openai:gpt-5-mini,openai:o4-mini \
    --task custom:my_eval.jsonl:first_int \
    --target-accuracy 0.95 \
    --max-latency 3.0 \
    --n-samples 16 \
    --daily-calls 10000
```

```
============================================================================================
Target accuracy ≥ 0.95  ·  Max latency ≤ 3.00s/pred
Probed 6 configurations, 6 passing.
============================================================================================

✅ Passing (cheapest first):
  openai:gpt-5-mini     d=1  effort=low      acc=1.00   $0.354/k-pred   0.45s/pred  ← cheapest
  openai:gpt-5-mini     d=1  effort=medium   acc=1.00   $0.485/k-pred   0.59s/pred
  openai:o4-mini        d=1  effort=low      acc=1.00   $0.736/k-pred   0.29s/pred  ← fastest
  openai:gpt-5-mini     d=1  effort=high     acc=1.00   $0.886/k-pred   0.69s/pred
  openai:o4-mini        d=1  effort=medium   acc=1.00   $1.061/k-pred   0.37s/pred
  openai:o4-mini        d=1  effort=high     acc=1.00   $1.365/k-pred   0.40s/pred

⚡ Cost-vs-speed tradeoff among passing configs:
  Cheapest is 1.5× slower than fastest; fastest costs 2.1× more per call.

============================================================================================
At 10,000 calls/day with the cheapest passing config:
  openai:gpt-5-mini @ effort=low
  → $3.54/day  $1,291/year

  Switching from openai:o4-mini @ effort=high ($13.65/day)
  saves $10.11/day = $3,691/year (74% reduction)
```

You now have a defensible answer to *"do we really need the bigger / more-thinking config?"* — backed by a real sweep with Wilson 95% CIs on **your** prompts. Swap `--models` for Anthropic / Gemini / vLLM (self-hosted) and re-run; the workflow is identical.

## Evidence: three real business tasks, three measurements

We ran depth-lens end-to-end on three production-style chatbot tasks that map to three different scoring needs. Same `recommend` workflow, three different scorers.

### Case 1 — Tenant-inquiry urgency classifier (real-estate management)

Classify tenant messages into `緊急 / 通常 / 翌営業日` (urgent / business hours / next business day). 20 realistic prompts: water leaks, gas leaks, lockouts, contract questions, noise complaints.

| Config | Accuracy | Latency p50 |
|---|---:|---:|
| **`openai:o4-mini @ effort=medium`** ← chosen | **100%** | **0.52 s** |
| `openai:gpt-5-mini @ effort=low` | 95% | 0.67 s |
| `openai:o4-mini @ effort=high` (default "safe") | 100% | 0.74 s |

**Cost reduction: ~88%** vs defaulting to `o4-mini @ high` or `gpt-5`.
**Domain insight depth-lens surfaced**: the 95% config's single miss was 通常 → 翌営業日 (safe direction). No 緊急 → 通常 errors — accuracy alone undersells the cheaper config's safety profile.

### Case 2 — System-monitoring quote estimator (MSP / IT ops)

Compute monthly quote estimates from free-form Japanese inquiries (plan tier × server count × options × volume discount). **53 prompts across 5 difficulty tiers** including typos, formal/casual mixed, implicit tier hints like "ミッションクリティカル" → premium.

| Config | All 5 tiers acc | Latency p50 |
|---|---:|---:|
| **`openai:gpt-5-mini @ effort=low`** ← chosen | **100% (53/53)** | **0.41 – 0.50 s** |
| `openai:o4-mini @ effort=medium` | 100% | 0.65 s |
| `openai:o4-mini @ effort=high` (default "safe") | 100% | 0.70 s |

**Cost reduction: ~88%** vs the "complex calculation needs a more capable model" intuition.
**Counter-intuitive finding**: multi-step pricing math + production-realistic messy input both solved by the cheapest config. `gpt-5-mini @ low` handles compound discount logic, mixed plans, AND colloquial Japanese ("がっつり監視で") at 100%.

### Case 3 — Tenant-reply quality, judged by LLM (v2.1)

Same property-management chatbot, but **generating free-form replies** to tenant inquiries. Quality judged by a separate LLM against a 3-criterion rubric (polite using 敬語, addresses the specific issue, proposes a concrete next step). 12 prompts spanning urgent / procedural / rules / complaints / repairs.

| Config | All 3 criteria met | Per-reply latency |
|---|---:|---:|
| **`openai:gpt-5-mini @ effort=low`** ← chosen | **100% (12/12)** | **1.4 s** |
| `openai:o4-mini @ effort=high` | 100% (12/12) | 1.8 s |
| `openai:gpt-5-mini @ effort=high` (default "safe") | 75% (9/12) | **15.6 s** ← unusable |

**Counter-intuitive**: `gpt-5-mini` accuracy *decreases* with higher effort (low 100% → high 75%) — over-elaboration breaks the rubric's "addresses the specific issue / concrete next step" criteria. `o4-mini` shows the *opposite* curve. **Optimal effort is per-(model, task), not universal.**
**Why this case matters**: until v2.1, depth-lens couldn't measure free-form replies — only structured tasks like Cases 1 and 2. The new [`llm:` scorer](#can-i-measure-my-task-three-scorer-families) made this measurable.

[Full case study →](docs/findings/v2.1-llm-judge-case-study.md)

### Five patterns these three cases collectively show

1. **"Use the bigger model to be safe" is a strict loss** when measured — same accuracy, more cost, no latency budget gained. Case 3 sharpens this: higher effort can *decrease* accuracy on free-form tasks where over-elaboration hurts the rubric.
2. **Stratified bench (simple → production-messy) reveals where each tier breaks** — or, as in Case 2, that none of the candidates do.
3. **~80-90% cost reduction is typical** when teams stop pre-judging model selection and run a quick depth-lens sweep instead.
4. **Production-realistic input must be in the bench from day 1.** Synthetic tier-1 prompts alone systematically over-recommend expensive models — Case 2's 30 messy "real-log-style" prompts were what produced the conclusion's confidence interval.
5. **The right scorer matters more than the model choice.** The three cases cover the three scorer families depth-lens ships (structured / regex / LLM-as-judge). New production tasks land in one of these buckets.

## Can I measure my task? Three scorer families

| Family | Spec form | When to use |
|---|---|---|
| **Structured** | `exact`, `first_int`, `last_int`, `yes_no`, `contains` | Classification, numeric answers, yes/no decisions. Cases 1 and 2 above. |
| **Regex** | `regex:<pattern>` | Format-checking, "answer must match this shape". |
| **LLM-as-judge** | `llm:<judge-model>:<criterion>` or `llm:<judge-model>:rubric:<text>` | Open-ended outputs: summaries, free-form Q&A, multi-criterion checks. Case 3 above. |

Built-in criteria for `llm:`: `correct` / `faithful` / `helpful` / `concise` / `format` / `polite`. Free-form rubrics are arbitrary text after `:rubric:`.

```bash
# LLM-judge example: grade summary faithfulness with gpt-5-mini
depth-lens recommend \
    --models openai:gpt-5-mini,openai:o4-mini \
    --task "custom:./summaries.jsonl:llm:openai:gpt-5-mini:faithful" \
    --target-accuracy 0.85 --n-samples 32
```

Pick a different (and ideally cheaper) judge than the model under test to avoid self-judging bias. As of 2026, gemini-3.1-flash-lite is the cheapest competent judge.

Between these three families, **almost every production AI task is measurable** — classification, structured extraction, RAG-faithfulness, customer-support reply quality, code review, tone checks. If your task doesn't fit, file an issue.

## What's in the box

### 6 adapter families

| Spec | Compute knob | Cost basis |
|---|---|---|
| `anthropic:<model>` | `thinking_budget_tokens` | API ($/M-token) |
| `openai:<model>` | `reasoning_effort` | API ($/M-token) |
| `gemini:<model>` | `thinking_budget_tokens` (2.5) / auto-mapped to `thinking_level` (3.x) | API ($/M-token) |
| `vllm:<model>` | `reasoning_effort` (thinking models) or `max_tokens` (instruct-only, OpenAI-compatible local server) | self-hosted ($/GPU-hour) |
| `hf:<hf-model-id>` | `max_thinking_tokens` (CoT length) | local GPU ($/GPU-hour) |
| `openmythos` | `n_loops` (Recurrent-Depth Transformer) | local GPU ($/GPU-hour) |

API adapters fan requests through a thread pool (`max_concurrent`); a 1000-prompt probe finishes in minutes, not hours.

### 5 built-in probe tasks + custom

| Task | Depth axis | Reasoning shape |
|---|---|---|
| `k-hop` | K (operators) | Forward composition (mod-arithmetic) |
| `parity` | n (bits) | Aggregation (XOR reduction) |
| `graph-reach` | path length | Single BFS pass |
| `state-tracking` | K (instructions) | 2-counter register machine |
| `mini-csp` | n (variables) | **Search / constraint propagation (2-SAT)** |
| `dict-lookup` | n (pairs) | **Field extraction from structured input** (v2.0) |
| `custom:<jsonl>:<scorer>` | optional `depth` field | **Bring your own data** |

### Diagnostics every `ProbeResult` exposes

- `.accuracy` — `[depth][compute]` grid in `[0, 1]`
- `.ci()` — Wilson 95% intervals on every cell
- `.effective_depth(threshold=0.5)` — biggest depth where some compute level clears the bar
- `.overthinking(depth, tolerance=0.02)` — peak compute is not max compute, by how much
- `.cost_per_cell(pricing)` — $/prediction. Token-based (`{input, output}` USD-per-1M) or GPU-hour (`{gpu_hourly, gpus}`) — pick whichever fits the adapter

## CLI

```bash
depth-lens recommend ... # find cheapest model meeting your accuracy bar (production workflow)
depth-lens probe ...     # detailed sweep of one model
depth-lens compare ...   # overlay several models on the same task
depth-lens dashboard     # Streamlit UI over your cached probes
```

Each subcommand has full `--help`. See [`docs/playbook/`](docs/playbook/) for end-to-end production scenarios:
[model-downgrade](docs/playbook/model-downgrade.md) · [cost-audit](docs/playbook/cost-audit.md) · [regression-detection](docs/playbook/regression-detection.md) · [self-hosting-with-vllm](docs/playbook/self-hosting-with-vllm.md).

## Python API

```python
from depth_lens import probe
from depth_lens.tasks import get_task
from depth_lens.adapters.anthropic_adapter import AnthropicAdapter

task = get_task("mini-csp")
adapter = AnthropicAdapter(model="claude-haiku-4-5", task_name="mini-csp")
result = probe(adapter, task, depths=[3, 5, 7, 9], n_samples=16)

print(f"effective depth: {result.effective_depth(0.5)}")
print(f"overthinking @ d=9: {result.overthinking(9)}")
print(f"$/pred @ d=9 mid budget: {result.cost_per_cell({'input': 1.0, 'output': 5.0})[3, 1]}")
```

## What depth-lens is NOT

- We **don't** run [MMLU](https://github.com/openai/simple-evals), [GSM8K](https://github.com/openai/grade-school-math), or similar leaderboards. Those crown frontier models on canonical benchmarks; production teams already picked a model family and need to tune *within* it.
- We **don't** test "is the model smart." We test "which configuration of *this* family meets *your* accuracy bar at the lowest cost / latency / GPU-time."
- We **don't** ship a managed dashboard. The OSS produces JSONs and plots locally; building hosted dashboards on top of those is outside scope.

| Capability | LLMThinkBench | usail-hkust bench | o1 scaling laws | **depth-lens** |
|---|---|---|---|---|
| Compute-axis curves (not single point) | ❌ | partial | ✅ (o1 only) | **✅** |
| Cross-vendor (Claude / o-series / Gemini / OSS) | ❌ HF only | partial | ❌ o1 only | **✅** |
| Self-hosted vLLM on same axis as APIs | ❌ | ❌ | ❌ | **✅** |
| Looped transformer (OpenMythos) | ❌ | ❌ | ❌ | **✅** |
| Bring-your-own JSONL | ❌ | ❌ | ❌ | **✅** |
| **LLM-as-judge scorer for open-ended tasks** | ❌ | ❌ | ❌ | **✅** |
| Cost per prediction with sweep | ❌ | ❌ | ❌ | **✅** |

Closest active competitor is [LLMThinkBench](https://github.com/ctrl-gaurav/LLMThinkBench), which targets math-task overthinking on HuggingFace models at a fixed operating point — orthogonal to depth-lens's compute-axis sweep across vendor APIs.

## Use cases depth-lens is built for

| You are asking… | What `depth-lens recommend` outputs | Headline evidence |
|---|---|---|
| **1. Which API tier / thinking budget should I be paying for?** | Cheapest passing (model, knob) across your prompts | Opus 4.7 → Haiku 4.5 saves **~$123k/year** at 10k call/day, same accuracy ([finding](docs/findings/v1.0-cost-savings.md)) |
| **2. Should I self-host an open model instead of paying the API?** | API and vLLM points on one Pareto ($/M-token vs $/GPU-hour, same axis) | `gemini-3.1-flash-lite` beats every 4080 SUPER self-hosted candidate at K-hop tier 4; Llama-3-8B AWQ is **cheapest** at tier 1 ($0.028/1k calls) ([finding](docs/findings/v1.2-self-hosted-vs-api.md)) |
| **3. Can I measure free-form output quality?** | LLM-judge scores with the same Wilson CIs as structured scorers | Case 3 above — `gpt-5-mini @ low` wins on a 3-criterion rubric; higher effort *decreases* quality ([finding](docs/findings/v2.1-llm-judge-case-study.md)) |

For research-oriented use (paradigm scaling, inference-time-compute measurement infrastructure), see the [v2.0 cross-paradigm measurement plot](docs/findings/v2.0-scaling-law.md) — a tool for putting Token-CoT API · Self-hosted vLLM · Looped transformers on a single FLOPs axis. We emphasize this is the *measurement tool* contribution; the underlying observation (specialized model beats generalist on the specific task it was trained for) is deep-learning textbook material.

## All findings the tool has produced

We ran depth-lens on every vendor we could get an API key for, on all bundled tasks — current generation **and** one generation back to keep the cross-vendor comparison fair. Total spend: **~$14 API + ~30 min local GPU + ~$1 LLM-judge** (case study 3).

| Use case | Finding | Why it matters |
|---|---|---|
| API ops | [Opus 4.7 → Haiku 4.5 saves ~$123k/year on a 10k-call/day task](docs/findings/v1.0-cost-savings.md) | 4 concrete tier-downgrade savings switches in $ |
| API ops | [gpt-5-mini cheaper-per-token but 3× slower than o4-mini](docs/findings/v1.0-cost-savings.md#cost-is-one-axis--latency-is-another) | $/token alone burns UI latency; Pareto frontier on K-hop tier 4 has 2 points |
| API ops | [Haiku 4.5 collapses on hard 2-SAT at default budget](docs/findings/v1.0-mini-csp-cross-vendor.md) | Constraint-style problems need `budget ≥ 4096` or 2× error rate |
| API ops | [Gemini 2.5 Flash uniquely weak vs same-era Anthropic / OpenAI cheap reasoning](docs/findings/v1.0-cross-vendor-summary.md#five-structural-findings-depth-lens-surfaced) | 3.1 Flash-Lite closes the gap |
| API ops | [Claude Opus 4.7 cost varies 10× across (depth × budget) at fixed accuracy](docs/findings/v1.0-anthropic-cross-vendor.md) | Maxing the budget is a strict cost loss |
| API ops | [Per-vendor cost-vs-latency plots](docs/findings/v1.1-cost-vs-latency-per-vendor.md) | One scatter per vendor — Pareto frontier vs budget knobs |
| Build vs buy | [Self-hosted vLLM vs hosted APIs on one Pareto](docs/findings/v1.2-self-hosted-vs-api.md) | Llama-3-8B AWQ is **cheapest** at tier 1; **0% acc** at tier 4. DeepSeek-R1-Distill-1.5B hits 0.75 at tier 4. Build-vs-buy as a chart, not a guess |
| Open-ended | [Customer-reply quality via LLM-as-judge (Case 3)](docs/findings/v2.1-llm-judge-case-study.md) | `gpt-5-mini` accuracy decreases with effort on free-form tasks; optimal effort is per-(model, task) |
| Research / tool | [v2.0 — 3 inference-time-compute paradigms on one FLOPs axis](docs/findings/v2.0-scaling-law.md) | Infrastructure to compare Token-CoT API · Self-hosted vLLM · Looped (OpenMythos 1M/10M/100M) on the same axis. The headline 24,000-410,000× FLOPs ratio is a deep-learning-textbook result; the **tool** is the contribution |
| Research | [OpenMythos vs Claude head-to-head](docs/findings/v1.1-architecture-comparison.md) | Within training distribution, 925K-param looped is ~10,000× faster than Claude at same accuracy. Outside it, API dominates |
| Research | [OpenMythos loops-vs-accuracy saturation](docs/findings/v1.1-cost-vs-latency-per-vendor.md#openmythos-looping-pays-latency-but-the-more-loops--more-depth) | "More loops = deeper reasoning" saturates at `training_max_loop_iters` |
| Research | [OpenMythos extrapolates 1-2 hops past training depth on K-hop](docs/findings/v0.5-openmythos.md) | Seed experiment that motivated the project |

**[→ Full v1.0 cross-vendor summary](docs/findings/v1.0-cross-vendor-summary.md)**

## Status

- [x] **v0.1 MVP** — first end-to-end probe (May 2026)
- [x] **v0.5** — 4 tasks, 5 adapters, Wilson CIs, cache, Streamlit dashboard
- [x] **v1.0** — 6 adapter families, 5 tasks, full cross-vendor benchmark (Anthropic/OpenAI/Gemini, current + 2025 prior gen), multi-stage Docker, GitHub Actions CI
- [x] **v1.1** — OpenMythos head-to-head; cross-paradigm Pareto
- [x] **v1.2** — self-hosted vLLM with GPU-hour pricing on the same Pareto
- [x] **v2.0** — 3-paradigm FLOPs measurement tool, `dict-lookup` task, `depth_lens.flops` module
- [x] **v2.1** — LLM-as-judge scorer for open-ended tasks (`llm:<judge>:<criterion>`), tenant-reply case study
- [ ] **v2.2** — PyPI publish, judge cost folded into `recommend` $/k-pred, `--free-form` CLI flag, code-generation task

128 unit tests passing. See [ROADMAP.md](./ROADMAP.md) for what's next.

## Install variants

```bash
# API-only (no GPU needed) — Anthropic, OpenAI, Gemini, dashboard
pip install -e .[anthropic,openai,gemini,dashboard]

# +looped transformer + HuggingFace local probes
pip install -e .[openmythos,huggingface,anthropic,openai,gemini,dashboard]

# +self-hosted vLLM (vLLM runs separately via docker compose)
pip install -e .[anthropic,openai,gemini,dashboard]   # OpenAI SDK is all that's needed client-side

# Just the framework (BYO adapters)
pip install -e .
```

Python 3.11+. The bundled OpenMythos training helper assumes CUDA; everything else is happy on CPU or against remote APIs.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for how to add a Task or an Adapter (both are ~50 lines + a test) and the conventions used in the bundled implementations.

## Citation

```bibtex
@software{depth_lens_2026,
  title  = {depth-lens: Measuring Inference-Time Compute for LLM Production Decisions},
  author = {yutoTachibana},
  year   = {2026},
  url    = {https://github.com/yutoTachibana/depth-lens}
}
```

## License

[MIT](./LICENSE).
