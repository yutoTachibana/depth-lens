# depth-lens

> Measure and visualize **reasoning depth** across model families.
> [日本語版](./README.ja.md)

Modern reasoning systems — looped transformers (OpenMythos, Parcae), extended-thinking
APIs (Claude, o-series, Gemini), agentic loops — spend variable amounts of inference-time
compute per query. Standard benchmarks like MMLU and GSM8K collapse that axis.
**depth-lens makes it legible.** Point it at any reasoning system and get back
an accuracy-vs-compute curve with confidence intervals, an effective-depth
estimate, and an overthinking detector — comparable across model families.

Pre-alpha. v0.5 in progress. See [ROADMAP.md](./ROADMAP.md) for the plan and
the empirical findings that motivated it.

## What you get

- A **probe engine** that sweeps task depth × compute budget for any wrapped
  model and produces an accuracy curve with Wilson 95% confidence intervals.
- A **library of bounded-depth tasks**: K-hop modular composition, binary
  parity, multi-hop graph reachability — each instance has a controllable
  depth axis so you can ask "how far does this model extrapolate?"
- **Adapters** for OpenMythos (looped transformer), HuggingFace causal LMs
  (CoT-token budget), Anthropic Claude extended thinking, OpenAI o-series
  reasoning effort, and Google Gemini thinking mode.
- **Auto-detected diagnostics**: `effective_depth`, per-depth overthinking
  detection, peak-compute reports.
- An **interactive Streamlit dashboard** to browse all your cached probe runs.
- A **disk cache** so iterating on plots doesn't re-run expensive probes.

## Install

Requires Python 3.11+.

```bash
git clone https://github.com/yutoTachibana/depth-lens.git
cd depth-lens
pip install -e .[openmythos,huggingface,anthropic,openai,gemini,dashboard]
```

Pick only the extras you need. Local probes on OpenMythos / HuggingFace work
with a CUDA GPU; API adapters need the corresponding `*_API_KEY`.

## Quick start

### 1. Probe a single model

Train a tiny OpenMythos on the K-hop task (~7 min on a consumer GPU), then
sweep loop count × task depth:

```bash
depth-lens probe \
    --model openmythos \
    --task k-hop \
    --depths 2,3,4,5,6,7,8,10 \
    --compute 1,2,4,8,16 \
    --train-steps 5000 \
    --save-checkpoint runs/openmythos.pt \
    --plot runs/probe_openmythos.png
```

You'll get a console summary and a curve plot. Example output:

```
effective depth (≥0.5 acc at some compute): 7
overthinking @ depth 4: peak=n_loops=4 (acc=1.00)  →  last=n_loops=16 (acc=0.87)
overthinking @ depth 7: peak=n_loops=4 (acc=0.92)  →  last=n_loops=16 (acc=0.45)
```

The task suite generalises: swap `k-hop` for `parity` or `graph-reach`.

### 2. Compare models on the same task

```bash
depth-lens compare \
    --models openmythos,hf:Qwen/Qwen2.5-1.5B-Instruct,anthropic:claude-opus-4-7 \
    --task k-hop \
    --depths 2,4,6,8 \
    --checkpoint runs/openmythos.pt \
    --plot runs/compare.png
```

Each adapter's native compute knob (n_loops, max_thinking_tokens, thinking
budget, reasoning effort) is plotted on its own axis, with one panel per task
depth so you can see where each model saturates.

### 3. Browse cached probes interactively

```bash
depth-lens dashboard
```

Streamlit picks up every probe you've already cached and lets you filter by
adapter and task, see curves with CIs, heatmaps, and overthinking reports.

## Python API

```python
from depth_lens import probe
from depth_lens.tasks import get_task
from depth_lens.adapters.openmythos_adapter import train_for_task, TrainConfig

task = get_task("parity")
adapter = train_for_task(task, cfg=TrainConfig(steps=2000))

result = probe(adapter, task, depths=[2, 4, 6, 8], n_samples=128)
print(f"effective depth: {result.effective_depth(0.5)}")
print(f"overthinking @ d=8: {result.overthinking(8)}")
print(f"Wilson 95% CIs:\n{result.ci()}")
```

## Built-in adapters

| Spec | Compute knob | Notes |
|---|---|---|
| `openmythos` | `n_loops` | Trains a small model on the task if no checkpoint is supplied. |
| `hf:<hf-model-id>` | `max_thinking_tokens` | Wraps any HF causal LM with a CoT prompt. Uses the model's chat template when available. |
| `anthropic:<model>` | `thinking_budget_tokens` | Claude extended thinking. Requires `ANTHROPIC_API_KEY`. |
| `openai:<model>` | `reasoning_effort` | o-series effort (low / medium / high). Requires `OPENAI_API_KEY`. |
| `gemini:<model>` | `thinking_budget_tokens` | Gemini 2.5 thinking. Requires `GOOGLE_API_KEY`. |
| `vllm:<model>` | `reasoning_effort` | OpenAI-compatible local server (vLLM / SGLang / TGI). Set `VLLM_BASE_URL` (default `http://localhost:8000/v1`). |

## Built-in tasks

| Task | Depth axis | Description |
|---|---|---|
| `k-hop` | K (number of operators) | Modular composition on Z/23Z with additive + multiplicative permutations. Saunshi-style latent-CoT probe. |
| `parity` | n (number of bits) | XOR over a length-n bit string. Classic state-tracking task. |
| `graph-reach` | path length | Yes/no reachability on a small DAG, balanced positives/negatives. |
| `state-tracking` | K (instructions) | Two-counter register machine (inc1, inc2, swap, add) with a final query. Vector-valued state. |

## Empirical motivation

> **See [docs/findings/v0.5-openmythos.md](docs/findings/v0.5-openmythos.md)
> for the published v0.5 probe results across all four tasks, with plots.**

The first probes — OpenMythos on K-hop, parity, graph-reach, and
state-tracking — already show qualitatively different compute-scaling
profiles, surfaced automatically:

- **K-hop**: heavy overthinking. Peak accuracy at training depth, degrades
  with more loops.
- **Parity**: monotonic improvement up to training depth, then flat. No
  overthinking.
- **Graph-reach**: loops produce no improvement; model saturates at ~0.70
  heuristic accuracy and collapses at K_test > K_train_max.

These are exactly the kinds of facts that a depth-aware benchmark should
surface — and that MMLU-style scores cannot.

## Status

- [x] **v0.1 MVP** — K-hop, OpenMythos + HF adapters, static plots, CLI
- [x] **v0.5 (in progress)** — +parity & graph-reach, +Anthropic/OpenAI/Gemini adapters, Wilson CIs, cache, Streamlit dashboard
- [ ] **v1.0** — cross-vendor benchmark, paper, PyPI release

## License

MIT. See [LICENSE](./LICENSE).

## Citation

```bibtex
@software{depth_lens_2026,
  title  = {depth-lens: Measuring Reasoning Depth Across Model Families},
  author = {yutoTachibana},
  year   = {2026},
  url    = {https://github.com/yutoTachibana/depth-lens}
}
```
