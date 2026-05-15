# depth-lens

> The only tool that compares **looped transformers, Claude extended thinking,
> o-series reasoning, and Gemini thinking modes on the same accuracy-vs-compute
> axis — using your own data.**
>
> [日本語版](./README.ja.md)

Everyone shipping a reasoning model in 2026 exposes a "how hard to think" knob:
Anthropic's `thinking_budget`, OpenAI's `reasoning_effort`, Gemini's
`thinking_budget`, OpenMythos's `n_loops`. Setting that knob right matters for
both accuracy and cost — but standard benchmarks (MMLU, GSM8K) collapse the
axis. The math-focused [LLMThinkBench](https://github.com/ctrl-gaurav/LLMThinkBench)
gives you one number per model at a fixed operating point. **depth-lens
gives you the curve, on your task, across vendors.**

```bash
pip install -e .[anthropic,openai,gemini,huggingface]

# Bring your own JSONL: {"prompt": "...", "target": "...", "depth": 4} per line.
depth-lens probe \
    --model anthropic:claude-opus-4-7 \
    --task custom:./my_eval.jsonl:first_int \
    --compute 1024,2048,4096,8192,16384 \
    --n-samples 64 \
    --plot probe.png
```

```
effective depth (≥0.5 acc at some compute): 6
overthinking @ depth 4: peak=think=4096 (acc=0.94)  →  last=think=16384 (acc=0.71)

cost @ peak compute: $0.018 / prediction
cost @ last compute: $0.092 / prediction  (5.1× cost for −0.23 accuracy)
```

You now have a defensible answer to *"how much should we let Claude think on
this task?"* — backed by a sweep on your real data with confidence intervals.

## What's the niche, exactly?

| | LLMThinkBench | usail-hkust bench | o1 scaling laws | **depth-lens** |
|---|---|---|---|---|
| Compute-axis curves (not single point) | ❌ | partial | ✅ (o1 only) | **✅** |
| Cross-vendor (Claude / o3 / Gemini / OSS) | ❌ HF only | partial | ❌ o1 only | **✅** |
| Looped transformer (OpenMythos) | ❌ | ❌ | ❌ | **✅** |
| Bring-your-own JSONL | ❌ | ❌ | ❌ | **✅** |
| Cost per prediction with sweep | ❌ | ❌ | ❌ | **✅** |
| Bounded-depth synthetic probes (K-hop, parity, …) | ❌ | partial | ❌ | **✅** |

depth-lens is the **measurement tool you reach for once you've already picked a
model and need to decide how much to spend per query** — on your real workload.

## Quickstart for API users

You have a CSV of test prompts and you want to know: at what thinking budget
does Claude/o3/Gemini start "overthinking" on this task, and what does that
cost?

1. Convert to JSONL with one `{"prompt", "target"}` per line (and optional
   `"depth"` if your prompts have a difficulty axis).
2. Run a sweep:

   ```bash
   depth-lens probe \
       --model anthropic:claude-opus-4-7 \
       --task custom:./my_eval.jsonl:first_int \
       --compute 1024,4096,16384 \
       --n-samples 32 \
       --save-json result.json
   ```

3. The console summary tells you the sweet spot. Plug that into your
   production call:

   ```python
   client.messages.create(
       model="claude-opus-4-7",
       thinking={"type": "enabled", "budget_tokens": 4096},  # what depth-lens found
       ...
   )
   ```

## Quickstart for researchers

```bash
depth-lens probe \
    --model openmythos --task k-hop \
    --depths 2,3,4,5,6,7,8 \
    --compute 1,2,4,8,16 \
    --train-steps 5000 \
    --plot probe.png
```

Trains a 925K-param OpenMythos on K-hop modular composition (~7 min on a
consumer GPU), then sweeps loop count × task depth. Bundled probes
(`k-hop`, `parity`, `graph-reach`, `state-tracking`) all expose a controllable
depth axis so you can study extrapolation.

Findings from the bundled probes:
[**docs/findings/v0.5-openmythos.md**](docs/findings/v0.5-openmythos.md)

## Compare models on the same task

```bash
depth-lens compare \
    --models openmythos,hf:Qwen/Qwen2.5-1.5B-Instruct,anthropic:claude-opus-4-7 \
    --task custom:./my_eval.jsonl:first_int \
    --plot compare.png
```

Each adapter plots its own compute knob on its own axis, side-by-side per
task depth.

## Browse cached runs

```bash
depth-lens dashboard
```

Streamlit picks up every probe you've already cached and lets you filter,
overlay curves, view heatmaps, and read the auto-generated overthinking
report.

## Python API

```python
from depth_lens import probe
from depth_lens.tasks import get_task

task = get_task("custom:./my_eval.jsonl:first_int")
adapter = ...   # any ModelAdapter — see CONTRIBUTING.md for the interface

result = probe(adapter, task, depths=task.available_depths(), n_samples=64)

print(f"effective depth: {result.effective_depth(0.5)}")
print(f"overthinking @ d=4: {result.overthinking(4)}")
print(f"Wilson 95% CIs:\n{result.ci()}")

# Cost per prediction (Claude Opus 4.6 approx pricing):
print(result.cost_per_cell({"input": 15.0, "output": 75.0}))
```

## Built-in adapters

| Spec | Compute knob | Notes |
|---|---|---|
| `openmythos` | `n_loops` | Trains a small model on the task if no checkpoint is supplied. |
| `hf:<hf-model-id>` | `max_thinking_tokens` | Any HF causal LM with a CoT prompt; auto-uses chat template. |
| `anthropic:<model>` | `thinking_budget_tokens` | Claude extended thinking. Requires `ANTHROPIC_API_KEY`. |
| `openai:<model>` | `reasoning_effort` | o-series effort (low/medium/high). `OPENAI_API_KEY`. |
| `gemini:<model>` | `thinking_budget_tokens` | Gemini 2.5 thinking. `GOOGLE_API_KEY`. |
| `vllm:<model>` | `reasoning_effort` | OpenAI-compatible local server. Set `VLLM_BASE_URL`. |

API adapters fan requests across a thread pool (`max_concurrent` kwarg) so
1000-prompt probes finish in minutes, not hours.

## Built-in tasks

| Task | Depth axis | Use |
|---|---|---|
| `custom:<jsonl>:<scorer>` | optional `depth` field | **Bring your own data.** Scorers: `exact`, `first_int`, `last_int`, `yes_no`, `contains`, `regex:<p>`. |
| `k-hop` | K (operators) | Modular composition on Z/23Z. Latent-CoT probe. |
| `parity` | n (bits) | XOR over n bits. State-tracking minimal case. |
| `graph-reach` | path length | Yes/no DAG reachability. Balanced positives/negatives. |
| `state-tracking` | K (instructions) | Two-counter register machine (inc1, inc2, swap, add). |

## Install

```bash
git clone https://github.com/yutoTachibana/depth-lens.git
cd depth-lens

# API-only — no GPU needed
pip install -e .[anthropic,openai,gemini,dashboard]

# +looped transformer + HuggingFace local probes
pip install -e .[openmythos,huggingface,anthropic,openai,gemini,dashboard]
```

Python 3.11+. The bundled OpenMythos training helper assumes CUDA; everything
else is happy on CPU or against remote APIs.

## Status

- [x] **v0.1 MVP**
- [x] **v0.5** — 4 tasks, 6 adapters, Wilson CIs, cache, Streamlit dashboard
- [x] **v1.0 (in progress)** — concurrent API eval, CONTRIBUTING, multi-stage Docker, JA docs, CustomTask + cost tracking
- [ ] **v1.0 final** — cross-vendor benchmark figure, PyPI publish

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
