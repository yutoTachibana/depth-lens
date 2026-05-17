# Self-hosting with vLLM — putting Llama-3 on the same Pareto as Claude

This playbook walks through measuring a **self-hosted** open-weights model
(Llama-3-8B-Instruct, DeepSeek-R1-Distill-Qwen-7B, etc.) on the same
accuracy / cost / latency axes as a hosted API like Claude or GPT.

The use case: you're paying $X/month for a frontier API and want to know
whether running an open model on a single GPU would meet your accuracy bar
at lower TCO. depth-lens lets you put both on the same plot.

## Requirements

- An NVIDIA GPU with at least ~8 GB free for the smaller models below.
  A 4080 SUPER (16 GB) handles both bundled configs comfortably.
- Docker with the NVIDIA Container Toolkit (`nvidia-container-runtime`).
- `pip install depth-lens` (or local checkout).

## 1. Start a vLLM server

Two ready-to-run compose files ship in [`docker/`](../../docker/):

### Llama-3-8B-Instruct (AWQ 4-bit, non-thinking baseline)

```bash
docker compose -f docker/vllm-llama3-8b.yml up -d
# wait ~60s for model download (first run only) and server start
curl -s localhost:8000/v1/models | head
```

This serves [`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`](https://huggingface.co/hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4)
quantized to 4 bits via AWQ Marlin — VRAM footprint ~5 GB, throughput on a
4080 SUPER ~100 tok/s for short generations.

### DeepSeek-R1-Distill-Qwen-7B (thinking model)

```bash
docker compose -f docker/vllm-deepseek-r1-distill.yml up -d
```

This serves
[`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
in fp16 with vLLM's reasoning parser enabled, so the model's thinking
output is separated from the final answer and depth-lens can sweep
`reasoning_effort` the same way it does for Claude / OpenAI o-series.

## 2. Point depth-lens at it

### Non-thinking model: sweep `max_tokens`

For an instruct-style model with no first-class thinking knob, sweep the
response length cap. Longer cap = more room for CoT.

```bash
depth-lens probe \
  --model vllm:hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --task k-hop --depths 2,4,6,8 \
  --compute-axis max_tokens --compute 256,1024,4096 \
  --n-samples 16 \
  --save-json runs/vllm_llama3_8b.json
```

### Thinking model: sweep `reasoning_effort` (default)

```bash
depth-lens probe \
  --model vllm:deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --task k-hop --depths 2,4,6,8 \
  --compute low,medium,high \
  --n-samples 16 \
  --save-json runs/vllm_deepseek_r1_distill.json
```

## 3. Recommend across hosted + self-hosted

The `recommend` command treats self-hosted specs the same as API specs
once you've supplied a GPU-hour rate:

```bash
depth-lens recommend \
  --models anthropic:claude-haiku-4-5,openai:o4-mini,vllm:hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --task custom:./my_eval.jsonl:first_int \
  --target-accuracy 0.95 \
  --daily-calls 100000 \
  --gpu-hourly-rate 0.50
```

For self-hosted specs (`vllm:*`, `hf:*`, `openmythos`), cost per call is
amortized as `latency_seconds × $/GPU-hour × num_GPUs / 3600`. The
`--gpu-hourly-rate` flag overrides the default ($0.50, AWS g5 spot
midpoint). For an on-prem GPU you own, $0.10–0.20/hour amortized over a
3-year hardware life is more realistic.

## 4. Cleanup

```bash
docker compose -f docker/vllm-llama3-8b.yml down
docker compose -f docker/vllm-deepseek-r1-distill.yml down
```

## Caveats

- **Quantization matters.** AWQ-INT4 is faster and smaller than fp16 but
  loses some accuracy. Always probe before deciding "self-hosted is good
  enough"; the Pareto plot lets you see how much.
- **`max_tokens` is a coarse compute knob.** For a non-thinking model,
  longer responses don't always mean better reasoning — they often mean
  rambling. The Pareto shape (accuracy vs latency) tells you whether the
  model is actually USING the extra tokens.
- **GPU-hour pricing is amortization, not billing.** It's meant to make
  self-hosted comparable to per-call API pricing in a single chart.
  Real on-prem TCO includes hardware, power, ops time, and idle capacity.
  See [`docs/findings/v1.2-self-hosted-vs-api.md`](../findings/v1.2-self-hosted-vs-api.md)
  for the worked example.
