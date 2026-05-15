# Your reasoning model has a shape. Standard benchmarks don't show it.

*Introducing depth-lens — a measurement toolkit for inference-time-compute scaling.*

[日本語版](./intro.ja.md)

---

Reasoning models are starting to share a strange new property: **how much you
let them think changes how well they answer.** Claude has `extended thinking`
with a budget in tokens. OpenAI's o-series takes a `reasoning_effort` knob.
Gemini exposes `thinking_budget`. Open-weight looped transformers like
OpenMythos take `n_loops`. Agentic systems iterate until they give up.

All of these expose the same hidden axis: **inference-time compute**. And all
of the standard benchmarks — MMLU, GSM8K, HumanEval — average over that axis
into a single number. You can't tell from a benchmark score whether a model
gets monotonically better with more compute, plateaus quickly, or
*overthinks* (gets worse with too much budget).

**depth-lens** is a small open-source tool that fixes this. Point it at any
reasoning system, and it sweeps task depth × compute budget and produces an
accuracy-vs-compute curve, with confidence intervals and automatic detection
of overthinking and effective-reasoning-depth ceilings.

```bash
pip install depth-lens[openmythos,huggingface,anthropic,openai,gemini]
depth-lens probe --model anthropic:claude-opus-4-7 --task k-hop --depths 2,4,6,8
```

```
effective depth (≥0.5 acc at some compute): 7
overthinking @ depth 4: peak=think=4096 (acc=1.00)  →  last=think=16384 (acc=0.87)
```

---

## Why this matters: three models, three different shapes

Here's what we found in a one-day prototyping session with depth-lens on
four bounded-depth reasoning tasks. We trained a tiny 925K-parameter
OpenMythos (Recurrent-Depth Transformer) on each task and ran the probe.
Same architecture, same training recipe, four tasks. The compute-scaling
curves are **qualitatively different**:

| Task | Compute-scaling profile | What it means |
|---|---|---|
| K-hop modular composition | **Strong overthinking.** Accuracy peaks at training depth (loops=4), then degrades. By 16 loops, accuracy at K=5 has fallen from 1.00 to 0.61. | The model committed to its answer early; more loops drift the hidden state past the solution. |
| Parity (XOR over n bits) | **Clean monotonic improvement** up to training depth, then flat. No overthinking. | The model uses every loop productively until convergence; extra compute is harmless but not helpful. |
| Graph reachability | **Zero compute benefit.** Loops 1, 2, 4, 8, 16 all give ~0.70 accuracy. Heuristic saturation. | The model isn't learning the recursive algorithm — it found a 70% heuristic and stopped. More compute won't help; more parameters or different training would. |
| Two-counter state tracking | **Overthinking + best extrapolation.** Peak at loops=4 across all depths (just like K-hop), but extrapolates further: K=8 still 0.61 at peak compute. | The vector-valued state is harder to memorise, so the model genuinely learns composition — visible as graceful degradation rather than the cliff K-hop shows past training depth. |

These three behaviours are **invisible to a single accuracy number**.
"OpenMythos achieves 0.7 on graph-reach" is the same number you'd report for
"saturated at a heuristic" and for "needs more compute to climb higher". Only
the curve tells you which one.

---

## What depth-lens actually does

The core abstraction is a **probe**: a sweep of (task depth × compute level)
that returns an accuracy grid with Wilson 95% confidence intervals. Each
adapter exposes its native compute knob:

- `openmythos` → `n_loops`
- `hf:<model>` → `max_thinking_tokens` (CoT budget)
- `anthropic:<model>` → `thinking_budget_tokens`
- `openai:<model>` → `reasoning_effort` (low / medium / high)
- `gemini:<model>` → `thinking_budget_tokens`
- `vllm:<model>` → `reasoning_effort` (OpenAI-compatible local server)

The probe engine doesn't care which knob — it sweeps it, scores predictions
with a task-specific lenient scorer (extracting "Final answer: X" from
verbose CoT outputs), and reports diagnostics:

- **`effective_depth(threshold)`** — the largest task depth where the model
  clears a given accuracy bar at *some* compute level. Like "this model can
  handle 7-hop composition if you give it enough budget".
- **`overthinking(depth)`** — peak compute is not the maximum compute, and
  the accuracy drop is non-trivial. The detector returns the peak compute
  level and the drop magnitude.
- **Wilson 95% CIs** on every cell, plotted as bands on curves and
  available as `.ci()` on the result.

Cached probes feed an **interactive Streamlit dashboard** (`depth-lens
dashboard`) so you can browse the runs you've accumulated.

---

## What depth-lens isn't

It's not lm-eval-harness or HELM. We don't ship Wikipedia or MATH; the tasks
are deliberately small, depth-controllable, and synthesisable. The point
isn't "is your model good at general reasoning" — it's "given that your
model spends compute on reasoning, what shape does the spend-vs-quality
curve have?"

It's also not a model zoo. We ship a tiny OpenMythos training helper for
reproducible demos, but depth-lens is meant to be pointed at *your* model,
whether that's a frontier API, an OSS reasoner, or a research artefact.

---

## Where it came from

depth-lens grew out of a one-day experiment on
[OpenMythos](https://github.com/kyegomez/OpenMythos), the open-source
PyTorch reconstruction of Anthropic's hypothesised Claude Mythos
architecture. We probed OpenMythos on a K-hop composition task,
saw that *more loops didn't always help* — and realized the existing
benchmarking ecosystem had no way to surface that fact for any model.

The OpenMythos experiment also produced a non-obvious second finding: the
ACT (Adaptive Computation Time) halting mechanism, not the recurrent block
itself, was the bottleneck on extrapolation. The raw hidden state at
loop=12 contained the right answer; the ACT-weighted readout had committed
to a wrong one by loop=4. This kind of structural finding is exactly what
depth-lens is built to make routine.

---

## What's next

- v0.5 (now): five adapter families, four tasks, Wilson CIs, cache layer,
  Streamlit dashboard.
- v1.0: a paper-quality cross-vendor benchmark figure (Claude / o-series /
  Gemini / open-weight reasoners on the full task suite), PyPI release,
  async batched eval, and a fifth task (mini-SAT).

If you've ever stared at a reasoning model and wondered "is more thinking
actually helping?", you can answer it now in two commands. Try it on your
favourite model and see what shape comes out.

→ https://github.com/yutoTachibana/depth-lens
