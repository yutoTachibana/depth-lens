# Playbook: "We spend $5k/mo on Claude — where can we cut?"

## The situation

Production LLM bills creep up. You don't know which prompts are paying
for thinking they don't need vs which genuinely need Opus. You want a
spreadsheet at the end of this saying *"task X switches from Opus to
Haiku, saves $X/year"* — for every task.

## Recipe

1. Group your production prompts by the task they're solving. Don't
   over-engineer — 5-10 categories is plenty. Examples:

   - intent classification
   - structured extraction (order ID, address, date)
   - free-form summarization
   - reasoning over multi-step constraints
   - safety / policy classification

2. Build one eval JSONL per task. 100-300 prompts each, with verifiable
   targets. Stratify by complexity if you can — include some hard ones.

3. For each task, run `recommend` to find the cheapest passing
   configuration:

   ```bash
   for task in intent extraction summary reasoning safety; do
     depth-lens recommend \
         --models anthropic:claude-haiku-4-5,anthropic:claude-sonnet-4-6,anthropic:claude-opus-4-7 \
         --task custom:./tasks/${task}.jsonl:first_int \
         --target-accuracy 0.97 \
         --n-samples 64 \
         --daily-calls "$(cat tasks/${task}.calls_per_day)" \
         > recommendations/${task}.txt
   done
   ```

4. Read the recommendations. Anything currently routed to Opus that
   the recommender says works on Haiku/Sonnet → that's your action item.

## How to estimate task-by-task daily call volume

If you have an LLM proxy (LiteLLM, Helicone, your own router), it
already logs per-prompt-type call rates. If not, sample 24h of your
production logs and bucket by prompt template.

`depth-lens recommend --daily-calls N` will project annualised savings
so the spreadsheet writes itself.

## Reading the output

```
✅ Passing (cheapest first):
  anthropic:claude-haiku-4-5    d=1  thinking_budget_tokens=4096  acc=0.98  $0.00280/k-pred  ← cheapest
  anthropic:claude-sonnet-4-6   d=1  thinking_budget_tokens=1024  acc=0.97  $0.00489/k-pred
  anthropic:claude-opus-4-7     d=1  thinking_budget_tokens=1024  acc=0.99  $0.02060/k-pred

At 5,000 calls/day with the cheapest passing config:
  → $14/day  $5,110/year

  Switching from anthropic:claude-opus-4-7 @ thinking_budget_tokens=16384 ($173/day)
  saves $159/day = $58,135/year (92% reduction)
```

**Actions to take immediately**:
- The "cheapest" row tells you where to point this task's traffic
- The "saves" line is the projected $-impact
- Pin `(model, thinking_budget_tokens)` in your prompt manager / router

## How conservative to be with `target-accuracy`

- `0.99` → keeps Opus-class behavior; only finds savings in places where
  Opus was overkill.
- `0.97` → realistic for most production workloads; finds many savings.
- `0.95` → aggressive; will demote some tasks to Haiku where Sonnet was
  picking up edge cases. Pair with a canary.

## Caveats

1. **One eval JSONL per task** is important. Mixing all your tasks
   together produces a recommender that hedges to the most-capable
   model.
2. **n_samples ≥ 64** for reliable cost decisions. The Wilson CIs widen
   fast below this.
3. **Re-run quarterly** or after major model releases. The cheapest tier
   that "works" can change overnight (see
   [v1.0-gemini-2.5-cross-vendor.md](../findings/v1.0-gemini-2.5-cross-vendor.md)
   for the canonical example).
