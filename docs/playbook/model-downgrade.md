# Playbook: "Can we switch from Opus to Haiku?"

## The situation

You picked the safest-looking model (Opus / GPT-5 / Gemini 3.1 Pro) at
prototype time and never re-evaluated. The bill is real now. You suspect
the cheaper tier could handle it — but you need data, not vibes.

## Recipe

1. Export 50-500 representative production prompts to JSONL.
   `prompt`, `target`, optional `depth` per line.

   ```jsonl
   {"prompt": "Classify the intent: 'How do I cancel?'", "target": "cancel_request", "depth": 1}
   {"prompt": "Extract the order ID from: ...",         "target": "ORD-7841",       "depth": 1}
   ```

   If your task doesn't have a verifiable target, build a regex-scorable
   ground truth — see `--task custom:<jsonl>:regex:...` and `:contains:`.

2. Run `recommend` across the candidates that span your vendor's price ladder:

   ```bash
   depth-lens recommend \
       --models anthropic:claude-haiku-4-5,anthropic:claude-sonnet-4-6,anthropic:claude-opus-4-7 \
       --task custom:./prod_eval.jsonl:first_int \
       --target-accuracy 0.97 \
       --n-samples 64 \
       --daily-calls 10000
   ```

3. Read the output. The cheapest passing config is your candidate
   downgrade. Pin the `(model, thinking_budget)` pair in your prompt
   manager.

## Cost of running this recipe

Roughly $0.50 – $3 depending on the size of the candidate set and
`n_samples`. A 500-prompt eval with 3 Anthropic tiers × 3 budgets × 64
samples = ~5,800 API calls; expect ~$10 in the worst case (lots of
Opus calls).

That's a single afternoon's investment vs the annualized savings —
~$100k/year in the headline README example.

## How to be honest with yourself

- **n_samples matters.** At n=16, a 1.0 accuracy might be 0.85 in reality
  (Wilson 95% lower bound). At n=64+, the CI tightens. depth-lens prints
  CIs in the JSON output — check them before pinning a budget.
- **Your eval JSONL is the load-bearing part.** Garbage in, garbage out.
  Prefer recent production prompts. Stratify by prompt type if you can.
- **The cheapest passing config can still fail in production** — different
  prompts, different load patterns. Treat this as the *prior*, not the
  decision. Run a small canary at the recommended setting before
  fully cutting over.

## After the cutover

Re-run the same command against new production samples each month.
Cache hits make this near-free; the only re-pay is when prompts drift
enough to invalidate the cache key. See
[regression-detection.md](./regression-detection.md) for the
fail-on-drop workflow.
