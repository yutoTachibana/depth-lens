# Playbook: "Anthropic shipped a new model — does it break my pipeline?"

## The situation

Sonnet 4.7 just deprecated 4.6. Or Opus 4.8 dropped. Or Gemini's quietly
swapping internals (this happens). Your prompts are tuned for the old
behavior. **Will the new model regress?**

This is the LLM-equivalent of running CI before merging a dependency
upgrade. depth-lens is the test runner.

## Recipe

1. Take a snapshot of the old model's behavior on your eval data. The
   first time, run a full probe and commit the JSON:

   ```bash
   depth-lens probe \
       --model anthropic:claude-sonnet-4-6 \
       --task custom:./prod_eval.jsonl:first_int \
       --compute 1024,4096,16384 \
       --n-samples 64 \
       --save-json baseline.json
   git add baseline.json && git commit -m "baseline: Sonnet 4.6 prod eval"
   ```

2. When the vendor ships a new model, run the same probe against it:

   ```bash
   depth-lens probe \
       --model anthropic:claude-sonnet-4-7 \
       --task custom:./prod_eval.jsonl:first_int \
       --compute 1024,4096,16384 \
       --n-samples 64 \
       --save-json candidate.json
   ```

3. Diff the two JSONs cell-by-cell:

   ```bash
   python -c '
   import json
   b = json.load(open("baseline.json"))
   c = json.load(open("candidate.json"))
   for di, d in enumerate(b["depths"]):
       for ci, cell in enumerate(b["compute_grid"]):
           ba, ca = b["accuracy"][di][ci], c["accuracy"][di][ci]
           if ca < ba - 0.02:  # 2pp drop is the alert threshold
               print(f"REGRESSION at d={d} {cell[\"label\"]}: {ba:.2f} -> {ca:.2f}")
   '
   ```

4. Cells flagged are what to investigate. Common patterns:

   - **Drop at low budget**: new model is more thinking-hungry. Bump
     budget to 4k for the affected prompts and re-evaluate.
   - **Drop at high budget**: new model overthinks where the old one
     didn't. Look at the `Final answer` parsing — sometimes the model
     started wrapping in markdown.
   - **Drop everywhere**: the new model genuinely is worse on your
     prompt class. Stay on the old one (until it deprecates).

## Hooking it into CI

`depth-lens` exit code is 0 on success. The diff script above is one
`exit(1)` away from being a GitHub Actions step that fails the build
when a model regression exceeds the threshold. Example:

```yaml
- name: Regression check on Sonnet 4.7
  run: |
    depth-lens probe --model anthropic:claude-sonnet-4-7 \
        --task custom:./prod_eval.jsonl:first_int \
        --compute 1024,4096,16384 --n-samples 64 \
        --save-json candidate.json
    python scripts/diff_probe.py baseline.json candidate.json --fail-on 0.02
```

Cost per CI run: ~$2-5 depending on `n_samples`. Cheap relative to a
silent prod regression.

## Wider testing

If you maintain prod across multiple Anthropic / OpenAI tiers, run the
above for each tier. The regression pattern is often
*tier-specific* — e.g., a Sonnet upgrade might be fine while a Haiku
update breaks edge cases.

## What this doesn't catch

- Latency or cost regressions (re-run `depth-lens probe` to check;
  `latency_per_cell` and `tokens_per_cell` are in the JSON).
- Behavior on prompts you don't have in your eval set. Keep the eval
  set close to current production distribution.
- Non-determinism in generation. depth-lens uses temperature=0 / greedy
  decoding by default, so this is mostly a non-issue, but verify in your
  adapter's `predict()` if you've customized.
