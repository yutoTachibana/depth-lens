---
name: Empirical finding
about: A non-obvious result you ran with depth-lens
title: "[finding] "
labels: finding
---

## What you observed

One-sentence summary. (e.g., *"Sonnet 4.6 overthinks parity past
budget=4096."*)

## How to reproduce

```bash
depth-lens probe --model ... --task ... --compute ... --n-samples ...
```

Or commit the JSONL + a one-shot script as a gist and link it.

## Why it matters

What decision would change if a practitioner saw this curve?

## Plot / numbers

Attach the depth-lens plot (or paste the accuracy / cost matrix from
`--save-json`).

## Wilson 95% CI

n_samples per cell, and the cells where the CI doesn't overlap 1.00 (or
your threshold). depth-lens prints this in the summary; paste it.

---

If the finding is interesting and reproducible, we'd love to add it to
[`docs/findings/`](../../docs/findings) — feel free to open a PR.
