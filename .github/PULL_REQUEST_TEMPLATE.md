<!-- Thanks for opening a PR! Brief checklist below. -->

## What this changes

One-liner.

## Why

Brief context. Link the issue if there is one.

## Test plan

- [ ] `pytest` passes locally
- [ ] If adding a Task: depth-lens can probe a real model on it (paste the smoke command)
- [ ] If adding an Adapter: mocked unit test in `tests/test_<adapter>.py` (see `tests/test_anthropic_adapter.py` for the pattern)
- [ ] If touching probe / cache / scoring: hand-checked against a small known case

## Reproducible smoke (optional but appreciated for findings PRs)

```bash
depth-lens ... 
```
