"""
Verify the gemini-2.5-flash tier 4 = 0.00 result.

We re-probe with n=32 at the lowest budget and ALSO print 4 raw model
responses so we can eyeball whether Flash is:
  (a) actually answering wrong (genuine 0.00) → keep the finding
  (b) returning empty / 429-degraded text → artifact, replace
"""

import os
from pathlib import Path

from depth_lens.adapters.base import ComputeLevel
from depth_lens.adapters.gemini_adapter import GeminiAdapter
from depth_lens.tasks import get_task

_DOCKER_ROOT = Path("/work")
_ROOT = _DOCKER_ROOT if _DOCKER_ROOT.exists() else Path(__file__).resolve().parent.parent

task = get_task(f"custom:{_ROOT}/runs/bench.jsonl:first_int")
adapter = GeminiAdapter(model="gemini-2.5-flash", task_name="custom:bench")

# Only tier 4 (the cell we're verifying), lowest budget.
N = 32
insts = task.generate(depth=4, n_samples=N, seed=0)
compute = ComputeLevel(1024, "thinking_budget_tokens=1024")

print(f"Probing gemini-2.5-flash on bench.jsonl tier=4 (mod 97 K=14), "
      f"budget=1024, n={N}")
print("-" * 80)

preds = adapter.predict([i.prompt for i in insts], compute)

correct = 0
for i, (inst, pred) in enumerate(zip(insts, preds, strict=False)):
    score = task.score(inst, pred.text)
    correct += int(score)
    if i < 4:
        print(f"\n=== Example {i+1} ===")
        print(f"PROMPT (truncated): {inst.prompt[:150]}...")
        print(f"TARGET: {inst.target}")
        print(f"PRED.text (extracted): {pred.text[:200]}")
        print(f"PRED.metadata.raw_text (truncated):")
        raw = pred.metadata.get("raw_text", "")
        print(f"  {raw[:500]}{'... (truncated)' if len(raw) > 500 else ''}")
        print(f"SCORE: {score}")

acc = correct / N
print()
print("=" * 80)
print(f"FINAL: gemini-2.5-flash @ tier 4, budget=1024, n={N}: acc = {acc:.3f}")
print(f"  (original bench had n=16, acc = 0.00)")
print("=" * 80)
