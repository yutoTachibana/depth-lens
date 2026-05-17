"""
v2.0 cross-vendor API probe sweep.

Run depth-lens probe on each of the 5 tasks against 3 vendor models that
sit on the inference-time-compute frontier we want to plot:

  - openai:gpt-5-mini (small/cheap reasoning)
  - openai:o4-mini    (small/efficient reasoning)
  - gemini:gemini-3.1-flash-lite (tiny/cheap thinking)

We pick *one model per vendor* (smallest/cheapest of the latest gen) so the
3-paradigm scaling-law plot stays readable. Saves results to runs/v2_api/.

Usage (from depth-lens repo root):
    python runs/v2_api_sweep.py
    python runs/v2_api_sweep.py --tasks k-hop,dict-lookup
    python runs/v2_api_sweep.py --models openai:o4-mini

Cost: roughly $5-15 total at the default n_samples=12 (per model per task
per compute level).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_DOCKER_ROOT = Path("/work")
ROOT = _DOCKER_ROOT if _DOCKER_ROOT.exists() else Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "runs" / "v2_api"

DEFAULT_MODELS = [
    "openai:gpt-5-mini",
    "openai:o4-mini",
    "gemini:gemini-3.1-flash-lite",
    # Anthropic would be nice but we'd need an ANTHROPIC_API_KEY at runtime.
    # The plot can still place Claude points later by reading old cvl_*.json.
]

# Tasks and the depths we want to probe at each. Chosen so:
#   - low depths are inside the OpenMythos training distribution
#   - high depths are outside, to see where each paradigm breaks
TASK_DEPTHS = {
    "k-hop": "2,4,6,8,10",
    "parity": "4,8,12,16",
    "state-tracking": "3,5,7,9",
    "mini-csp": "3,5,7",
    "dict-lookup": "2,4,6,8,10",
}

# OpenAI / Gemini both use enum efforts — sweep all 3 for the curve.
DEFAULT_COMPUTE = "low,medium,high"


def probe_one(model: str, task: str, depths: str, compute: str, n_samples: int) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace(":", "_").replace("/", "_")
    out = OUT_DIR / f"{task}__{safe_model}.json"
    if out.exists():
        print(f"  [skip-existing] {out}", flush=True)
        return out

    cmd = [
        sys.executable, "-m", "depth_lens.cli", "probe",
        "--model", model,
        "--task", task,
        "--depths", depths,
        "--compute", compute,
        "--n-samples", str(n_samples),
        "--batch-size", "8",
        "--save-json", str(out),
        "--plot", str(out.with_suffix(".png")),
    ]
    print(f"  [run] {model} / {task}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"   FAILED ({dt:.0f}s):\n{r.stderr[-400:]}", flush=True)
    else:
        print(f"   ok ({dt:.0f}s) -> {out.name}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--tasks", default=",".join(TASK_DEPTHS))
    p.add_argument("--compute", default=DEFAULT_COMPUTE)
    p.add_argument("--n-samples", default=12, type=int)
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print(f"v2.0 API sweep — models={models}\n  tasks={tasks}\n")
    log = []
    for task in tasks:
        depths = TASK_DEPTHS.get(task, "2,4,6,8")
        for model in models:
            out = probe_one(model, task, depths, args.compute, args.n_samples)
            log.append({"task": task, "model": model, "out": str(out)})

    (OUT_DIR / "_index.json").write_text(json.dumps(log, indent=2))
    print(f"\nWrote {len(log)} probe JSONs to {OUT_DIR}")


if __name__ == "__main__":
    main()
