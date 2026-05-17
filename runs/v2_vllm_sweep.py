"""
v2.0 self-hosted vLLM probe sweep.

For each (model, task) combination, expects a vLLM server already running
at VLLM_BASE_URL (default http://localhost:8000/v1). Probes the served
model across the same task-specific depth grid as v2_api_sweep.py and
saves results to runs/v2_vllm/.

Designed to be run twice — once per model — between docker-compose up/down
cycles. The 4080 SUPER doesn't have enough VRAM to host both models
simultaneously.

Usage:
    # Llama-3-8B-Instruct AWQ (non-thinking, sweeps max_tokens)
    docker compose -f docker/vllm-llama3-8b.yml up -d
    python runs/v2_vllm_sweep.py \
        --model vllm:hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
        --compute-axis max_tokens --compute 256,1024,3000
    docker compose -f docker/vllm-llama3-8b.yml down

    # DeepSeek-R1-Distill-Qwen-1.5B (thinking, sweeps reasoning_effort)
    docker compose -f docker/vllm-deepseek-r1-distill.yml up -d
    python runs/v2_vllm_sweep.py \
        --model vllm:deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --compute-axis reasoning_effort --compute low,medium,high
    docker compose -f docker/vllm-deepseek-r1-distill.yml down
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
OUT_DIR = ROOT / "runs" / "v2_vllm"

TASK_DEPTHS = {
    "k-hop":          "2,4,6,8,10",
    "parity":         "4,8,12,16",
    "state-tracking": "3,5,7,9",
    "mini-csp":       "3,5,7",
    "dict-lookup":    "2,4,6,8,10",
}


def probe_one(model: str, task: str, depths: str, compute_axis: str,
              compute: str, n_samples: int) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace(":", "_").replace("/", "_")
    out = OUT_DIR / f"{task}__{safe_model}.json"
    if out.exists():
        print(f"  [skip-existing] {out.name}", flush=True)
        return out

    cmd = [
        sys.executable, "-m", "depth_lens.cli", "probe",
        "--model", model,
        "--task", task,
        "--depths", depths,
        "--compute-axis", compute_axis,
        "--compute", compute,
        "--n-samples", str(n_samples),
        "--batch-size", "8",
        "--save-json", str(out),
        "--plot", str(out.with_suffix(".png")),
    ]
    print(f"  [run] {model} / {task} ({compute_axis})", flush=True)
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
    p.add_argument("--model", required=True,
                   help="vllm:<model> spec for the currently-served model")
    p.add_argument("--tasks", default=",".join(TASK_DEPTHS))
    p.add_argument("--compute-axis", default="reasoning_effort",
                   choices=["reasoning_effort", "max_tokens"])
    p.add_argument("--compute", default="low,medium,high",
                   help="comma-separated compute levels (efforts or token counts)")
    p.add_argument("--n-samples", default=12, type=int)
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print(f"v2.0 vLLM sweep — model={args.model}\n  tasks={tasks}\n  "
          f"axis={args.compute_axis} compute={args.compute}\n")

    log = []
    for task in tasks:
        depths = TASK_DEPTHS.get(task, "2,4,6,8")
        out = probe_one(args.model, task, depths,
                        args.compute_axis, args.compute, args.n_samples)
        log.append({"task": task, "model": args.model, "out": str(out)})

    (OUT_DIR / f"_index_{args.model.replace(':', '_').replace('/', '_')}.json").write_text(
        json.dumps(log, indent=2)
    )
    print(f"\nWrote {len(log)} probe JSONs to {OUT_DIR}")


if __name__ == "__main__":
    main()
