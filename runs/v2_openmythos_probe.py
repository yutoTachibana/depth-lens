"""
v2.0 OpenMythos probe sweep.

For each (size, task) checkpoint produced by v2_train_sweep.py, run a
probe across (depth × n_loops) and save the result JSON. The downstream
scaling-law plot reads these JSONs along with v2_api_sweep.py outputs
and (in v2.0) vLLM probe outputs.

Usage (inside depth-lens:gpu container, after training):
    python runs/v2_openmythos_probe.py
    python runs/v2_openmythos_probe.py --sizes 1M,10M --tasks k-hop
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

_DOCKER_ROOT = Path("/work")
ROOT = _DOCKER_ROOT if _DOCKER_ROOT.exists() else Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "runs" / "v2_ckpts"
OUT_DIR = ROOT / "runs" / "v2_openmythos"

SIZES = ("1M", "10M", "100M")
TASKS = ("k-hop", "parity", "state-tracking", "mini-csp", "dict-lookup")

# Probe depths per task — should match v2_api_sweep.TASK_DEPTHS so the
# scaling-law plot can overlay matching x values.
TASK_DEPTHS = {
    "k-hop":          [2, 4, 6, 8, 10],
    "parity":         [4, 8, 12, 16],
    "state-tracking": [3, 5, 7, 9],
    "mini-csp":       [3, 5, 7],
    "dict-lookup":    [2, 4, 6, 8, 10],
}

# n_loops sweep — exercise the looped scaling axis. We include values past
# the training max_loop_iters to probe extrapolation behavior.
N_LOOPS_GRID = [1, 2, 4, 8, 16]

N_SAMPLES = 64  # local GPU is cheap; tight CIs are essentially free here


def probe_one(size: str, task_name: str) -> Path | None:
    from depth_lens.adapters.base import ComputeLevel
    from depth_lens.adapters.openmythos_adapter import load_checkpoint
    from depth_lens.metrics import probe
    from depth_lens.tasks import get_task

    ckpt = CKPT_DIR / f"{size}_{task_name.replace('-', '_')}.pt"
    if not ckpt.exists():
        print(f"  [missing-ckpt] {ckpt}", flush=True)
        return None

    out = OUT_DIR / f"{size}__{task_name}.json"
    if out.exists():
        print(f"  [skip-existing] {out.name}", flush=True)
        return out

    print(f"\n=== Probing {size}/{task_name} ===", flush=True)
    t0 = time.time()
    adapter = load_checkpoint(ckpt)
    task = get_task(task_name)
    depths = TASK_DEPTHS.get(task_name, [2, 4, 6])
    compute_grid = [ComputeLevel(n, f"n_loops={n}") for n in N_LOOPS_GRID]

    result = probe(
        adapter=adapter,
        task=task,
        depths=depths,
        compute_grid=compute_grid,
        n_samples=N_SAMPLES,
        batch_size=64,
        use_cache=False,
        verbose=False,
    )

    # Count trained params for FLOPs estimation downstream.
    n_params = sum(p.numel() for p in adapter.model.parameters())

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "size": size,
        "task": task_name,
        "n_params": n_params,
        "depths": result.depths,
        "compute_grid": [{"value": c.value, "label": c.label} for c in result.compute_grid],
        "accuracy": result.accuracy,
        "n_per_cell": result.n_per_cell,
        "latency_per_cell": result.latency_per_cell,
        "tokens_per_cell": result.tokens_per_cell,
    }, indent=2, ensure_ascii=False))
    dt = time.time() - t0
    print(f"   {n_params/1e6:.1f}M params, {dt:.0f}s -> {out.name}", flush=True)

    # Free GPU memory before the next size.
    adapter.teardown()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default=",".join(SIZES))
    p.add_argument("--tasks", default=",".join(TASKS))
    args = p.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"v2.0 OpenMythos probe — sizes={sizes}, tasks={tasks}\n")

    log = []
    for size in sizes:
        for task in tasks:
            out = probe_one(size, task)
            log.append({"size": size, "task": task, "out": str(out) if out else None})

    (OUT_DIR / "_index.json").write_text(json.dumps(log, indent=2))
    print(f"\nWrote {sum(1 for r in log if r['out'])} probe JSONs to {OUT_DIR}")


if __name__ == "__main__":
    main()
