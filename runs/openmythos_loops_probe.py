"""
Train a small OpenMythos on K-hop, then probe with an n_loops sweep
that exceeds the training max_loop_iters. Captures accuracy + latency
per cell so we can show the canonical 'pay latency for accuracy' trade.
"""

import json
import time
from pathlib import Path

from depth_lens.adapters.base import ComputeLevel
from depth_lens.adapters.openmythos_adapter import TrainConfig, train_for_task
from depth_lens.metrics import probe
from depth_lens.tasks import get_task

DEPTHS = [2, 4, 6, 8]             # matches the API h2h bench
LOOP_GRID = [1, 2, 4, 8, 16, 32]  # train was max_loop_iters=4
N_SAMPLES = 64                    # tight CIs for the plot

task = get_task("k-hop")
print(f"[1/3] Training OpenMythos on k-hop (5000 steps)…")
t0 = time.time()
adapter = train_for_task(task, cfg=TrainConfig(steps=5000))
print(f"  trained in {time.time() - t0:.1f}s")

print(f"\n[2/3] Probing n_loops {LOOP_GRID} × depths {DEPTHS} × n={N_SAMPLES}…")
result = probe(
    adapter=adapter,
    task=task,
    depths=DEPTHS,
    compute_grid=[ComputeLevel(n, f"n_loops={n}") for n in LOOP_GRID],
    n_samples=N_SAMPLES,
    batch_size=64,
    use_cache=False,
)

_DOCKER_ROOT = Path("/work")
_ROOT = _DOCKER_ROOT if _DOCKER_ROOT.exists() else Path(__file__).resolve().parent.parent
out_path = _ROOT / "runs/openmythos_loops.json"
out_path.write_text(json.dumps({
    "task": result.task_name,
    "adapter": result.adapter_name,
    "compute_axis": result.compute_axis,
    "depths": result.depths,
    "compute_grid": [{"value": c.value, "label": c.label} for c in result.compute_grid],
    "accuracy": result.accuracy,
    "n_per_cell": result.n_per_cell,
    "latency_per_cell": result.latency_per_cell,
    "tokens_per_cell": result.tokens_per_cell,
}, indent=2))
print(f"\n[3/3] Saved -> {out_path}")

print("\n=== Accuracy ===")
print(f"{'depth':>6s}  " + "  ".join(f"{c.label:>11s}" for c in result.compute_grid))
for di, d in enumerate(result.depths):
    cells = result.accuracy[di]
    print(f"d={d:<5d} " + "  ".join(f"{c:>11.2f}" for c in cells))

print("\n=== Latency (sec/pred) ===")
for di, d in enumerate(result.depths):
    cells = result.latency_per_cell[di] if result.latency_per_cell else [0]*len(LOOP_GRID)
    print(f"d={d:<5d} " + "  ".join(f"{c:>11.4f}" for c in cells))
