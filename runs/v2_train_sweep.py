"""
v2.0 scaling-law training sweep.

Train OpenMythos at 3 sizes (1M / 10M / 100M) across 5 tasks (k-hop,
parity, state-tracking, mini-csp, dict-lookup) and save checkpoints
under runs/v2_ckpts/{size}_{task}.pt for later probing.

Usage:
    python runs/v2_train_sweep.py --sizes 1M,10M --tasks k-hop,parity
    python runs/v2_train_sweep.py --sizes 100M --tasks k-hop      # single big job
    python runs/v2_train_sweep.py --all                            # full sweep (15 jobs)

The 100M jobs each take 3-4 hr on a 4080 SUPER; 10M takes 30-60 min;
1M finishes in 5-10 min. Run inside the depth-lens:gpu docker image.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Resolve repo root whether we're inside the docker /work mount or on a
# native host clone.
_DOCKER_ROOT = Path("/work")
ROOT = _DOCKER_ROOT if _DOCKER_ROOT.exists() else Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "runs" / "v2_ckpts"
LOG_PATH = ROOT / "runs" / "v2_train_log.jsonl"


# Size-preset configs. Param counts verified empirically:
#   1M    → 1.29 M params (slight overshoot of "1M" branding)
#   10M   → 11.89 M params
#   100M  → 106.15 M params
SIZE_CONFIGS = {
    "1M": dict(
        dim=160, n_heads=4, max_loop_iters=4,
        prelude_layers=1, coda_layers=1,
        n_experts=4, expert_dim=256,
    ),
    "10M": dict(
        dim=384, n_heads=6, max_loop_iters=8,
        prelude_layers=1, coda_layers=1,
        n_experts=8, expert_dim=768,
    ),
    "100M": dict(
        dim=1024, n_heads=8, max_loop_iters=12,
        prelude_layers=1, coda_layers=1,
        n_experts=16, expert_dim=1536,
    ),
}

# Training step counts scale with size so each model sees a comparable
# *number of training sequences*: 8000 steps × batch 32 (100M) gave only
# 256K sequences vs 1M (4000 × 256 = 1024K). 100M was severely under-
# trained on the original budget; bumped to 24000 steps so 100M sees
# 768K sequences (roughly parity with 1M and 10M).
STEPS_BY_SIZE = {"1M": 4000, "10M": 6000, "100M": 24000}

# Batch sizes that fit a 16 GB GPU at each scale.
BATCH_BY_SIZE = {"1M": 256, "10M": 128, "100M": 32}

TASKS = ("k-hop", "parity", "state-tracking", "mini-csp", "dict-lookup")


def train_one(size: str, task_name: str) -> dict:
    """Train one (size, task) cell. Returns a log dict."""
    from depth_lens.adapters.openmythos_adapter import (
        TrainConfig,
        save_checkpoint,
        train_for_task,
    )
    from depth_lens.tasks import get_task

    cfg_kwargs = dict(SIZE_CONFIGS[size])
    cfg = TrainConfig(
        steps=STEPS_BY_SIZE[size],
        batch_size=BATCH_BY_SIZE[size],
        **cfg_kwargs,
    )
    task = get_task(task_name)
    ckpt_path = CKPT_DIR / f"{size}_{task_name.replace('-', '_')}.pt"
    if ckpt_path.exists():
        return {
            "size": size, "task": task_name, "ckpt": str(ckpt_path),
            "status": "skipped_existing", "seconds": 0.0,
        }

    print(f"\n=== Training {size} on {task_name} ===", flush=True)
    t0 = time.time()
    adapter = train_for_task(task, cfg=cfg)
    dt = time.time() - t0
    save_checkpoint(adapter, ckpt_path)
    print(f"   saved {ckpt_path} ({dt/60:.1f} min)", flush=True)
    return {
        "size": size, "task": task_name, "ckpt": str(ckpt_path),
        "status": "trained", "seconds": dt,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="1M,10M,100M",
                   help="Comma-separated subset of: 1M,10M,100M")
    p.add_argument("--tasks", default=",".join(TASKS),
                   help="Comma-separated subset of bundled task names")
    p.add_argument("--all", action="store_true",
                   help="Full sweep (overrides --sizes and --tasks)")
    args = p.parse_args()

    if args.all:
        sizes = list(SIZE_CONFIGS)
        tasks = list(TASKS)
    else:
        sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"v2.0 training sweep — sizes={sizes}, tasks={tasks}")
    print(f"Output: {CKPT_DIR}\nLog: {LOG_PATH}\n")

    results: list[dict] = []
    for size in sizes:
        for task in tasks:
            try:
                r = train_one(size, task)
            except Exception as e:
                print(f"   FAILED {size}/{task}: {e}", flush=True)
                r = {"size": size, "task": task, "status": "failed", "error": str(e)}
            results.append(r)
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== Done ===")
    for r in results:
        status_tag = r["status"]
        sec = r.get("seconds", 0)
        print(f"  {r['size']:5s} {r['task']:18s} {status_tag:18s} {sec/60:6.1f} min")


if __name__ == "__main__":
    main()
