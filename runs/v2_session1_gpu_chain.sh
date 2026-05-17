#!/bin/bash
# v2.0 Session 1 — chained GPU training sequence.
#
# Runs sequentially (GPU is shared, can't parallelize):
#   1. 1M training × 5 tasks  (~25 min)
#   2. 10M training × 5 tasks (~3-4 hr)
#   3. 100M training × k-hop  (~3-4 hr)
#
# After this completes, Session 1 ends. The remaining 100M jobs
# (parity, state-tracking, mini-csp, dict-lookup) get queued by
# v2_overnight.sh which the user kicks off before bed.
#
# Usage (from host):
#   docker run -d --name dl-v2-session1 --gpus all \
#     -v //c/Users/burger/claude/depth-lens:/work -w //work \
#     depth-lens:gpu bash runs/v2_session1_gpu_chain.sh
#
# To skip a step (e.g. if 1M is already done), comment out the
# corresponding line.

set -e

cd /work

echo "[$(date)] === v2.0 Session 1 GPU chain starting ==="

# Phase 1: 1M sweep (skips any task whose ckpt already exists)
echo "[$(date)] Phase 1/3: 1M × 5 tasks"
python runs/v2_train_sweep.py --sizes 1M \
    --tasks k-hop,parity,state-tracking,mini-csp,dict-lookup

# Phase 2: 10M sweep
echo "[$(date)] Phase 2/3: 10M × 5 tasks"
python runs/v2_train_sweep.py --sizes 10M \
    --tasks k-hop,parity,state-tracking,mini-csp,dict-lookup

# Phase 3: 100M × k-hop only (canonical task; others go overnight)
echo "[$(date)] Phase 3/3: 100M × k-hop"
python runs/v2_train_sweep.py --sizes 100M --tasks k-hop

echo "[$(date)] === Session 1 GPU chain COMPLETE ==="
echo "Next: run v2_overnight.sh for the remaining 100M jobs."
