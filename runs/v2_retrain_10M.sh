#!/bin/bash
# Retrain 10M at the bumped 8000-step budget (1024K sequences, parity
# with 1M). The original 10M ckpts saw 768K sequences and converged on
# these tasks, but for a clean "all sizes saw ≥1M samples" claim in the
# v2.0 scaling-law plot we re-train.
#
# After re-training, re-runs the OpenMythos probe sweep ONLY for the
# 10M cells.

set -e

cd /work
export PYTHONPATH=/work

echo "[$(date)] === 10M retrain @ 8000 steps starting ==="

# Delete the old 10M checkpoints + probe JSONs so the sweep regenerates.
rm -f runs/v2_ckpts/10M_*.pt
rm -f runs/v2_openmythos/10M__*.json

python runs/v2_train_sweep.py --sizes 10M \
    --tasks k-hop,parity,state-tracking,mini-csp,dict-lookup

echo "[$(date)] All 10M training done. Probing 10M cells."

python runs/v2_openmythos_probe.py --sizes 10M

echo "[$(date)] === 10M retrain COMPLETE ==="
