#!/bin/bash
# Retrain 100M at the bumped 24000-step budget (parity with 10M's
# total sample count, ~768K sequences each). The original 100M run
# saw only 256K sequences and was severely under-trained — see the
# v2.0 finding doc for the data.
#
# After re-training, re-runs the OpenMythos probe sweep ONLY for the
# 100M cells (the 1M/10M JSONs from the original run are still valid
# and get skipped by v2_openmythos_probe.py's skip-existing logic).

set -e

cd /work
export PYTHONPATH=/work

echo "[$(date)] === 100M retrain @ 24000 steps starting ==="

python runs/v2_train_sweep.py --sizes 100M \
    --tasks k-hop,parity,state-tracking,mini-csp,dict-lookup

echo "[$(date)] All 100M training done. Probing 100M cells."

python runs/v2_openmythos_probe.py --sizes 100M

echo "[$(date)] === 100M retrain COMPLETE ==="
