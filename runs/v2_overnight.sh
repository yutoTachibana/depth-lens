#!/bin/bash
# v2.0 overnight training queue.
#
# To be invoked from inside the depth-lens:gpu docker container at the
# end of Session 1 (after 1M sweep, 10M sweep, and the first 100M run
# on K-hop have all completed). This runs the remaining 100M jobs
# (parity, state-tracking, mini-csp, dict-lookup) sequentially so the
# scaling-law plot has all 3 sizes × all 5 tasks by Session 2.
#
# Usage from host (Windows / WSL):
#   docker run -d --name dl-v2-overnight --gpus all \
#     -v //c/Users/burger/claude/depth-lens:/work -w //work \
#     depth-lens:gpu bash runs/v2_overnight.sh
#
# Estimated total runtime: ~12-16 hours of GPU time on a 4080 SUPER.

set -e

cd /work

echo "[$(date)] v2.0 overnight queue starting"

for TASK in parity state-tracking mini-csp dict-lookup; do
    echo "[$(date)] Training 100M on $TASK"
    python runs/v2_train_sweep.py --sizes 100M --tasks "$TASK"
done

echo "[$(date)] All 100M training done. Running OpenMythos probe sweep."

python runs/v2_openmythos_probe.py

echo "[$(date)] v2.0 overnight queue COMPLETE"
