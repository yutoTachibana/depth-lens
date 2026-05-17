"""OpenMythos: accuracy and latency vs n_loops — the original looped-transformer
"pay latency for accuracy" claim, measured directly."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DOCKER_ROOT = Path("/work")
ROOT = _DOCKER_ROOT if _DOCKER_ROOT.exists() else Path(__file__).resolve().parent.parent

d = json.load(open(ROOT / "runs/openmythos_loops.json"))
loops = [c["value"] for c in d["compute_grid"]]

fig, (ax_acc, ax_lat) = plt.subplots(1, 2, figsize=(12, 4.5))

colors = ["#0a7f8a", "#cc4125", "#6a3093"]

for di, (depth, color) in enumerate(zip(d["depths"], colors, strict=False)):
    accs = d["accuracy"][di]
    lats = [x * 1000 for x in d["latency_per_cell"][di]]  # ms
    ax_acc.plot(loops, accs, "o-", color=color, lw=2, ms=8, label=f"depth={depth}")
    ax_lat.plot(loops, lats, "o-", color=color, lw=2, ms=8, label=f"depth={depth}")

# Mark training depth
for ax in (ax_acc, ax_lat):
    ax.axvline(4, color="gray", ls=":", alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(loops, [str(x) for x in loops])
    ax.set_xlabel("n_loops (inference-time recurrent depth)")
    ax.grid(alpha=0.3, which="both")

ax_acc.text(4.2, 0.02, "training\nmax_loop_iters=4", fontsize=8, color="gray")
ax_acc.set_ylabel("accuracy")
ax_acc.set_ylim(0, 1.05)
ax_acc.set_title(
    "Accuracy peaks AT training depth and degrades past it",
    fontsize=11, loc="left",
)
ax_acc.legend(loc="lower right", fontsize=9)

ax_lat.text(4.2, ax_lat.get_ylim()[1] * 0.85, "training\nmax_loop_iters=4",
            fontsize=8, color="gray")
ax_lat.set_ylabel("latency (ms / pred)")
ax_lat.set_title(
    "Latency scales roughly linearly with n_loops, as expected",
    fontsize=11, loc="left",
)

plt.suptitle(
    "OpenMythos: the 'pay latency for accuracy' trade has a saturation point.\n"
    "Looped accuracy gains stop at training_max_loop_iters; latency keeps growing.",
    fontsize=11, fontweight="bold",
)

plt.tight_layout(rect=(0, 0, 1, 0.91))
out = ROOT / "docs/findings/figures/openmythos-loops-acc-vs-latency.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"wrote {out}")
