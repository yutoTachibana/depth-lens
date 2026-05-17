"""OpenMythos vs Claude Haiku/Sonnet on built-in k-hop.

Both architectures plot on the same accuracy-vs-latency axes. depth-lens's
'inference-time-compute meter' use case made literal."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DOCKER_ROOT = Path("/work")
ROOT = _DOCKER_ROOT if _DOCKER_ROOT.exists() else Path(__file__).resolve().parent.parent

# Load API data (3 budgets × 4 depths each, accuracy + latency per cell)
haiku = json.load(open(ROOT / "runs/h2h_haiku.json"))
sonnet = json.load(open(ROOT / "runs/h2h_sonnet.json"))

# Load OpenMythos data (6 loop counts × 3 depths)
om = json.load(open(ROOT / "runs/openmythos_loops.json"))

# Build a (depth → list of (latency, accuracy, label)) per architecture.
def cells_by_depth(d):
    out = {}
    for di, depth in enumerate(d["depths"]):
        out[depth] = []
        for ci, c in enumerate(d["compute_grid"]):
            lat = d["latency_per_cell"][di][ci] if d["latency_per_cell"] else 0
            acc = d["accuracy"][di][ci]
            label = c["label"].split("=")[-1]
            out[depth].append((lat, acc, label))
    return out

haiku_by_d = cells_by_depth(haiku)
sonnet_by_d = cells_by_depth(sonnet)
om_by_d = cells_by_depth(om)

# Plot per-depth panels. Show only depths where we have data on both sides.
common_depths = sorted(set(haiku_by_d) & set(om_by_d))
print(f"Common depths: {common_depths}")

fig, axes = plt.subplots(1, len(common_depths), figsize=(5 * len(common_depths), 4.5), sharey=True, squeeze=False)
axes = axes[0]

for ax, depth in zip(axes, common_depths, strict=False):
    # Haiku: connect by increasing budget
    pts = haiku_by_d.get(depth, [])
    if pts:
        lats = [p[0] for p in pts]
        accs = [p[1] for p in pts]
        ax.plot(lats, accs, "o-", color="#3a9b5c", lw=2, ms=9, label="Claude Haiku 4.5 (budget sweep)")
        for (lat, acc, lab) in pts:
            ax.annotate(lab, xy=(lat, acc), xytext=(5, -10), textcoords="offset points",
                        fontsize=7, alpha=0.8, color="#3a9b5c")

    # Sonnet
    pts = sonnet_by_d.get(depth, [])
    if pts:
        lats = [p[0] for p in pts]
        accs = [p[1] for p in pts]
        ax.plot(lats, accs, "s-", color="#cc4125", lw=2, ms=9, label="Claude Sonnet 4.6 (budget sweep)")
        for (lat, acc, lab) in pts:
            ax.annotate(lab, xy=(lat, acc), xytext=(5, -10), textcoords="offset points",
                        fontsize=7, alpha=0.8, color="#cc4125")

    # OpenMythos: connect by increasing n_loops
    pts = om_by_d.get(depth, [])
    if pts:
        lats = [p[0] for p in pts]
        accs = [p[1] for p in pts]
        ax.plot(lats, accs, "^--", color="#3a86ff", lw=2, ms=9,
                label="OpenMythos 925K params (n_loops sweep)")
        for (lat, acc, lab) in pts:
            ax.annotate(lab, xy=(lat, acc), xytext=(5, 5), textcoords="offset points",
                        fontsize=7, alpha=0.8, color="#3a86ff")

    ax.set_xscale("log")
    ax.set_xlabel("median latency per call (sec)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3, which="both")
    ax.set_title(f"K-hop depth = {depth}")

axes[0].set_ylabel("accuracy")
axes[-1].legend(loc="lower right", fontsize=8)

fig.suptitle(
    "Two paradigms, one accuracy-vs-latency plot — depth-lens as inference-time-compute meter\n"
    "OpenMythos (latent recursion) vs Claude (token-level CoT) on built-in k-hop",
    fontsize=11, fontweight="bold",
)
fig.text(0.5, 0.005,
         "Latency is per-call median wall-clock. OpenMythos is local (small model on a single GPU); "
         "Claude is over the network — absolute latency comparison is not fair, the SHAPE of each curve is what matters.",
         fontsize=7, color="#666", ha="center")

plt.tight_layout(rect=(0, 0.025, 1, 0.92))
out = ROOT / "docs/findings/figures/architecture-comparison-h2h.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"wrote {out}")
