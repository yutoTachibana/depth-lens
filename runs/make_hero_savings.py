"""Render the cost-savings hero plot for README — one visual, one message."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Numbers from docs/findings/v1.0-cost-savings.md
# K-hop tier 4 (mod 97, K=14), 10,000 calls/day, accuracy = 1.00 across all rows.
rows = [
    ("Anthropic Opus 4.7\n@ thinking_budget=16384 (default-max)",   126655, "#cc4125"),
    ("Anthropic Opus 4.7\n@ thinking_budget=1024",                  75190,  "#e07b6a"),
    ("Anthropic Sonnet 4.6\n@ thinking_budget=1024",                18250,  "#888"),
    ("Anthropic Haiku 4.5\n@ thinking_budget=1024",                 3650,   "#1d8a4f"),
]

labels = [r[0] for r in rows]
costs = [r[1] for r in rows]
colors = [r[2] for r in rows]

fig, ax = plt.subplots(figsize=(11, 5.3))

bars = ax.barh(range(len(labels)), costs, color=colors, edgecolor="black", linewidth=0.6)
ax.set_yticks(range(len(labels)), labels, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel("USD / year (10,000 calls/day)", fontsize=12)
ax.set_xlim(0, 145000)

# Number labels at end of each bar
for bar, v in zip(bars, costs, strict=False):
    ax.text(v + 2200, bar.get_y() + bar.get_height() / 2,
            f"${v:,}", va="center", fontsize=12, fontweight="bold")

# Headline annotation
ax.text(0.5, 1.15,
        "All four rows score 1.00 accuracy on K-hop tier 4 (mod 97, K=14).",
        transform=ax.transAxes, ha="center", fontsize=11, style="italic", color="#555")
ax.text(0.5, 1.06,
        "depth-lens recommends Haiku 4.5 @ budget=1024  →  saves $123,005/year (97%)",
        transform=ax.transAxes, ha="center", fontsize=14, fontweight="bold")

# Annotation arrow
ax.annotate("", xy=(8000, 3), xytext=(125000, 0),
            arrowprops=dict(arrowstyle="->", color="black", lw=2))
ax.text(60000, 1.6, "switch saves\n$123k/year", fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff5cc", edgecolor="black"))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.3, axis="x")

# Footnote
fig.text(0.5, 0.005,
         "Numbers derived from real depth-lens probe data on K-hop modular composition. "
         "Opus per-call cost first-class measured; Sonnet/Haiku estimated from token counts × Anthropic published rates. "
         "See docs/findings/v1.0-cost-savings.md for sources.",
         fontsize=7, color="#888", ha="center")

plt.tight_layout(rect=(0, 0.025, 1, 0.88))

import pathlib
_DOCKER_ROOT = pathlib.Path("/work")
_ROOT = _DOCKER_ROOT if _DOCKER_ROOT.exists() else pathlib.Path(__file__).resolve().parent.parent
out = _ROOT / "docs/findings/figures/hero-cost-savings.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"wrote {out}")
