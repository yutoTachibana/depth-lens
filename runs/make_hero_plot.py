"""Render the v1.0 hero plot: two stories side by side.

Top panel:    K-hop tier 4 acc across 2025-era cheap reasoning models.
              Anthropic Sonnet 4 / OpenAI o3-mini are at ceiling;
              Gemini 2.5 Flash collapses; Gemini 3.1 Flash-Lite recovers.
Bottom panel: Haiku 4.5 on mini-CSP at depth 9 — collapses at default
              budget, recovers when you 4x it.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_DOCKER_ROOT = Path("/work")
ROOT = _DOCKER_ROOT if _DOCKER_ROOT.exists() else Path(__file__).resolve().parent.parent

# --- top panel: 2025-era cross-vendor K-hop tier 4 ---
TOP = [
    ("Anthropic\nSonnet 4 (May 2025)",       ROOT / "runs/bench_sonnet4_may.json"),
    ("OpenAI\no3-mini (Jan 2025)",            ROOT / "runs/bench_o3mini.json"),
    ("Google\nGemini 2.5 Flash (Mar 2025)",   ROOT / "runs/csp_g25flash.json"),  # placeholder, replaced below
    ("Google\nGemini 3.1 Flash-Lite (current)",ROOT / "runs/csp_g31flash.json"),
]

# Need K-hop tier-4 data; Gemini 2.5 Flash is in the Gemini 2.5 bench file.
gemini_bench = json.load(open(ROOT / "runs/gemini_bench.json"))
g25_flash = next(r for r in gemini_bench if r["adapter"] == "gemini:gemini-2.5-flash")

# Replace placeholders for Gemini with the proper K-hop tier-4 acc at low budget
g25_flash_tier4_lo = g25_flash["accuracy"][3][0]  # tier 4, lowest budget
gemini3_bench = json.load(open(ROOT / "runs/gemini3_bench.json"))
g31_flash_lite = next(r for r in gemini3_bench if r["adapter"] == "gemini:gemini-3.1-flash-lite")
g31_lite_tier4_lo = g31_flash_lite["accuracy"][3][0]

sonnet4 = json.load(open(ROOT / "runs/bench_sonnet4_may.json"))
sonnet4_tier4_lo = sonnet4["accuracy"][3][0]
o3mini = json.load(open(ROOT / "runs/bench_o3mini.json"))
o3mini_tier4_lo = o3mini["accuracy"][3][0]  # OpenAI uses effort=low

top_labels = [
    "Anthropic\nSonnet 4\n(May 2025)",
    "OpenAI\no3-mini\n(Jan 2025)",
    "Google\nGemini 2.5 Flash\n(Mar 2025)",
    "Google\nGemini 3.1 Flash-Lite\n(May 2026 GA)",
]
top_vals = [sonnet4_tier4_lo, o3mini_tier4_lo, g25_flash_tier4_lo, g31_lite_tier4_lo]
top_colors = ["#0a7f8a", "#0a7f8a", "#cc4125", "#3a86ff"]

# --- bottom panel: Haiku 4.5 on mini-CSP at depth 9 across budgets ---
csp_haiku = json.load(open(ROOT / "runs/csp_haiku.json"))
d9_idx = csp_haiku["depths"].index(9)
csp_budgets = [c["label"].split("=")[-1] for c in csp_haiku["compute_grid"]]
csp_accs = csp_haiku["accuracy"][d9_idx]

# Build figure
fig = plt.figure(figsize=(12, 7))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.55)

ax_top = fig.add_subplot(gs[0])
bars = ax_top.bar(range(len(top_labels)), top_vals, color=top_colors, edgecolor="black", linewidth=0.5)
ax_top.set_xticks(range(len(top_labels)), top_labels, fontsize=9)
ax_top.set_ylabel("accuracy")
ax_top.set_ylim(0, 1.10)
ax_top.set_title(
    "Story 1 — In early-mid 2025, only Gemini Flash collapsed on hard K-hop.\n"
    "Anthropic & OpenAI's same-era cheap reasoning models were already at ceiling.",
    fontsize=11, loc="left", pad=10,
)
ax_top.axhline(0.5, ls=":", color="gray", lw=1)
ax_top.text(-0.45, 0.52, "chance", fontsize=8, color="gray")
for bar, v in zip(bars, top_vals, strict=False):
    ax_top.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=11, fontweight="bold")
ax_top.grid(alpha=0.3, axis="y")
ax_top.text(2, -0.20, "← K-hop tier 4 (mod 97, K=14), cheapest budget, n=16 per cell",
            fontsize=8, color="gray", ha="center")

ax_bot = fig.add_subplot(gs[1])
bot_colors = ["#cc4125", "#888", "#444"]
bars = ax_bot.bar(csp_budgets, csp_accs, color=bot_colors, edgecolor="black", linewidth=0.5)
for bar, v in zip(bars, csp_accs, strict=False):
    ax_bot.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=11, fontweight="bold")
ax_bot.set_xlabel("thinking_budget_tokens")
ax_bot.set_ylabel("accuracy")
ax_bot.set_ylim(0, 1.15)
ax_bot.set_title(
    "Story 2 — Claude Haiku 4.5 (current cheap tier) collapses on hard 2-SAT at default budget,\n"
    "but a 4× budget bump fully recovers it. Actionable: `budget≥4096` for CSP-style tasks.",
    fontsize=11, loc="left", pad=10,
)
ax_bot.axhline(0.5, ls=":", color="gray", lw=1)
ax_bot.grid(alpha=0.3, axis="y")
ax_bot.text(1, -0.20, "← mini-CSP depth=9 (2-SAT, 9 vars, ⌈1.5n⌉ clauses), n=12 per cell",
            fontsize=8, color="gray", ha="center")

out = ROOT / "docs/findings/figures/hero-v1.0.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"wrote {out}")
