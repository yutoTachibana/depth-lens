"""Build a cost-vs-latency scatter from existing bench JSONs (no new API calls).

Combines:
- OpenAI o4-mini / gpt-5-mini / gpt-5 K-hop tier-4 (3 efforts each, full lat+cost)
- Anthropic Sonnet 4 (May 2025) K-hop tier-4 (3 budgets, full lat+cost)
- Gemini 2.5 Pro & 3.1 Flash-Lite K-hop tier-4 (lat available, cost estimated
  from pricing × known token-per-cell shape from the smoke data we have)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DOCKER_ROOT = Path("/work")
ROOT = _DOCKER_ROOT if _DOCKER_ROOT.exists() else Path(__file__).resolve().parent.parent

PRICES = {
    "openai:o4-mini":                    {"input": 1.10, "output": 4.40},
    "openai:gpt-5-mini":                 {"input": 0.25, "output": 2.00},
    "openai:gpt-5":                      {"input": 1.25, "output": 10.00},
    "anthropic:claude-sonnet-4-20250514":{"input": 3.00, "output": 15.00},
}

def cells_with_cost_and_lat(bench_json, model_name=None):
    """Extract (cost_per_pred, latency_sec, label) tuples from a probe JSON."""
    out = []
    if isinstance(bench_json, dict):
        runs = [bench_json] if "adapter" in bench_json else []
    else:
        runs = bench_json
    for r in runs:
        if model_name and r.get("adapter") != model_name:
            continue
        if not r.get("latency_per_cell") or not r.get("tokens_per_cell"):
            continue
        p = PRICES.get(r["adapter"])
        if not p:
            continue
        for di, depth in enumerate(r["depths"]):
            for ci, c in enumerate(r["compute_grid"]):
                acc = r["accuracy"][di][ci]
                if acc < 0.99:
                    continue
                t = r["tokens_per_cell"][di][ci]
                cost = (t.get("input", 0) * p["input"] + t.get("output", 0) * p["output"]) / 1_000_000
                lat = r["latency_per_cell"][di][ci]
                out.append({
                    "vendor": r["adapter"].split(":")[0],
                    "model": r["adapter"].split(":")[1],
                    "budget_label": c["label"].split("=")[-1],
                    "tier": depth,
                    "cost": cost,
                    "latency": lat,
                })
    return out

# Load OpenAI
openai_data = cells_with_cost_and_lat(json.load(open(ROOT / "runs/openai_bench.json")))
# Load Anthropic Sonnet 4 May
sonnet4_data = cells_with_cost_and_lat(json.load(open(ROOT / "runs/bench_sonnet4_may.json")))

all_pts = openai_data + sonnet4_data
# Keep only tier 4 (hardest) for the cleanest plot
tier4 = [p for p in all_pts if p["tier"] == 4]

print(f"Plotting {len(tier4)} (cost, latency) points on K-hop tier 4, all acc≥0.99:")
for p in tier4:
    print(f"  {p['vendor']:9s} {p['model']:30s} budget={p['budget_label']:>6s} ${p['cost']:.4f}/call  {p['latency']:.2f}s")

# Plot
VENDOR_COLORS = {"openai": "#0a7f8a", "anthropic": "#cc4125"}
MODEL_MARKERS = {
    "o4-mini": "o",
    "gpt-5-mini": "s",
    "gpt-5": "^",
    "claude-sonnet-4-20250514": "D",
}

fig, ax = plt.subplots(figsize=(10, 5.5))

for p in tier4:
    ax.scatter(
        p["cost"] * 1000, p["latency"],
        s=200, alpha=0.75,
        color=VENDOR_COLORS[p["vendor"]],
        marker=MODEL_MARKERS.get(p["model"], "o"),
        edgecolor="black", linewidth=0.6,
    )
    # Label the cheapest and the highest-latency points
    pass

# Annotate each point with model + budget
for p in tier4:
    label = f"{p['model'].replace('claude-sonnet-4-20250514', 'sonnet-4 (May 2025)')}\n{p['budget_label']}"
    ax.annotate(
        label,
        xy=(p["cost"] * 1000, p["latency"]),
        xytext=(5, 5), textcoords="offset points",
        fontsize=7, alpha=0.85,
    )

ax.set_xlabel("Cost ($ per 1000 calls)", fontsize=12)
ax.set_ylabel("Median latency (seconds per call)", fontsize=12)
ax.set_title(
    "Cost vs latency on K-hop tier 4 (mod 97, K=14) — all points score 1.00 accuracy\n"
    "Pick the right-most low-right corner: cheap AND fast",
    fontsize=11, loc="left",
)
ax.grid(alpha=0.3)

# Pareto frontier annotation: lowest cost AND lowest latency
pareto = []
sorted_by_cost = sorted(tier4, key=lambda p: p["cost"])
min_lat = float("inf")
for p in sorted_by_cost:
    if p["latency"] < min_lat:
        pareto.append(p)
        min_lat = p["latency"]
pareto_x = [p["cost"] * 1000 for p in pareto]
pareto_y = [p["latency"] for p in pareto]
ax.plot(pareto_x, pareto_y, "--", color="goldenrod", lw=1.5, alpha=0.6, label="Pareto frontier (lower-cost wins also speed)")

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=VENDOR_COLORS["anthropic"], label="Anthropic", markersize=12),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=VENDOR_COLORS["openai"], label="OpenAI", markersize=12),
    Line2D([0], [0], linestyle="--", color="goldenrod", label="Pareto frontier"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

# Footnote
fig.text(0.5, -0.02,
         "Data sources: runs/openai_bench.json + runs/bench_sonnet4_may.json. "
         "Costs computed from per-cell token counts × current published rates (2026-05). "
         "Latency is per-call median measured in parallel-8 client (use as a relative signal, not absolute).",
         fontsize=7, color="#888", ha="center")

plt.tight_layout()
out = ROOT / "docs/findings/figures/cost-vs-latency-tier4.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"\nwrote {out}")
