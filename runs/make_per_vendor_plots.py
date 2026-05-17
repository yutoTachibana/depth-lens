"""Generate 3 per-vendor cost-vs-latency scatter plots from the cvl_*.json
re-bench data + the existing OpenAI K-hop bench."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from depth_lens.pricing import DEFAULT_PRICING

_DOCKER_ROOT = Path("/work")
ROOT = _DOCKER_ROOT if _DOCKER_ROOT.exists() else Path(__file__).resolve().parent.parent

def points_from_probe(path: Path):
    """Read a single-model probe JSON saved by `depth-lens probe`. Returns
    list of {model, budget_label, cost, latency, accuracy}."""
    d = json.load(open(path))
    name = d["adapter"]
    price = DEFAULT_PRICING.get(name)
    if price is None or not d.get("tokens_per_cell"):
        return []
    out = []
    for di, depth in enumerate(d["depths"]):
        for ci, c in enumerate(d["compute_grid"]):
            t = d["tokens_per_cell"][di][ci]
            lat = d["latency_per_cell"][di][ci]
            acc = d["accuracy"][di][ci]
            cost = (t.get("input", 0) * price["input"] + t.get("output", 0) * price["output"]) / 1_000_000
            out.append({
                "model": name.split(":")[-1],
                "budget_label": c["label"].split("=")[-1],
                "cost": cost,
                "latency": lat,
                "accuracy": acc,
            })
    return out


def points_from_compare(path: Path, vendor_prefix: str, tier: int):
    """Read a compare-style list JSON; filter by vendor prefix; only `tier`."""
    d = json.load(open(path))
    out = []
    for r in d:
        if not r["adapter"].startswith(vendor_prefix):
            continue
        if not r.get("tokens_per_cell"):
            continue
        price = DEFAULT_PRICING.get(r["adapter"])
        if price is None:
            continue
        if tier not in r["depths"]:
            continue
        di = r["depths"].index(tier)
        for ci, c in enumerate(r["compute_grid"]):
            t = r["tokens_per_cell"][di][ci]
            lat = r["latency_per_cell"][di][ci]
            acc = r["accuracy"][di][ci]
            cost = (t.get("input", 0) * price["input"] + t.get("output", 0) * price["output"]) / 1_000_000
            out.append({
                "model": r["adapter"].split(":")[-1],
                "budget_label": c["label"].split("=")[-1],
                "cost": cost,
                "latency": lat,
                "accuracy": acc,
            })
    return out


def make_plot(points, vendor_name, color_map, marker_map, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for p in points:
        ax.scatter(
            p["cost"] * 1000, p["latency"],
            s=200, alpha=0.85,
            color=color_map.get(p["model"], "#666"),
            marker=marker_map.get(p["model"], "o"),
            edgecolor="black", linewidth=0.7,
        )
        ax.annotate(
            f"{p['model']}\n{p['budget_label']}",
            xy=(p["cost"] * 1000, p["latency"]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=7, alpha=0.85,
        )

    # Pareto frontier highlight
    by_cost = sorted(points, key=lambda p: p["cost"])
    pareto = []
    min_lat = float("inf")
    for p in by_cost:
        if p["latency"] < min_lat:
            pareto.append(p)
            min_lat = p["latency"]
    if len(pareto) >= 2:
        xs = [p["cost"] * 1000 for p in pareto]
        ys = [p["latency"] for p in pareto]
        ax.plot(xs, ys, "--", color="goldenrod", lw=1.5, alpha=0.6, label="Pareto frontier")
        ax.legend(loc="upper right", fontsize=9)

    ax.set_xlabel("Cost ($ per 1000 calls)", fontsize=12)
    ax.set_ylabel("Median latency (sec/call)", fontsize=12)
    ax.set_title(f"{vendor_name} — K-hop tier 4 cost vs latency", fontsize=12, loc="left")
    ax.grid(alpha=0.3)

    # Below-axis acc note
    accs = [p["accuracy"] for p in points]
    if accs:
        acc_range = f"acc {min(accs):.2f}–{max(accs):.2f}"
        ax.text(0.02, -0.16, f"{len(points)} configs, {acc_range}, n=16 per cell  ·  "
                "mod-97 K=14  ·  pricing from DEFAULT_PRICING",
                transform=ax.transAxes, fontsize=8, color="#666")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


def main():
    # ANTHROPIC current gen
    anth_points = []
    for p in [ROOT / "runs/cvl_haiku.json", ROOT / "runs/cvl_sonnet.json", ROOT / "runs/cvl_opus.json"]:
        anth_points += points_from_probe(p)
    make_plot(
        anth_points, "Anthropic (Haiku 4.5 / Sonnet 4.6 / Opus 4.7)",
        color_map={"claude-haiku-4-5": "#3a9b5c", "claude-sonnet-4-6": "#cc4125", "claude-opus-4-7": "#6a3093"},
        marker_map={"claude-haiku-4-5": "o", "claude-sonnet-4-6": "s", "claude-opus-4-7": "^"},
        out_path=ROOT / "docs/findings/figures/cost-vs-latency-anthropic.png",
    )

    # OPENAI — from compare-style openai_bench.json
    oa_points = points_from_compare(ROOT / "runs/openai_bench.json", "openai:", tier=4)
    make_plot(
        oa_points, "OpenAI (o4-mini / gpt-5-mini / gpt-5)",
        color_map={"o4-mini": "#0a7f8a", "gpt-5-mini": "#3a86ff", "gpt-5": "#1d3557"},
        marker_map={"o4-mini": "o", "gpt-5-mini": "s", "gpt-5": "^"},
        out_path=ROOT / "docs/findings/figures/cost-vs-latency-openai.png",
    )

    # GEMINI current + 2.5
    g_points = []
    for p in [ROOT / "runs/cvl_g25f.json", ROOT / "runs/cvl_g25p.json",
              ROOT / "runs/cvl_g31f.json", ROOT / "runs/cvl_g31p.json"]:
        g_points += points_from_probe(p)
    make_plot(
        g_points, "Google Gemini (2.5 + 3.1 lines)",
        color_map={
            "gemini-2.5-flash": "#888",
            "gemini-2.5-pro": "#444",
            "gemini-3.1-flash-lite": "#1d8a4f",
            "gemini-3.1-pro-preview": "#3a86ff",
        },
        marker_map={
            "gemini-2.5-flash": "x",
            "gemini-2.5-pro": "D",
            "gemini-3.1-flash-lite": "o",
            "gemini-3.1-pro-preview": "^",
        },
        out_path=ROOT / "docs/findings/figures/cost-vs-latency-gemini.png",
    )

    print("\nAll 3 per-vendor plots written.")


if __name__ == "__main__":
    main()
