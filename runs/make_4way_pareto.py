"""4-way Pareto across paradigms — hosted APIs vs self-hosted vLLM
(Llama-3-8B-Instruct + DeepSeek-R1-Distill-Qwen-1.5B) on K-hop tiers 1-4.

Two panels:
  (1) Cost vs latency at tier 4 (mod-97 K=14): the production decision plot.
      API points use $/M-token billing; self-hosted points use $0.50/GPU-hr
      amortization. Same axis, both paradigms.
  (2) Self-hosted accuracy across tiers 1-4: shows the "self-hosted
      ceiling" — Llama-3-8B AWQ is great on tier 1 (0.94) and useless on
      tier 4 (0.00). DeepSeek-R1-Distill-Qwen-1.5B is the opposite,
      bad on easy / OK on hard. Production-relevant: pick the model
      whose ceiling sits above YOUR task's tier.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from depth_lens.pricing import DEFAULT_GPU_HOURLY_RATE, DEFAULT_PRICING

_DOCKER_ROOT = Path("/work")
ROOT = _DOCKER_ROOT if _DOCKER_ROOT.exists() else Path(__file__).resolve().parent.parent


def points_from_probe(path: Path, only_tier: int | None = None):
    d = json.load(open(path))
    name = d["adapter"]
    price = DEFAULT_PRICING.get(name)
    is_self_hosted = name.startswith(("vllm:", "hf:")) or name == "openmythos"
    out = []
    for di, depth in enumerate(d["depths"]):
        if only_tier is not None and depth != only_tier:
            continue
        for ci, c in enumerate(d["compute_grid"]):
            lat = d["latency_per_cell"][di][ci]
            acc = d["accuracy"][di][ci]
            if is_self_hosted:
                cost = lat * DEFAULT_GPU_HOURLY_RATE / 3600.0
            else:
                if price is None or not d.get("tokens_per_cell"):
                    continue
                t = d["tokens_per_cell"][di][ci]
                cost = (
                    t.get("input", 0) * price["input"]
                    + t.get("output", 0) * price["output"]
                ) / 1_000_000
            out.append({
                "model": name.split(":", 1)[-1] if ":" in name else name,
                "spec": name,
                "tier": depth,
                "budget": c["label"].split("=", 1)[-1],
                "cost": cost,
                "latency": lat,
                "accuracy": acc,
                "kind": "self-hosted" if is_self_hosted else "API",
            })
    return out


def points_from_compare(path: Path, vendor_prefix: str, tier: int):
    d = json.load(open(path))
    out = []
    for r in d:
        if not r["adapter"].startswith(vendor_prefix):
            continue
        if not r.get("tokens_per_cell") or tier not in r["depths"]:
            continue
        price = DEFAULT_PRICING.get(r["adapter"])
        if price is None:
            continue
        di = r["depths"].index(tier)
        for ci, c in enumerate(r["compute_grid"]):
            t = r["tokens_per_cell"][di][ci]
            lat = r["latency_per_cell"][di][ci]
            acc = r["accuracy"][di][ci]
            cost = (
                t.get("input", 0) * price["input"]
                + t.get("output", 0) * price["output"]
            ) / 1_000_000
            out.append({
                "model": r["adapter"].split(":", 1)[-1],
                "spec": r["adapter"],
                "tier": tier,
                "budget": c["label"].split("=", 1)[-1],
                "cost": cost,
                "latency": lat,
                "accuracy": acc,
                "kind": "API",
            })
    return out


VLLM_LLAMA = ROOT / "runs/vllm_llama3_8b.json"
VLLM_DEEPSEEK = ROOT / "runs/vllm_deepseek_r1_distill.json"


def short_name(model: str) -> str:
    return model.split("/")[-1]


def main():
    # ---------------- Panel 1: cost vs latency at tier 4 ----------------
    api_points = []
    for p in [
        ROOT / "runs/cvl_haiku.json",
        ROOT / "runs/cvl_sonnet.json",
        ROOT / "runs/cvl_opus.json",
        ROOT / "runs/cvl_g25f.json",
        ROOT / "runs/cvl_g25p.json",
        ROOT / "runs/cvl_g31f.json",
        ROOT / "runs/cvl_g31p.json",
    ]:
        if p.exists():
            api_points += points_from_probe(p, only_tier=4)
    api_points += points_from_compare(ROOT / "runs/openai_bench.json", "openai:", tier=4)

    sh_points = []
    for p in [VLLM_LLAMA, VLLM_DEEPSEEK]:
        if p.exists():
            sh_points += points_from_probe(p, only_tier=4)

    all_points = api_points + sh_points

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))

    family_colors = {
        "claude-haiku-4-5": "#3a9b5c",
        "claude-sonnet-4-6": "#cc4125",
        "claude-opus-4-7": "#6a3093",
        "o4-mini": "#0a7f8a",
        "gpt-5-mini": "#3a86ff",
        "gpt-5": "#1d3557",
        "gemini-2.5-flash": "#888",
        "gemini-2.5-pro": "#444",
        "gemini-3.1-flash-lite": "#1d8a4f",
        "gemini-3.1-pro-preview": "#ff8c00",
        "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4": "#e63946",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "#7209b7",
    }

    seen = set()
    for p in all_points:
        marker = "*" if p["kind"] == "self-hosted" else "o"
        size = 380 if p["kind"] == "self-hosted" else 110
        color = family_colors.get(p["model"], "#666")
        # Fade out below-accuracy points so the eye goes to working configs.
        alpha = 0.95 if p["accuracy"] >= 0.5 else 0.35
        label = short_name(p["model"]) if p["model"] not in seen else None
        ax1.scatter(
            p["cost"] * 1000, p["latency"],
            s=size, alpha=alpha,
            color=color, marker=marker,
            edgecolor="black", linewidth=0.7,
            label=label,
        )
        seen.add(p["model"])

    for p in sh_points:
        ax1.annotate(
            f"{short_name(p['model'])}\n{p['budget']} · acc={p['accuracy']:.2f}",
            xy=(p["cost"] * 1000, p["latency"]),
            xytext=(10, 6), textcoords="offset points",
            fontsize=8, color="#333", fontweight="bold",
        )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Cost ($ per 1000 calls) · log", fontsize=11)
    ax1.set_ylabel("Median latency (sec/call) · log", fontsize=11)
    ax1.set_title(
        "Tier 4 (mod-97 K=14) — production decision plot\n"
        "★ self-hosted (cost = latency × $0.50/GPU-hr) · ● API · faded = acc<0.5",
        fontsize=10, loc="left",
    )
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(loc="upper right", fontsize=6, ncol=2, framealpha=0.95)

    # ---------------- Panel 2: self-hosted accuracy by tier ----------------
    for path, color, marker, label in [
        (VLLM_LLAMA, "#e63946", "o", "Llama-3-8B-Instruct AWQ (max_tokens)"),
        (VLLM_DEEPSEEK, "#7209b7", "*", "DeepSeek-R1-Distill-Qwen-1.5B (reasoning_effort)"),
    ]:
        if not path.exists():
            continue
        d = json.load(open(path))
        # Use the best compute level per tier (peak accuracy across grid).
        peak_acc = []
        for di, depth in enumerate(d["depths"]):
            peak_acc.append(max(d["accuracy"][di]))
        ax2.plot(
            d["depths"], peak_acc,
            marker=marker, color=color, lw=2.2, ms=12,
            label=label,
        )
        for depth, acc in zip(d["depths"], peak_acc, strict=False):
            ax2.annotate(
                f"{acc:.2f}",
                xy=(depth, acc), xytext=(6, 6), textcoords="offset points",
                fontsize=8, color=color, fontweight="bold",
            )

    # API reference line: frontier APIs hit 1.00 on all 4 tiers.
    ax2.axhline(1.0, color="#888", lw=1, ls="--", alpha=0.7)
    ax2.text(1.0, 1.02, "frontier APIs (Haiku/Sonnet/Opus, gpt-5*, gemini-3.1*) — all 1.00",
             fontsize=8, color="#888")

    ax2.set_xlabel("K-hop tier (1=easy mod-13 K=3, 4=very hard mod-97 K=14)", fontsize=11)
    ax2.set_ylabel("Peak accuracy (across compute grid)", fontsize=11)
    ax2.set_xticks([1, 2, 3, 4])
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_title(
        "Self-hosted accuracy ceiling by task tier\n"
        "Production read: pick a model whose ceiling sits above YOUR task's tier",
        fontsize=10, loc="left",
    )
    ax2.grid(alpha=0.3)
    ax2.legend(loc="center left", fontsize=8)

    fig.suptitle(
        "depth-lens v1.2 — APIs vs self-hosted vLLM, one Pareto",
        fontsize=12, fontweight="bold", y=0.99,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out = ROOT / "docs/findings/figures/4way-pareto.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"\nPanel 1: {len(all_points)} points "
          f"({sum(1 for p in all_points if p['kind']=='self-hosted')} self-hosted, "
          f"{sum(1 for p in all_points if p['kind']=='API')} API)")


if __name__ == "__main__":
    main()
