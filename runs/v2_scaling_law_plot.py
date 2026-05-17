"""
v2.0 killer artifact: 3-paradigm scaling-law plot.

For each of 5 tasks, one panel. Within a panel:
  - OpenMythos sizes (1M / 10M / 100M) × n_loops grid → blue line + markers
  - Self-hosted vLLM models (Llama AWQ, DeepSeek-R1-Distill 1.5B)  → orange markers
  - Token-CoT APIs (gpt-5-mini, o4-mini, gemini-3.1-flash-lite)   → green/red markers

x-axis: log(FLOPs/inference)
y-axis: accuracy (at the task's largest probed depth)

The story the plot is meant to tell: each paradigm has a sweet spot on the
FLOPs axis, and depth-lens is the only OSS tool that lets you see them on
one chart for your task class.

Usage:
    python runs/v2_scaling_law_plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from depth_lens.flops import estimate_flops_per_call, paradigm_of

_DOCKER_ROOT = Path("/work")
ROOT = _DOCKER_ROOT if _DOCKER_ROOT.exists() else Path(__file__).resolve().parent.parent

OM_DIR = ROOT / "runs" / "v2_openmythos"
API_DIR = ROOT / "runs" / "v2_api"
VLLM_DIR = ROOT / "runs" / "v2_vllm"

TASKS = ("k-hop", "parity", "state-tracking", "mini-csp", "dict-lookup")

# Colors per paradigm
COLOR = {
    "looped":      "#3a86ff",
    "self_hosted": "#ff8c00",
    "token_cot":   "#3a9b5c",
}
MARKER = {
    "looped":      "o",
    "self_hosted": "*",
    "token_cot":   "s",
}


def load_openmythos_points(task: str) -> list[dict]:
    """For each (size, n_loops) cell, compute (flops, peak_accuracy_over_depths).

    We collapse depths by taking the max accuracy across depths within the
    probed range — this matches the "what's the best this model can do on
    this task" question the scaling plot is asking."""
    out = []
    for size in ("1M", "10M", "100M"):
        path = OM_DIR / f"{size}__{task}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        n_params = d.get("n_params")
        if n_params is None:
            continue
        # For each n_loops, find best accuracy across depths.
        for ci, c in enumerate(d["compute_grid"]):
            n_loops = int(c["value"])
            accs = [d["accuracy"][di][ci] for di in range(len(d["depths"]))]
            best_acc = max(accs)
            # Rough token estimate per call: shortest depth's prompt length.
            tok = (d.get("tokens_per_cell") or [[None]])[0][ci]
            # OpenMythos uses single-token output; we estimate input tokens
            # from the longest prompt at the largest depth.
            input_tok = 20 + d["depths"][-1] * 4   # heuristic, fine for FLOPs ranking
            output_tok = 1
            f = estimate_flops_per_call(
                "openmythos",
                input_tokens=input_tok, output_tokens=output_tok,
                openmythos_params=n_params, n_loops=n_loops,
            )
            out.append({
                "label": f"{size}/loops={n_loops}",
                "size": size,
                "n_loops": n_loops,
                "flops": f["flops"],
                "accuracy": best_acc,
                "paradigm": "looped",
            })
    return out


def load_api_points(task: str) -> list[dict]:
    out = []
    for path in API_DIR.glob(f"{task}__*.json"):
        d = json.loads(path.read_text())
        spec = d.get("adapter")
        if spec is None:
            continue
        for ci, c in enumerate(d["compute_grid"]):
            accs = [d["accuracy"][di][ci] for di in range(len(d["depths"]))]
            best_acc = max(accs)
            # Use average tokens across depths
            if d.get("tokens_per_cell"):
                cells = [d["tokens_per_cell"][di][ci] for di in range(len(d["depths"]))]
                avg_in = sum(c.get("input", 0) for c in cells) / max(len(cells), 1)
                avg_out = sum(c.get("output", 0) for c in cells) / max(len(cells), 1)
            else:
                avg_in, avg_out = 100, 200
            f = estimate_flops_per_call(spec, input_tokens=int(avg_in), output_tokens=int(avg_out))
            if f["flops"] is None:
                continue
            short_label = spec.split(":", 1)[-1]
            out.append({
                "label": f"{short_label}/{c['label']}",
                "spec": spec,
                "flops": f["flops"],
                "accuracy": best_acc,
                "paradigm": paradigm_of(spec),
            })
    return out


def load_vllm_points(task: str) -> list[dict]:
    out = []
    for path in VLLM_DIR.glob(f"{task}__*.json"):
        d = json.loads(path.read_text())
        spec = d.get("adapter")
        if spec is None:
            continue
        for ci, c in enumerate(d["compute_grid"]):
            accs = [d["accuracy"][di][ci] for di in range(len(d["depths"]))]
            best_acc = max(accs)
            if d.get("tokens_per_cell"):
                cells = [d["tokens_per_cell"][di][ci] for di in range(len(d["depths"]))]
                avg_in = sum(c.get("input", 0) for c in cells) / max(len(cells), 1)
                avg_out = sum(c.get("output", 0) for c in cells) / max(len(cells), 1)
            else:
                avg_in, avg_out = 100, 300
            f = estimate_flops_per_call(spec, input_tokens=int(avg_in), output_tokens=int(avg_out))
            if f["flops"] is None:
                continue
            short_label = spec.split("/")[-1][:32]
            out.append({
                "label": f"{short_label}/{c['label']}",
                "spec": spec,
                "flops": f["flops"],
                "accuracy": best_acc,
                "paradigm": "self_hosted",
            })
    return out


def main():
    n_tasks = len(TASKS)
    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 4.5), squeeze=False)
    axes = axes[0]

    legend_seen: set[str] = set()
    for ax, task in zip(axes, TASKS, strict=False):
        om = load_openmythos_points(task)
        api = load_api_points(task)
        vllm = load_vllm_points(task)

        # Connect OpenMythos points within the same size by n_loops order.
        for size in ("1M", "10M", "100M"):
            pts = sorted([p for p in om if p["size"] == size], key=lambda p: p["n_loops"])
            if pts:
                ax.plot(
                    [p["flops"] for p in pts],
                    [p["accuracy"] for p in pts],
                    marker="o", lw=1.5, ms=8, alpha=0.85,
                    color=COLOR["looped"],
                    label=f"OpenMythos {size} (n_loops sweep)" if f"om-{size}" not in legend_seen else None,
                )
                legend_seen.add(f"om-{size}")

        for pts, label_key, marker_key in [(api, "API", "token_cot"), (vllm, "vLLM self-hosted", "self_hosted")]:
            if pts:
                ax.scatter(
                    [p["flops"] for p in pts],
                    [p["accuracy"] for p in pts],
                    marker=MARKER[marker_key], s=110, alpha=0.85,
                    color=COLOR[marker_key], edgecolor="black", linewidth=0.5,
                    label=label_key if label_key not in legend_seen else None,
                )
                legend_seen.add(label_key)

        ax.set_xscale("log")
        ax.set_xlabel("FLOPs per inference (log scale)", fontsize=10)
        ax.set_ylabel("Peak accuracy on task" if task == TASKS[0] else "")
        ax.set_title(f"Task: {task}", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        "Inference-time compute scaling laws across 3 paradigms — depth-lens v2.0\n"
        "Looped (OpenMythos) ● — Self-hosted vLLM ★ — Token-CoT API ■",
        fontsize=12, fontweight="bold",
    )
    fig.text(0.5, 0.005,
             "FLOPs estimated from 2N(T_in+T_out) × n_loops. "
             "API params are community estimates good to ~2×; vLLM and OpenMythos counts are exact. "
             "Accuracy = max across probed depths per (model, compute) cell.",
             fontsize=8, color="#666", ha="center")

    # One combined legend at the top-right axis.
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l, strict=False):
            if li and li not in labels:
                handles.append(hi)
                labels.append(li)
    axes[-1].legend(handles, labels, loc="lower right", fontsize=7, framealpha=0.95)

    plt.tight_layout(rect=(0, 0.025, 1, 0.92))
    out = ROOT / "docs" / "findings" / "figures" / "v2.0-scaling-law.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
