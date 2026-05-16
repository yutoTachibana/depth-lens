"""
Streamlit dashboard for depth-lens.

Loads probe result JSONs from a cache directory (default: the depth-lens cache
under ~/.cache/depth-lens/probes/) and lets the user explore them:

  - filter by adapter / task
  - view accuracy-vs-compute curves with CIs
  - view (depth × compute) heatmaps
  - see overthinking / effective-depth diagnostics

Launch with:
    depth-lens dashboard
or:
    streamlit run depth_lens/dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from depth_lens.cache import cache_dir as default_cache_dir
from depth_lens.metrics import ProbeResult


def _load_all(cache_directory: Path) -> list[tuple[str, ProbeResult]]:
    """Return list of (filename, ProbeResult) tuples for every cached probe."""
    if not cache_directory.exists():
        return []
    results: list[tuple[str, ProbeResult]] = []
    for path in sorted(cache_directory.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        from depth_lens.adapters.base import ComputeLevel

        results.append(
            (
                path.name,
                ProbeResult(
                    task_name=data["task_name"],
                    adapter_name=data["adapter_name"],
                    compute_axis=data["compute_axis"],
                    depths=data["depths"],
                    compute_grid=[
                        ComputeLevel(c["value"], c["label"]) for c in data["compute_grid"]
                    ],
                    accuracy=data["accuracy"],
                    n_per_cell=data["n_per_cell"],
                ),
            )
        )
    return results


def render_curve(result: ProbeResult):
    """Return a matplotlib figure with accuracy-vs-compute curves."""
    A = result.as_array()
    ci = result.ci()
    xs = list(range(len(result.compute_grid)))
    labels = [c.label for c in result.compute_grid]

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, max(1, len(result.depths))))
    for i, d in enumerate(result.depths):
        ax.plot(xs, A[i], "-o", color=cmap[i], label=f"depth={d}")
        ax.fill_between(xs, ci[i, :, 0], ci[i, :, 1], color=cmap[i], alpha=0.15)
    ax.set_xticks(xs, labels)
    ax.set_xlabel(f"compute ({result.compute_axis})")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{result.adapter_name} · {result.task_name}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    return fig


def render_heatmap(result: ProbeResult):
    A = result.as_array()
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(A, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="lower")
    ax.set_xticks(range(len(result.compute_grid)), [c.label for c in result.compute_grid])
    ax.set_yticks(range(len(result.depths)), [str(d) for d in result.depths])
    ax.set_xlabel(f"compute ({result.compute_axis})")
    ax.set_ylabel("depth")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(
                j, i, f"{A[i, j]:.2f}",
                ha="center", va="center",
                color="white" if A[i, j] < 0.5 else "black",
                fontsize=8,
            )
    plt.colorbar(im, ax=ax, label="accuracy")
    plt.tight_layout()
    return fig


def main():
    import streamlit as st

    st.set_page_config(page_title="depth-lens", layout="wide")
    st.title("depth-lens · reasoning depth dashboard")

    cache_path = Path(st.sidebar.text_input(
        "Cache directory",
        value=str(default_cache_dir()),
        help="Probe results JSONs are loaded from here.",
    ))

    results = _load_all(cache_path)
    st.sidebar.write(f"{len(results)} cached probes")

    if not results:
        st.info(
            "No cached probe results found. Run `depth-lens probe --model ... --task ...` "
            "first; results are cached to the directory above."
        )
        return

    # Index for filtering
    adapters = sorted({r.adapter_name for _, r in results})
    tasks = sorted({r.task_name for _, r in results})

    sel_adapters = st.sidebar.multiselect("Adapters", adapters, default=adapters)
    sel_tasks = st.sidebar.multiselect("Tasks", tasks, default=tasks)

    chosen = [
        (fn, r)
        for fn, r in results
        if r.adapter_name in sel_adapters and r.task_name in sel_tasks
    ]

    # Summary table
    rows = []
    for _, r in chosen:
        eff = r.effective_depth(0.5)
        rows.append({
            "adapter": r.adapter_name,
            "task": r.task_name,
            "compute_axis": r.compute_axis,
            "depths": str(r.depths),
            "n_per_cell": r.n_per_cell,
            "effective_depth (≥0.5)": eff if eff is not None else "—",
        })
    st.subheader("Summary")
    st.dataframe(rows, use_container_width=True)

    # Per-probe drill-down
    st.subheader("Drill-down")
    for fn, r in chosen:
        with st.expander(f"{r.adapter_name} · {r.task_name}  ({fn})"):
            col1, col2 = st.columns(2)
            col1.pyplot(render_curve(r))
            col2.pyplot(render_heatmap(r))

            # Overthinking report
            over_msgs = []
            for d in r.depths:
                over = r.overthinking(d)
                if over:
                    over_msgs.append(
                        f"- depth **{d}**: peak `{over['peak_compute'].label}` "
                        f"acc={over['peak_accuracy']:.2f} → last `{over['last_compute'].label}` "
                        f"acc={over['last_accuracy']:.2f} (drop {over['drop']:.2f})"
                    )
            if over_msgs:
                st.markdown("**Overthinking detected at**:\n" + "\n".join(over_msgs))
            else:
                st.markdown("No overthinking detected.")


if __name__ == "__main__":
    main()
