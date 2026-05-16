"""
Static plotting for ProbeResult.

v0.1 ships matplotlib-only static plots. v0.5 will add Streamlit dashboards.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from depth_lens.metrics import ProbeResult


def plot_accuracy_curve(
    result: ProbeResult,
    out_path: Path,
    title: str | None = None,
    log_x: bool = True,
    show_ci: bool = True,
) -> Path:
    """
    accuracy-vs-compute curve, one line per task depth.

    Wilson 95% CI bands are drawn when `show_ci=True` (default).

    Returns the path written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    A = result.as_array()
    xs = [c.value for c in result.compute_grid]
    xlabels = [c.label for c in result.compute_grid]
    ci = result.ci() if show_ci else None

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(result.depths)))
    for i, d in enumerate(result.depths):
        ax.plot(xs, A[i], "-o", color=cmap[i], label=f"depth={d}")
        if ci is not None:
            ax.fill_between(xs, ci[i, :, 0], ci[i, :, 1], color=cmap[i], alpha=0.15)

    if log_x:
        ax.set_xscale("log", base=2)
    ax.set_xticks(xs, xlabels, rotation=0)
    ax.set_xlabel(f"compute ({result.compute_axis})")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, which="both")
    ax.legend(ncol=2, fontsize=8)

    ax.set_title(
        title
        or f"{result.adapter_name}  ·  {result.task_name}  ·  n={result.n_per_cell}/cell"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    return out_path


def plot_overlay(
    results: list[ProbeResult],
    out_path: Path,
    focus_depths: list[int] | None = None,
    title: str | None = None,
) -> Path:
    """
    Overlay accuracy-vs-compute curves from multiple adapters.

    Each adapter has its own compute axis (n_loops vs thinking_tokens etc.),
    so the x-axis here is the *compute level index* (0..N-1). Each model's
    actual compute labels are shown at each tick, stacked per model.

    `focus_depths` selects which task depths to draw. Defaults to all depths
    common across the results.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if focus_depths is None:
        common = set(results[0].depths)
        for r in results[1:]:
            common &= set(r.depths)
        focus_depths = sorted(common)
    if not focus_depths:
        raise ValueError("No common depths across the supplied results.")

    n_panels = len(focus_depths)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5 * n_panels, 5), sharey=True, squeeze=False
    )
    axes_flat = axes[0]

    model_colors = plt.cm.tab10(np.linspace(0, 0.9, len(results)))

    for ax, d in zip(axes_flat, focus_depths, strict=False):
        for r, color in zip(results, model_colors, strict=False):
            di = r.depths.index(d)
            ys = r.as_array()[di]
            xs = list(range(len(r.compute_grid)))
            ax.plot(xs, ys, "-o", color=color, label=r.adapter_name)
        ax.set_xlabel("compute level (rank)")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.set_title(f"depth = {d}")
        ax.legend(fontsize=8)

    axes_flat[0].set_ylabel("accuracy")

    # Tick labels per model, stacked under the panel.
    tick_lines = []
    max_grid = max(len(r.compute_grid) for r in results)
    for r in results:
        labels = [c.label for c in r.compute_grid]
        if len(labels) < max_grid:
            labels = labels + [""] * (max_grid - len(labels))
        tick_lines.append(f"{r.adapter_name}: " + " | ".join(labels))
    footer = "\n".join(tick_lines)

    fig.suptitle(title or f"depth-lens compare — task: {results[0].task_name}")
    plt.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.text(0.02, 0.01, footer, fontsize=7, family="monospace")

    plt.savefig(out_path, dpi=140)
    plt.close()
    return out_path


def plot_accuracy_heatmap(result: ProbeResult, out_path: Path) -> Path:
    """Heatmap of accuracy on the (depth × compute) grid."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    A = result.as_array()
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(A, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="lower")
    ax.set_xticks(range(len(result.compute_grid)), [c.label for c in result.compute_grid])
    ax.set_yticks(range(len(result.depths)), [str(d) for d in result.depths])
    ax.set_xlabel(f"compute ({result.compute_axis})")
    ax.set_ylabel("depth")
    ax.set_title(
        f"{result.adapter_name}  ·  {result.task_name}  ·  accuracy heatmap"
    )
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
    plt.savefig(out_path, dpi=140)
    plt.close()
    return out_path
