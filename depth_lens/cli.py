"""depth-lens command-line entry point."""

from __future__ import annotations

import json
import os
from pathlib import Path

import click

from depth_lens.adapters.base import ComputeLevel, ModelAdapter
from depth_lens.metrics import ProbeResult, probe
from depth_lens.tasks import get_task


def _build_adapter(
    model_spec: str,
    task_name: str,
    *,
    checkpoint: str | None = None,
    train_steps: int = 6000,
    save_checkpoint: str | None = None,
) -> ModelAdapter:
    """Construct an adapter from a CLI --model spec.

    Supported specs:
        openmythos                  — train (or load) a tiny OpenMythos for this task
        hf:<hf-model-id>            — wrap a HuggingFace causal LM
    """
    if model_spec == "openmythos":
        from depth_lens.adapters.openmythos_adapter import (
            TrainConfig,
            load_checkpoint,
            save_checkpoint as _save_ckpt,
            train_for_task,
        )

        if checkpoint:
            click.echo(f"[openmythos] loading checkpoint: {checkpoint}")
            return load_checkpoint(Path(checkpoint))

        click.echo(
            f"[openmythos] no checkpoint given — training a tiny model for "
            f"{train_steps} steps on task '{task_name}' (a few minutes)."
        )
        adapter = train_for_task(get_task(task_name), cfg=TrainConfig(steps=train_steps))
        if save_checkpoint:
            _save_ckpt(adapter, Path(save_checkpoint))
            click.echo(f"[openmythos] saved checkpoint: {save_checkpoint}")
        return adapter

    if model_spec.startswith("hf:"):
        from depth_lens.adapters.hf_adapter import HuggingFaceAdapter

        hf_id = model_spec[len("hf:"):]
        click.echo(f"[hf] loading {hf_id}")
        return HuggingFaceAdapter(model_name=hf_id, task_name=task_name)

    if model_spec.startswith("anthropic:"):
        from depth_lens.adapters.anthropic_adapter import AnthropicAdapter

        model = model_spec[len("anthropic:"):]
        click.echo(f"[anthropic] using {model} via API")
        return AnthropicAdapter(model=model, task_name=task_name)

    if model_spec.startswith("openai:"):
        from depth_lens.adapters.openai_adapter import OpenAIAdapter

        model = model_spec[len("openai:"):]
        click.echo(f"[openai] using {model} via API")
        return OpenAIAdapter(model=model, task_name=task_name)

    if model_spec.startswith("gemini:"):
        from depth_lens.adapters.gemini_adapter import GeminiAdapter

        model = model_spec[len("gemini:"):]
        click.echo(f"[gemini] using {model} via API")
        return GeminiAdapter(model=model, task_name=task_name)

    if model_spec.startswith("vllm:"):
        from depth_lens.adapters.vllm_adapter import VLLMAdapter

        model = model_spec[len("vllm:"):]
        base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        click.echo(f"[vllm] using {model} at {base_url}")
        return VLLMAdapter(model=model, task_name=task_name, base_url=base_url)

    raise click.UsageError(
        f"Unknown --model {model_spec!r}. Use 'openmythos', 'hf:<hf-id>', "
        "'anthropic:<model>', 'openai:<model>', 'gemini:<model>', or 'vllm:<model>'."
    )


def _parse_compute(s: str | None, axis_name: str) -> list[ComputeLevel] | None:
    if not s:
        return None
    return [ComputeLevel(int(v), f"{axis_name}={v}") for v in s.split(",") if v.strip()]


def _dump_result_json(result: ProbeResult, path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "task": result.task_name,
                "adapter": result.adapter_name,
                "compute_axis": result.compute_axis,
                "depths": result.depths,
                "compute_grid": [
                    {"value": c.value, "label": c.label} for c in result.compute_grid
                ],
                "accuracy": result.accuracy,
                "n_per_cell": result.n_per_cell,
            },
            indent=2,
        )
    )


@click.group()
def cli():
    """depth-lens — measure reasoning depth across model families."""


# ---------------------------------------------------------------------------
# probe — single-model sweep
# ---------------------------------------------------------------------------


@cli.command("probe")
@click.option("--model", "-m", required=True, help="'openmythos' or 'hf:<model-id>'")
@click.option("--task", "-t", default="k-hop", help="Task name (default: k-hop).")
@click.option("--depths", default="2,3,4,5,6,7,8,10", help="Comma-separated task depths.")
@click.option("--compute", default=None, help="Comma-separated compute levels.")
@click.option("--n-samples", default=128, type=int, help="Samples per cell.")
@click.option("--batch-size", default=32, type=int, help="Adapter batch size.")
@click.option("--seed", default=0, type=int)
@click.option("--plot", default="probe.png", type=click.Path(dir_okay=False))
@click.option("--heatmap", default=None, type=click.Path(dir_okay=False))
@click.option("--save-json", default=None, type=click.Path(dir_okay=False))
@click.option("--checkpoint", default=None, type=click.Path(dir_okay=False))
@click.option("--train-steps", default=6000, type=int)
@click.option("--save-checkpoint", default=None, type=click.Path(dir_okay=False))
def probe_cmd(
    model: str,
    task: str,
    depths: str,
    compute: str | None,
    n_samples: int,
    batch_size: int,
    seed: int,
    plot: str,
    heatmap: str | None,
    save_json: str | None,
    checkpoint: str | None,
    train_steps: int,
    save_checkpoint: str | None,
):
    """Run a depth × compute probe for one model."""
    depths_list = [int(x) for x in depths.split(",") if x.strip()]
    task_obj = get_task(task)
    adapter = _build_adapter(
        model,
        task,
        checkpoint=checkpoint,
        train_steps=train_steps,
        save_checkpoint=save_checkpoint,
    )

    compute_grid = _parse_compute(compute, adapter.compute_axis_name)

    result = probe(
        adapter=adapter,
        task=task_obj,
        depths=depths_list,
        compute_grid=compute_grid,
        n_samples=n_samples,
        batch_size=batch_size,
        seed=seed,
    )

    from depth_lens.viz import plot_accuracy_curve, plot_accuracy_heatmap

    click.echo(f"Curve plot: {plot_accuracy_curve(result, Path(plot))}")
    if heatmap:
        click.echo(f"Heatmap:    {plot_accuracy_heatmap(result, Path(heatmap))}")
    if save_json:
        _dump_result_json(result, Path(save_json))
        click.echo(f"JSON:       {save_json}")

    eff = result.effective_depth(0.5)
    click.echo("")
    click.echo("---- summary ----")
    click.echo(f"effective depth (≥0.5 acc at some compute): {eff}")
    for d in result.depths:
        over = result.overthinking(d)
        if over:
            click.echo(
                f"overthinking @ depth {d}: peak={over['peak_compute']} (acc={over['peak_accuracy']:.2f})"
                f"  →  last={over['last_compute']} (acc={over['last_accuracy']:.2f})"
            )


# ---------------------------------------------------------------------------
# compare — overlay multiple models on the same task
# ---------------------------------------------------------------------------


@cli.command("compare")
@click.option(
    "--models",
    required=True,
    help="Comma-separated model specs (e.g. 'openmythos,hf:Qwen/Qwen2.5-1.5B-Instruct').",
)
@click.option("--task", default="k-hop", help="Task name.")
@click.option("--depths", default="2,4,6,8", help="Comma-separated depths to evaluate.")
@click.option(
    "--focus-depths",
    default=None,
    help="Comma-separated depths to draw in the overlay (default: all evaluated).",
)
@click.option("--n-samples", default=128, type=int)
@click.option("--batch-size", default=32, type=int)
@click.option("--seed", default=0, type=int)
@click.option("--plot", default="compare.png", type=click.Path(dir_okay=False))
@click.option(
    "--save-json",
    default=None,
    type=click.Path(dir_okay=False),
    help="If given, dump all per-model probe results into this path.",
)
@click.option("--checkpoint", default=None, type=click.Path(dir_okay=False),
              help="OpenMythos checkpoint to load (applies only to 'openmythos').")
@click.option("--train-steps", default=4000, type=int)
def compare_cmd(
    models: str,
    task: str,
    depths: str,
    focus_depths: str | None,
    n_samples: int,
    batch_size: int,
    seed: int,
    plot: str,
    save_json: str | None,
    checkpoint: str | None,
    train_steps: int,
):
    """Probe multiple models and overlay their accuracy-vs-compute curves."""
    depths_list = [int(x) for x in depths.split(",") if x.strip()]
    focus_list = (
        [int(x) for x in focus_depths.split(",") if x.strip()] if focus_depths else None
    )
    task_obj = get_task(task)
    model_specs = [m.strip() for m in models.split(",") if m.strip()]

    results: list[ProbeResult] = []
    for spec in model_specs:
        click.echo(f"\n=== {spec} ===")
        adapter = _build_adapter(
            spec,
            task,
            checkpoint=checkpoint if spec == "openmythos" else None,
            train_steps=train_steps,
        )
        r = probe(
            adapter=adapter,
            task=task_obj,
            depths=depths_list,
            n_samples=n_samples,
            batch_size=batch_size,
            seed=seed,
        )
        results.append(r)
        adapter.teardown()

    from depth_lens.viz import plot_overlay

    out_path = plot_overlay(results, Path(plot), focus_depths=focus_list)
    click.echo(f"\nOverlay plot: {out_path}")

    if save_json:
        out = []
        for r in results:
            out.append(
                {
                    "task": r.task_name,
                    "adapter": r.adapter_name,
                    "compute_axis": r.compute_axis,
                    "depths": r.depths,
                    "compute_grid": [
                        {"value": c.value, "label": c.label} for c in r.compute_grid
                    ],
                    "accuracy": r.accuracy,
                    "n_per_cell": r.n_per_cell,
                }
            )
        Path(save_json).write_text(json.dumps(out, indent=2))
        click.echo(f"JSON:         {save_json}")

    # Console summary table
    click.echo("\n---- effective depth (≥0.5 acc at some compute) ----")
    for r in results:
        click.echo(f"  {r.adapter_name:<40} {r.effective_depth(0.5)}")


# ---------------------------------------------------------------------------
# dashboard — launch the Streamlit explorer
# ---------------------------------------------------------------------------


@cli.command("dashboard")
@click.option("--port", default=8501, type=int, help="Port to bind the Streamlit app to.")
def dashboard_cmd(port: int):
    """Launch the local Streamlit dashboard for cached probe results."""
    import subprocess
    import sys
    from pathlib import Path

    try:
        import streamlit  # noqa: F401
    except ImportError:
        raise click.UsageError(
            "Streamlit not installed. `pip install streamlit` or "
            "`pip install -e .[dashboard]`."
        )

    dash = Path(__file__).parent / "dashboard.py"
    click.echo(f"Launching depth-lens dashboard on port {port}…")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dash), "--server.port", str(port)]
    )


if __name__ == "__main__":
    cli()
