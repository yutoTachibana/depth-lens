"""depth-lens command-line entry point."""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import click

from depth_lens.adapters.base import ComputeLevel, ModelAdapter
from depth_lens.metrics import ProbeResult, probe
from depth_lens.tasks import get_task

# Force UTF-8 stdout/stderr on Windows. The default cp932 console can't encode
# the em-dashes, arrows, and status emoji we use in help text and result tables
# — without this, `depth-lens --help` crashes on a fresh Windows box.
if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def _build_adapter(
    model_spec: str,
    task_name: str,
    *,
    checkpoint: str | None = None,
    train_steps: int = 6000,
    save_checkpoint: str | None = None,
    compute_axis: str | None = None,
) -> ModelAdapter:
    """Construct an adapter from a CLI --model spec.

    Supported specs:
        openmythos                  — train (or load) a tiny OpenMythos for this task
        hf:<hf-model-id>            — wrap a HuggingFace causal LM
        vllm:<model-id>             — vLLM / OpenAI-compatible local server

    `compute_axis` is currently only honored by vLLM (`reasoning_effort` vs
    `max_tokens`). Other adapters ignore it.
    """
    if model_spec == "openmythos":
        from depth_lens.adapters.openmythos_adapter import (
            TrainConfig,
            load_checkpoint,
            train_for_task,
        )
        from depth_lens.adapters.openmythos_adapter import (
            save_checkpoint as _save_ckpt,
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
        axis = compute_axis or "reasoning_effort"
        click.echo(f"[vllm] using {model} at {base_url} (compute_axis={axis})")
        return VLLMAdapter(
            model=model,
            task_name=task_name,
            base_url=base_url,
            compute_axis=axis,
        )

    raise click.UsageError(
        f"Unknown --model {model_spec!r}. Use 'openmythos', 'hf:<hf-id>', "
        "'anthropic:<model>', 'openai:<model>', 'gemini:<model>', or 'vllm:<model>'."
    )


def _parse_compute(s: str | None, axis_name: str) -> list[ComputeLevel] | None:
    """Parse a comma-separated --compute string.

    Values that parse as ints are used directly as ComputeLevel.value
    (e.g., thinking_budget=1024 or max_tokens=256). Non-numeric values
    (e.g., effort=low) get a rank index as the value so plots can sort
    them; the original string is preserved in the label."""
    if not s:
        return None
    out: list[ComputeLevel] = []
    for rank, v in enumerate(p for p in s.split(",") if p.strip()):
        v = v.strip()
        try:
            value: int | float = int(v)
        except ValueError:
            value = rank
        out.append(ComputeLevel(value, f"{axis_name}={v}"))
    return out


#: Exception types that should be surfaced as click.UsageError
#: (single-line "Error: ..." instead of a full Python traceback).
#: Used by the @cli.group() so all subcommands inherit the behavior.
_USER_ERROR_TYPES: tuple[type[BaseException], ...] = (
    FileNotFoundError,   # custom: JSONL path doesn't exist
    KeyError,            # custom: depth not present in JSONL
    json.JSONDecodeError,  # custom: malformed JSONL
    RuntimeError,        # adapter constructors when API key is missing
    ValueError,          # adapter compute_axis / pricing validation
)


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
                "latency_per_cell": result.latency_per_cell,
                "tokens_per_cell": result.tokens_per_cell,
            },
            indent=2,
        )
    )


class _FriendlyErrorGroup(click.Group):
    """A click Group that converts well-known user-input errors into
    click.UsageError so they render as a clean single-line `Error: ...`
    instead of a Python traceback."""

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except click.ClickException:
            raise
        except _USER_ERROR_TYPES as e:
            raise click.UsageError(f"{type(e).__name__}: {e}") from e


@click.group(cls=_FriendlyErrorGroup)
def cli():
    """depth-lens — measure reasoning depth across model families."""


# ---------------------------------------------------------------------------
# probe — single-model sweep
# ---------------------------------------------------------------------------


@cli.command("probe")
@click.option(
    "--model", "-m", required=True,
    help="Adapter spec: 'openmythos', 'hf:<model-id>', "
         "'anthropic:<model>', 'openai:<model>', 'gemini:<model>', or "
         "'vllm:<model>' (OpenAI-compatible local server).",
)
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
@click.option(
    "--compute-axis", default=None,
    type=click.Choice(["reasoning_effort", "max_tokens"]),
    help="Compute knob for vLLM adapters. 'reasoning_effort' for thinking "
         "models (DeepSeek-R1-Distill, Qwen-Thinking); 'max_tokens' for "
         "non-thinking models (Llama-3-8B-Instruct). Default: reasoning_effort.",
)
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
    compute_axis: str | None,
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
        compute_axis=compute_axis,
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
    click.echo(f"effective depth (>=0.5 acc at some compute): {eff}")
    for d in result.depths:
        over = result.overthinking(d)
        if over:
            click.echo(
                f"overthinking @ depth {d}: peak={over['peak_compute']} (acc={over['peak_accuracy']:.2f})"
                f"  ->  last={over['last_compute']} (acc={over['last_accuracy']:.2f})"
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
@click.option(
    "--compute",
    default=None,
    help=(
        "Comma-separated integer compute levels applied to every adapter "
        "(useful when you want a shared x-axis across vendors). "
        "Default: each adapter's own grid."
    ),
)
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
    compute: str | None,
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
        compute_grid = _parse_compute(compute, adapter.compute_axis_name)
        r = probe(
            adapter=adapter,
            task=task_obj,
            depths=depths_list,
            compute_grid=compute_grid,
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
                    "latency_per_cell": r.latency_per_cell,
                    "tokens_per_cell": r.tokens_per_cell,
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
    except ImportError as e:
        # ImportError fires for both "not installed" and dep-conflict
        # ("cannot import name X from Y"). Surface the real exception so the
        # user can tell which case they're in instead of running `pip install
        # streamlit` against an already-installed-but-broken environment.
        raise click.UsageError(
            f"Cannot import streamlit ({type(e).__name__}: {e}).\n"
            "If you have not installed it: `pip install streamlit` or "
            "`pip install -e .[dashboard]`.\n"
            "If it IS installed, you likely have a dependency version "
            "conflict — try `pip install --upgrade streamlit starlette` or "
            "reinstall in a fresh venv."
        ) from e

    dash = Path(__file__).parent / "dashboard.py"
    click.echo(f"Launching depth-lens dashboard on port {port}…")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dash), "--server.port", str(port)]
    )


# ---------------------------------------------------------------------------
# recommend — production model picker
# ---------------------------------------------------------------------------


@cli.command("recommend")
@click.option(
    "--models", "-m", required=True,
    help="Comma-separated candidate model specs (e.g., 'anthropic:claude-haiku-4-5,anthropic:claude-sonnet-4-6,openai:o4-mini').",
)
@click.option("--task", "-t", required=True, help="Task name or 'custom:<jsonl>:<scorer>'.")
@click.option(
    "--target-accuracy", "-a", required=True, type=float,
    help="Minimum accuracy required (e.g., 0.95).",
)
@click.option("--depths", default=None, help="Comma-separated task depths (default: all available in custom JSONL, else 4).")
@click.option("--compute", default=None, help="Comma-separated compute levels (default: each adapter's own grid).")
@click.option("--n-samples", default=64, type=int, help="Samples per cell.")
@click.option("--batch-size", default=8, type=int)
@click.option("--seed", default=0, type=int)
@click.option("--pricing", default=None, type=click.Path(dir_okay=False), help="Optional JSON pricing override.")
@click.option("--daily-calls", default=None, type=int, help="If set, show projected daily / yearly cost at this volume.")
@click.option(
    "--max-latency", default=None, type=float,
    help="Drop configurations whose median latency exceeds this many seconds per prediction. "
         "Use to enforce a UX-side speed SLA (e.g., --max-latency 2.0 for a chat UI).",
)
@click.option(
    "--gpu-hourly-rate", default=None, type=float,
    help="USD per GPU-hour used to amortize cost for self-hosted models "
         "(vllm:*, hf:*, openmythos) when no explicit pricing entry is provided. "
         "Default: 0.50 (midpoint between AWS g5 spot and on-demand).",
)
@click.option(
    "--compute-axis", default=None,
    type=click.Choice(["reasoning_effort", "max_tokens"]),
    help="Compute knob for vLLM adapters in --models. Only the vllm:* specs "
         "in --models read this; other adapters use their native knob. "
         "Use 'max_tokens' when including an instruct-only model like "
         "Llama-3-8B-Instruct (it does not accept reasoning_effort).",
)
def recommend_cmd(
    models: str,
    task: str,
    target_accuracy: float,
    depths: str | None,
    compute: str | None,
    n_samples: int,
    batch_size: int,
    seed: int,
    pricing: str | None,
    daily_calls: int | None,
    max_latency: float | None,
    gpu_hourly_rate: float | None,
    compute_axis: str | None,
):
    """
    Find the cheapest model + compute setting that meets your accuracy bar.

    Probes every (model, compute) combination on your task and ranks the
    passing ones by $/prediction. Standard production workflow:

    \b
        depth-lens recommend \\
            --models anthropic:claude-haiku-4-5,anthropic:claude-sonnet-4-6,anthropic:claude-opus-4-7 \\
            --task custom:./my_eval.jsonl:first_int \\
            --target-accuracy 0.95 \\
            --daily-calls 10000
    """
    from depth_lens.pricing import get_pricing, load_pricing_file, maybe_gpu_hour_fallback

    pricing_override = load_pricing_file(pricing) if pricing else None

    task_obj = get_task(task)

    # Resolve depths
    if depths:
        depths_list = [int(x) for x in depths.split(",") if x.strip()]
    elif hasattr(task_obj, "available_depths"):
        depths_list = task_obj.available_depths()
    else:
        depths_list = [4]

    model_specs = [m.strip() for m in models.split(",") if m.strip()]

    rows: list[dict] = []
    for spec in model_specs:
        click.echo(f"\n=== {spec} ===")
        spec_pricing = maybe_gpu_hour_fallback(
            spec, get_pricing(spec, pricing_override), gpu_hourly_rate
        )
        if spec_pricing is None:
            click.echo(
                f"  [warn] no pricing for {spec}; cost columns will be blank. "
                f"Provide --pricing to override."
            )
        elif "gpu_hourly" in spec_pricing and get_pricing(spec, pricing_override) is None:
            click.echo(
                f"  [info] self-hosted spec — costing at ${spec_pricing['gpu_hourly']:.2f}/GPU-hour "
                f"× {spec_pricing.get('gpus', 1)} GPU(s)"
            )
        adapter = _build_adapter(spec, task, compute_axis=compute_axis)
        compute_grid = _parse_compute(compute, adapter.compute_axis_name)
        r = probe(
            adapter=adapter,
            task=task_obj,
            depths=depths_list,
            compute_grid=compute_grid,
            n_samples=n_samples,
            batch_size=batch_size,
            seed=seed,
        )
        cost_grid = r.cost_per_cell(spec_pricing) if spec_pricing else None
        lat_grid = r.latency_per_cell
        for di, d in enumerate(r.depths):
            for ci, c in enumerate(r.compute_grid):
                acc = r.accuracy[di][ci]
                cost = float(cost_grid[di, ci]) if cost_grid is not None else None
                latency = float(lat_grid[di][ci]) if lat_grid else None
                acc_ok = acc >= target_accuracy
                lat_ok = (max_latency is None) or (latency is None) or (latency <= max_latency)
                rows.append({
                    "model": spec,
                    "depth": d,
                    "compute": c.label,
                    "accuracy": acc,
                    "cost_per_pred": cost,
                    "latency_sec": latency,
                    "passes": acc_ok and lat_ok,
                    "failed_on": (
                        None if (acc_ok and lat_ok)
                        else "accuracy+latency" if (not acc_ok and not lat_ok)
                        else "accuracy" if not acc_ok
                        else "latency"
                    ),
                })
        adapter.teardown()

    passing = [r for r in rows if r["passes"]]
    failing = [r for r in rows if not r["passes"]]
    passing.sort(key=lambda r: (r["cost_per_pred"] if r["cost_per_pred"] is not None else float("inf")))
    fastest = sorted(
        passing, key=lambda r: (r["latency_sec"] if r["latency_sec"] is not None else float("inf"))
    )[0] if passing else None
    cheapest = passing[0] if passing else None

    click.echo("")
    click.echo("=" * 92)
    bar_msg = f"Target accuracy ≥ {target_accuracy:.2f}"
    if max_latency is not None:
        bar_msg += f"  ·  Max latency ≤ {max_latency:.2f}s/pred"
    click.echo(bar_msg)
    click.echo(f"Probed {len(rows)} configurations, {len(passing)} passing.")
    click.echo("=" * 92)

    def _fmt(r):
        cost = "(no pricing)" if r["cost_per_pred"] is None else f"${r['cost_per_pred']*1000:.3f}/k-pred"
        lat = "(no lat)" if r["latency_sec"] is None else f"{r['latency_sec']:>5.2f}s/pred"
        return (
            f"  {r['model']:<40s}  d={r['depth']:<2d}  {r['compute']:<28s} "
            f"acc={r['accuracy']:.2f}  {cost:>18s}  {lat}"
        )

    if passing:
        click.echo("\n✅ Passing (cheapest first):")
        for r in passing[:10]:
            tags = []
            if r is cheapest and r["cost_per_pred"] is not None:
                tags.append("← cheapest")
            if r is fastest and fastest is not cheapest and r["latency_sec"] is not None:
                tags.append("← fastest")
            tag_str = ("  " + " ".join(tags)) if tags else ""
            click.echo(_fmt(r) + tag_str)
        if len(passing) > 10:
            click.echo(f"  … and {len(passing) - 10} more")
        # If cheapest and fastest differ, highlight the tradeoff.
        if fastest is not cheapest and cheapest is not None and fastest is not None:
            slowdown = (cheapest["latency_sec"] / fastest["latency_sec"]
                        if cheapest["latency_sec"] and fastest["latency_sec"] else None)
            speedup_premium = (fastest["cost_per_pred"] / cheapest["cost_per_pred"]
                               if cheapest["cost_per_pred"] and fastest["cost_per_pred"] else None)
            click.echo("")
            click.echo("⚡ Cost-vs-speed tradeoff among passing configs:")
            if slowdown is not None and speedup_premium is not None:
                click.echo(
                    f"  Cheapest is {slowdown:.1f}× slower than fastest; "
                    f"fastest costs {speedup_premium:.1f}× more per call."
                )
    else:
        click.echo("\n❌ No configuration met the bar.")
        if failing:
            click.echo("  Closest attempts:")
            failing.sort(key=lambda r: (-r["accuracy"], r["latency_sec"] or float("inf")))
            for r in failing[:5]:
                fail_tag = f"  ← failed on {r['failed_on']}" if r.get("failed_on") else ""
                click.echo(_fmt(r) + fail_tag)

    if passing and daily_calls is not None and cheapest is not None and cheapest["cost_per_pred"] is not None:
        daily_cost = cheapest["cost_per_pred"] * daily_calls
        yearly = daily_cost * 365
        click.echo("")
        click.echo("=" * 92)
        click.echo(f"At {daily_calls:,} calls/day with the cheapest passing config:")
        click.echo(f"  {cheapest['model']} @ {cheapest['compute']}")
        click.echo(f"  → ${daily_cost:.2f}/day  ${yearly:,.0f}/year")

        # If a more-expensive passing option exists, show savings vs it
        most_expensive_passing = max(passing, key=lambda r: r["cost_per_pred"] or 0)
        if most_expensive_passing is not cheapest and most_expensive_passing["cost_per_pred"]:
            mep_daily = most_expensive_passing["cost_per_pred"] * daily_calls
            savings = mep_daily - daily_cost
            click.echo(
                f"\n  Switching from {most_expensive_passing['model']} @ {most_expensive_passing['compute']} "
                f"(${mep_daily:.2f}/day)\n"
                f"  saves ${savings:.2f}/day = ${savings*365:,.0f}/year "
                f"({(savings/mep_daily*100):.0f}% reduction)"
            )


if __name__ == "__main__":
    cli()
