"""Smoke tests for dashboard data plumbing (no Streamlit runtime)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from depth_lens.adapters.base import ComputeLevel
from depth_lens.cache import cache_key, save_cached
from depth_lens.dashboard import _load_all, render_curve, render_heatmap
from depth_lens.metrics import ProbeResult


def _result(adapter="openmythos", task="k-hop"):
    return ProbeResult(
        task_name=task,
        adapter_name=adapter,
        compute_axis="n_loops",
        depths=[2, 4, 6],
        compute_grid=[ComputeLevel(1, "n_loops=1"), ComputeLevel(4, "n_loops=4"), ComputeLevel(16, "n_loops=16")],
        accuracy=[[0.5, 1.0, 0.9], [0.2, 0.9, 0.5], [0.1, 0.3, 0.2]],
        n_per_cell=100,
    )


def test_load_all_returns_results():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        r1 = _result(adapter="openmythos", task="k-hop")
        r2 = _result(adapter="hf:foo", task="parity")
        save_cached(cache_key(r1.adapter_name, r1.task_name, r1.depths, r1.compute_grid, 100, 0), r1, cache_dir_path=td_path)
        save_cached(cache_key(r2.adapter_name, r2.task_name, r2.depths, r2.compute_grid, 100, 0), r2, cache_dir_path=td_path)

        loaded = _load_all(td_path)
        assert len(loaded) == 2
        adapters = {r.adapter_name for _, r in loaded}
        assert adapters == {"openmythos", "hf:foo"}


def test_load_all_empty_dir():
    with tempfile.TemporaryDirectory() as td:
        assert _load_all(Path(td)) == []


def test_render_curve_and_heatmap_dont_crash():
    r = _result()
    fig1 = render_curve(r)
    fig2 = render_heatmap(r)
    assert fig1 is not None
    assert fig2 is not None
