"""Unit tests for the probe result cache."""

from __future__ import annotations

import tempfile
from pathlib import Path

from depth_lens.adapters.base import ComputeLevel
from depth_lens.cache import cache_key, load_cached, save_cached
from depth_lens.metrics import ProbeResult


def _result() -> ProbeResult:
    return ProbeResult(
        task_name="k-hop",
        adapter_name="openmythos",
        compute_axis="n_loops",
        depths=[2, 4],
        compute_grid=[ComputeLevel(1, "n_loops=1"), ComputeLevel(2, "n_loops=2")],
        accuracy=[[0.5, 0.7], [0.3, 0.4]],
        n_per_cell=100,
    )


def test_cache_key_deterministic():
    grid = [ComputeLevel(1, "a"), ComputeLevel(2, "b")]
    k1 = cache_key("om", "t", [1, 2], grid, n_samples=10, seed=0)
    k2 = cache_key("om", "t", [1, 2], grid, n_samples=10, seed=0)
    assert k1 == k2


def test_cache_key_differs_on_change():
    grid = [ComputeLevel(1, "a")]
    k1 = cache_key("om", "t", [1], grid, n_samples=10, seed=0)
    k2 = cache_key("om", "t", [1], grid, n_samples=10, seed=1)
    assert k1 != k2


def test_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        r = _result()
        grid = r.compute_grid
        key = cache_key(r.adapter_name, r.task_name, r.depths, grid, 100, 0)
        path = save_cached(key, r, cache_dir_path=td)
        assert Path(path).exists()
        loaded = load_cached(key, cache_dir_path=td)
        assert loaded is not None
        assert loaded.task_name == r.task_name
        assert loaded.adapter_name == r.adapter_name
        assert loaded.accuracy == r.accuracy
        assert loaded.depths == r.depths


def test_cache_miss_returns_none():
    with tempfile.TemporaryDirectory() as td:
        assert load_cached("nonexistent_key", cache_dir_path=td) is None
