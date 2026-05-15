"""
JSON-on-disk cache for probe results.

Keyed by a SHA256 of the probe inputs (adapter name, task name, depths,
compute grid, n_samples, seed). If a probe with the same key has already run,
the cached ProbeResult is returned and the model is not invoked at all.

The cache directory defaults to `~/.cache/depth-lens/probes/`. Override with
the `DEPTH_LENS_CACHE` environment variable, or pass `cache_dir=...` to
`probe()`.

The cache is intentionally simple: one JSON file per probe key, no eviction.
Delete files manually to invalidate.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

from depth_lens.adapters.base import ComputeLevel
from depth_lens.metrics import ProbeResult


def cache_dir(override: str | Path | None = None) -> Path:
    if override is not None:
        return Path(override)
    env = os.environ.get("DEPTH_LENS_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "depth-lens" / "probes"


def cache_key(
    adapter_name: str,
    task_name: str,
    depths: Iterable[int],
    compute_grid: Iterable[ComputeLevel],
    n_samples: int,
    seed: int,
) -> str:
    payload = {
        "adapter": adapter_name,
        "task": task_name,
        "depths": list(depths),
        "compute": [{"value": c.value, "label": c.label} for c in compute_grid],
        "n_samples": n_samples,
        "seed": seed,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:24]


def load_cached(
    key: str, cache_dir_path: str | Path | None = None
) -> ProbeResult | None:
    path = cache_dir(cache_dir_path) / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return _deserialize(data)


def save_cached(
    key: str,
    result: ProbeResult,
    cache_dir_path: str | Path | None = None,
) -> Path:
    d = cache_dir(cache_dir_path)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{key}.json"
    path.write_text(json.dumps(_serialize(result), indent=2))
    return path


def _serialize(r: ProbeResult) -> dict:
    return {
        "task_name": r.task_name,
        "adapter_name": r.adapter_name,
        "compute_axis": r.compute_axis,
        "depths": r.depths,
        "compute_grid": [{"value": c.value, "label": c.label} for c in r.compute_grid],
        "accuracy": r.accuracy,
        "n_per_cell": r.n_per_cell,
    }


def _deserialize(data: dict) -> ProbeResult:
    return ProbeResult(
        task_name=data["task_name"],
        adapter_name=data["adapter_name"],
        compute_axis=data["compute_axis"],
        depths=data["depths"],
        compute_grid=[ComputeLevel(c["value"], c["label"]) for c in data["compute_grid"]],
        accuracy=data["accuracy"],
        n_per_cell=data["n_per_cell"],
    )
