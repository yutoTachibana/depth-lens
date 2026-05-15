"""
Core probe engine: sweep depth × compute, score, report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from tqdm import tqdm

from depth_lens.adapters.base import ComputeLevel, ModelAdapter
from depth_lens.tasks.base import Task


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion. More accurate than the
    normal approximation, especially for accuracy near 0 or 1, with no extra
    cost. Returns (lower, upper) bounds in [0, 1].
    """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(max(0.0, p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


@dataclass
class ProbeResult:
    """Result of one probe run: accuracy on a (depth × compute) grid."""

    task_name: str
    adapter_name: str
    compute_axis: str
    depths: list[int]
    compute_grid: list[ComputeLevel]
    # accuracy[d_idx][c_idx] in [0,1]
    accuracy: list[list[float]] = field(default_factory=list)
    # n_samples scored at each cell (for CI estimation)
    n_per_cell: int = 0

    def as_array(self) -> np.ndarray:
        """Return accuracy as a (D, C) numpy array."""
        return np.array(self.accuracy, dtype=float)

    def best_compute_for(self, depth: int) -> ComputeLevel:
        di = self.depths.index(depth)
        ci = int(np.argmax(self.accuracy[di]))
        return self.compute_grid[ci]

    def effective_depth(self, threshold: float = 0.5) -> int | None:
        """
        The largest depth at which *some* compute level achieves
        accuracy ≥ threshold. None if no depth clears the bar.
        """
        A = self.as_array()
        ok = (A.max(axis=1) >= threshold)
        if not ok.any():
            return None
        idx = int(np.where(ok)[0].max())
        return self.depths[idx]

    def ci(self, z: float = 1.96) -> np.ndarray:
        """
        Wilson 95%-CI bounds for every (depth, compute) cell.

        Returns an array of shape (D, C, 2) where the last axis is (lower, upper).
        """
        A = self.as_array()
        n = self.n_per_cell
        out = np.zeros((*A.shape, 2), dtype=float)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                k = int(round(float(A[i, j]) * n))
                lo, hi = wilson_ci(k, n, z=z)
                out[i, j, 0] = lo
                out[i, j, 1] = hi
        return out

    def overthinking(self, depth: int, tolerance: float = 0.02) -> dict | None:
        """
        Detect overthinking at a depth: peak compute is not the max compute,
        and there's a real drop after the peak.

        Returns a small report dict or None if no overthinking detected.
        """
        di = self.depths.index(depth)
        row = self.as_array()[di]
        peak_idx = int(np.argmax(row))
        if peak_idx == len(row) - 1:
            return None
        drop = float(row[peak_idx] - row[-1])
        if drop <= tolerance:
            return None
        return {
            "depth": depth,
            "peak_compute": self.compute_grid[peak_idx],
            "peak_accuracy": float(row[peak_idx]),
            "last_compute": self.compute_grid[-1],
            "last_accuracy": float(row[-1]),
            "drop": drop,
        }


def probe(
    adapter: ModelAdapter,
    task: Task,
    depths: Iterable[int],
    compute_grid: Iterable[ComputeLevel] | None = None,
    n_samples: int = 256,
    batch_size: int = 64,
    seed: int = 0,
    verbose: bool = True,
    use_cache: bool = True,
    cache_dir: str | None = None,
) -> ProbeResult:
    """
    Run a full depth × compute sweep and return a ProbeResult.

    Args:
        adapter      -- ModelAdapter exposing predict(prompts, compute_level)
        task         -- Task that generates ProbeInstances at each depth
        depths       -- list of integer task depths to probe
        compute_grid -- list of ComputeLevels for the sweep. Falls back to
                        adapter.default_compute_grid() if None.
        n_samples    -- examples per (depth, compute) cell
        batch_size   -- batch size for adapter calls (depends on model + memory)
        seed         -- seed for reproducible task generation
        verbose      -- tqdm progress bar
    """
    depths_list = list(depths)
    if compute_grid is None:
        compute_grid = adapter.default_compute_grid()
    else:
        compute_grid = list(compute_grid)

    # Try cache before doing any work.
    if use_cache:
        from depth_lens.cache import cache_key, load_cached

        key = cache_key(
            adapter_name=adapter.name,
            task_name=task.name,
            depths=depths_list,
            compute_grid=compute_grid,
            n_samples=n_samples,
            seed=seed,
        )
        cached = load_cached(key, cache_dir)
        if cached is not None:
            if verbose:
                print(f"[probe] cache hit: {key}")
            return cached

    result = ProbeResult(
        task_name=task.name,
        adapter_name=adapter.name,
        compute_axis=adapter.compute_axis_name,
        depths=depths_list,
        compute_grid=list(compute_grid),
        n_per_cell=n_samples,
    )

    # Pre-generate one set of instances per depth, reused across compute levels
    # so the comparison is apples-to-apples.
    instances_per_depth = {
        d: task.generate(d, n_samples, seed=seed + d) for d in depths_list
    }

    total_cells = len(depths_list) * len(compute_grid)
    iterator = tqdm(total=total_cells, disable=not verbose, desc=f"probe {adapter.name}/{task.name}")

    for d in depths_list:
        row: list[float] = []
        instances = instances_per_depth[d]
        for c in compute_grid:
            correct = 0
            for s in range(0, n_samples, batch_size):
                batch = instances[s : s + batch_size]
                preds = adapter.predict([x.prompt for x in batch], c)
                scores = task.score_batch(batch, [p.text for p in preds])
                correct += int(sum(scores))
            row.append(correct / n_samples)
            iterator.update(1)
            if verbose:
                iterator.set_postfix({"d": d, "c": str(c), "acc": f"{row[-1]:.2f}"})
        result.accuracy.append(row)

    iterator.close()

    if use_cache:
        from depth_lens.cache import save_cached

        save_cached(key, result, cache_dir)

    return result
