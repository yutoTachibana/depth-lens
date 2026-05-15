"""
Core probe engine: sweep depth × compute, score, report.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from statistics import median
from typing import Iterable

import numpy as np
from tqdm import tqdm

from depth_lens.adapters.base import ComputeLevel, ModelAdapter
from depth_lens.tasks.base import Task


def _extract_token_usage(metadata: dict) -> dict[str, int]:
    """
    Pull token counts out of adapter-specific metadata into a normalized
    {input, output, thinking} dict. Returns {} when nothing is reported
    (e.g., local OpenMythos predictions don't have token bookkeeping).
    """
    if not metadata:
        return {}
    usage = metadata.get("usage")
    out: dict[str, int] = {}
    if isinstance(usage, dict):
        if "input_tokens" in usage:
            out["input"] = int(usage["input_tokens"])
        if "output_tokens" in usage:
            out["output"] = int(usage["output_tokens"])
        if "prompt_tokens" in usage:
            out["input"] = out.get("input", 0) + int(usage["prompt_tokens"])
        if "completion_tokens" in usage:
            out["output"] = out.get("output", 0) + int(usage["completion_tokens"])
        if "thinking_tokens" in usage:
            out["thinking"] = int(usage["thinking_tokens"])
    # Anthropic adapter records explicit thinking_chars; convert later.
    if "thinking_chars" in metadata and "thinking" not in out:
        # Rough thinking-token estimate: ~4 chars per token.
        out["thinking"] = max(1, int(metadata["thinking_chars"]) // 4)
    return out


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
    # Optional cost / latency tracking. None when an adapter doesn't report
    # usage (e.g., OpenMythos local). When present:
    #   - latency_per_cell[d][c] = median wall-clock seconds per prediction
    #   - tokens_per_cell[d][c]  = {'input': mean_input_tok, 'output': mean_output_tok}
    latency_per_cell: list[list[float]] | None = None
    tokens_per_cell: list[list[dict]] | None = None

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

    def cost_per_cell(self, pricing: dict[str, float]) -> np.ndarray | None:
        """
        Estimated cost in USD *per prediction*, given a pricing dict.

        pricing is a dict like {"input": 15.0, "output": 75.0} where the values
        are USD per **million** tokens. Returns a (D, C) numpy array, or None if
        the result has no token data.
        """
        if self.tokens_per_cell is None:
            return None
        out = np.zeros((len(self.depths), len(self.compute_grid)), dtype=float)
        for i, row in enumerate(self.tokens_per_cell):
            for j, cell in enumerate(row):
                cost = 0.0
                for key, rate in pricing.items():
                    cost += cell.get(key, 0) * rate / 1_000_000.0
                out[i, j] = cost
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

    latency_rows: list[list[float]] = []
    tokens_rows: list[list[dict]] = []
    any_latency = False
    any_tokens = False

    for d in depths_list:
        row: list[float] = []
        lat_row: list[float] = []
        tok_row: list[dict] = []
        instances = instances_per_depth[d]
        for c in compute_grid:
            correct = 0
            batch_latencies: list[float] = []
            tok_sums: dict[str, int] = {}
            n_seen = 0
            for s in range(0, n_samples, batch_size):
                batch = instances[s : s + batch_size]
                t0 = time.perf_counter()
                preds = adapter.predict([x.prompt for x in batch], c)
                dt = time.perf_counter() - t0
                # Per-prediction wall-clock estimate.
                if preds:
                    batch_latencies.append(dt / len(preds))
                scores = task.score_batch(batch, [p.text for p in preds])
                correct += int(sum(scores))
                # Aggregate token usage if adapter reports it.
                for p in preds:
                    for key in _extract_token_usage(p.metadata):
                        tok_sums[key] = tok_sums.get(key, 0) + _extract_token_usage(p.metadata)[key]
                    n_seen += 1
            row.append(correct / n_samples)
            lat = median(batch_latencies) if batch_latencies else 0.0
            lat_row.append(lat)
            if batch_latencies:
                any_latency = True
            mean_toks: dict = {}
            for k, v in tok_sums.items():
                mean_toks[k] = v / max(n_seen, 1)
            tok_row.append(mean_toks)
            if mean_toks:
                any_tokens = True
            iterator.update(1)
            if verbose:
                iterator.set_postfix({"d": d, "c": str(c), "acc": f"{row[-1]:.2f}"})
        result.accuracy.append(row)
        latency_rows.append(lat_row)
        tokens_rows.append(tok_row)

    if any_latency:
        result.latency_per_cell = latency_rows
    if any_tokens:
        result.tokens_per_cell = tokens_rows

    iterator.close()

    if use_cache:
        from depth_lens.cache import save_cached

        save_cached(key, result, cache_dir)

    return result
