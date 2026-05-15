"""Unit tests for the parallel_map helper used by API adapters."""

import time

from depth_lens.adapters._concurrency import parallel_map


def test_parallel_map_preserves_order():
    out = parallel_map(lambda x: x * 2, [1, 2, 3, 4, 5], max_workers=4)
    assert out == [2, 4, 6, 8, 10]


def test_parallel_map_fan_out_speedup():
    """Sanity-check that parallel runs are faster than sequential for IO."""

    def slow(x):
        time.sleep(0.05)
        return x

    t0 = time.perf_counter()
    parallel_map(slow, list(range(8)), max_workers=1)
    seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    parallel_map(slow, list(range(8)), max_workers=8)
    par = time.perf_counter() - t0

    # With 8 workers, 8 × 0.05s should run in well under 0.2s vs ~0.4s sequential.
    assert par < seq * 0.7


def test_parallel_map_single_worker_fallback():
    out = parallel_map(lambda x: x + 1, [10, 20, 30], max_workers=1)
    assert out == [11, 21, 31]


def test_parallel_map_empty():
    assert parallel_map(lambda x: x, [], max_workers=4) == []
