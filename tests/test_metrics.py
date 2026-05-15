"""Unit tests for metrics helpers."""

import pytest

from depth_lens.adapters.base import ComputeLevel
from depth_lens.metrics import ProbeResult, wilson_ci


def test_wilson_ci_extremes():
    # k=0 and k=n should give finite, well-ordered intervals
    lo, hi = wilson_ci(0, 100)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo == 0.0
    lo, hi = wilson_ci(100, 100)
    assert 0.0 <= lo <= hi <= 1.0
    assert hi == pytest.approx(1.0)


def test_wilson_ci_half():
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi
    # Should be a reasonable width — ~0.4..0.6 for n=100
    assert 0.35 < lo < 0.45
    assert 0.55 < hi < 0.65


def test_wilson_ci_narrows_with_n():
    width_small = wilson_ci(5, 10)
    width_large = wilson_ci(500, 1000)
    assert (width_large[1] - width_large[0]) < (width_small[1] - width_small[0])


def test_probe_result_ci_shape():
    r = ProbeResult(
        task_name="t",
        adapter_name="a",
        compute_axis="x",
        depths=[1, 2, 3],
        compute_grid=[ComputeLevel(1, "a"), ComputeLevel(2, "b")],
        accuracy=[[1.0, 0.5], [0.7, 0.8], [0.0, 0.1]],
        n_per_cell=100,
    )
    ci = r.ci()
    assert ci.shape == (3, 2, 2)
    # acc=1.0 with n=100 → CI like (0.96, ~1.0)
    assert ci[0, 0, 1] == pytest.approx(1.0)
    assert 0.9 < ci[0, 0, 0] < 1.0
