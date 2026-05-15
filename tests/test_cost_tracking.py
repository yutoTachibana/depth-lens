"""Tests for cost and latency tracking on ProbeResult."""

from __future__ import annotations

import numpy as np

from depth_lens.adapters.base import ComputeLevel, ModelAdapter, Prediction
from depth_lens.metrics import ProbeResult, _extract_token_usage, probe
from depth_lens.tasks.base import ProbeInstance, Task


def test_extract_anthropic_style_usage():
    md = {"usage": {"input_tokens": 100, "output_tokens": 50, "thinking_tokens": 200}}
    u = _extract_token_usage(md)
    assert u == {"input": 100, "output": 50, "thinking": 200}


def test_extract_openai_style_usage():
    md = {"usage": {"prompt_tokens": 80, "completion_tokens": 30}}
    u = _extract_token_usage(md)
    assert u == {"input": 80, "output": 30}


def test_extract_empty():
    assert _extract_token_usage({}) == {}
    assert _extract_token_usage(None) == {}  # type: ignore[arg-type]


def test_cost_per_cell():
    r = ProbeResult(
        task_name="t",
        adapter_name="a",
        compute_axis="x",
        depths=[1, 2],
        compute_grid=[ComputeLevel(1, "a"), ComputeLevel(2, "b")],
        accuracy=[[1.0, 1.0], [1.0, 1.0]],
        n_per_cell=10,
        tokens_per_cell=[
            [{"input": 100, "output": 50}, {"input": 100, "output": 200}],
            [{"input": 200, "output": 100}, {"input": 200, "output": 400}],
        ],
    )
    # Claude Opus 4.6 approx pricing: $15 in / $75 out per 1M tokens.
    pricing = {"input": 15.0, "output": 75.0}
    cost = r.cost_per_cell(pricing)
    assert cost is not None
    assert cost.shape == (2, 2)
    # cell [0][0]: 100 * 15e-6 + 50 * 75e-6 = 0.0015 + 0.00375 = 0.00525
    np.testing.assert_almost_equal(cost[0, 0], 0.00525)
    # cell [0][1]: 100 * 15e-6 + 200 * 75e-6 = 0.0015 + 0.015 = 0.0165
    np.testing.assert_almost_equal(cost[0, 1], 0.0165)


def test_cost_returns_none_without_tokens():
    r = ProbeResult(
        task_name="t",
        adapter_name="a",
        compute_axis="x",
        depths=[1],
        compute_grid=[ComputeLevel(1, "a")],
        accuracy=[[1.0]],
        n_per_cell=10,
    )
    assert r.cost_per_cell({"input": 15.0, "output": 75.0}) is None


# -- end-to-end probe with a mock adapter -------------------------------------


class _MockTask(Task):
    name = "mock"

    def generate(self, depth, n_samples, seed=0):
        return [
            ProbeInstance(prompt=f"q{i}", target="42", depth=depth)
            for i in range(n_samples)
        ]

    def score(self, instance, prediction):
        return 1.0


class _MockAdapter(ModelAdapter):
    name = "mock"

    @property
    def compute_axis_name(self):
        return "mock_axis"

    def default_compute_grid(self):
        return [ComputeLevel(1, "low"), ComputeLevel(2, "high")]

    def predict(self, prompts, compute):
        # Pretend each prompt cost depends on the compute level.
        in_tok = 50 * int(compute.value)
        out_tok = 100 * int(compute.value)
        return [
            Prediction(
                text="42",
                metadata={"usage": {"input_tokens": in_tok, "output_tokens": out_tok}},
            )
            for _ in prompts
        ]


def test_probe_aggregates_tokens_and_latency():
    task = _MockTask()
    adapter = _MockAdapter()
    result = probe(
        adapter, task, depths=[1, 2], n_samples=4, batch_size=2,
        use_cache=False, verbose=False,
    )
    assert result.tokens_per_cell is not None
    assert result.latency_per_cell is not None
    # 2 depths × 2 compute levels
    assert len(result.tokens_per_cell) == 2
    assert len(result.tokens_per_cell[0]) == 2
    # compute=1 → 50 input, 100 output
    assert result.tokens_per_cell[0][0] == {"input": 50, "output": 100}
    # compute=2 → 100 input, 200 output
    assert result.tokens_per_cell[0][1] == {"input": 100, "output": 200}
    # Latency is positive
    assert result.latency_per_cell[0][0] >= 0


def test_probe_cost_per_cell_end_to_end():
    task = _MockTask()
    adapter = _MockAdapter()
    result = probe(
        adapter, task, depths=[1], n_samples=4, batch_size=2,
        use_cache=False, verbose=False,
    )
    cost = result.cost_per_cell({"input": 15.0, "output": 75.0})
    assert cost is not None
    # compute=1: 50*15e-6 + 100*75e-6 = 0.00075 + 0.0075 = 0.00825
    np.testing.assert_almost_equal(cost[0, 0], 0.00825)
    # compute=2: 100*15e-6 + 200*75e-6 = 0.0015 + 0.015 = 0.0165
    np.testing.assert_almost_equal(cost[0, 1], 0.0165)
