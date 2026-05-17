"""Tests for the pricing module."""

import json
import tempfile

from depth_lens.pricing import DEFAULT_PRICING, get_pricing, load_pricing_file


def test_default_pricing_known_models():
    assert get_pricing("anthropic:claude-haiku-4-5")["input"] == 1.0
    assert get_pricing("anthropic:claude-opus-4-7")["output"] == 75.0
    assert get_pricing("openai:o4-mini")["input"] == 1.10
    assert get_pricing("gemini:gemini-3.1-flash-lite")["output"] == 0.4


def test_unknown_model_returns_none():
    assert get_pricing("anthropic:claude-7000") is None


def test_override_wins():
    override = {"anthropic:claude-haiku-4-5": {"input": 99.0, "output": 99.0}}
    assert get_pricing("anthropic:claude-haiku-4-5", override) == {"input": 99.0, "output": 99.0}


def test_load_pricing_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"foo:bar": {"input": 1.5, "output": 7.5}}, f)
        path = f.name
    loaded = load_pricing_file(path)
    assert loaded["foo:bar"]["input"] == 1.5


def test_load_pricing_file_validates_schema():
    import pytest
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"bad": {"only_input": 1.0}}, f)
        path = f.name
    with pytest.raises(ValueError, match="must be"):
        load_pricing_file(path)


def test_default_pricing_has_expected_vendors():
    vendors = {k.split(":", 1)[0] for k in DEFAULT_PRICING}
    assert vendors == {"anthropic", "openai", "gemini"}


def test_gpu_hour_pricing_constructor():
    from depth_lens.pricing import gpu_hour_pricing

    p = gpu_hour_pricing(0.50, gpus=2)
    assert p == {"gpu_hourly": 0.5, "gpus": 2}


def test_is_gpu_hour_pricing():
    from depth_lens.pricing import is_gpu_hour_pricing

    assert is_gpu_hour_pricing({"gpu_hourly": 0.5, "gpus": 1}) is True
    assert is_gpu_hour_pricing({"input": 1.0, "output": 5.0}) is False
    assert is_gpu_hour_pricing(None) is False


def test_maybe_gpu_hour_fallback_for_vllm_spec():
    from depth_lens.pricing import maybe_gpu_hour_fallback

    p = maybe_gpu_hour_fallback("vllm:meta-llama/Meta-Llama-3-8B-Instruct", None, None)
    assert p == {"gpu_hourly": 0.50, "gpus": 1}

    p = maybe_gpu_hour_fallback("vllm:foo", None, 1.25)
    assert p == {"gpu_hourly": 1.25, "gpus": 1}


def test_maybe_gpu_hour_fallback_preserves_explicit_pricing():
    from depth_lens.pricing import maybe_gpu_hour_fallback

    p = maybe_gpu_hour_fallback("anthropic:claude-haiku-4-5", {"input": 1.0, "output": 5.0}, None)
    assert p == {"input": 1.0, "output": 5.0}


def test_maybe_gpu_hour_fallback_skips_api_spec():
    from depth_lens.pricing import maybe_gpu_hour_fallback

    # Spec without explicit pricing AND not self-hosted → None.
    assert maybe_gpu_hour_fallback("openai:gpt-9000-future", None, None) is None


def test_load_pricing_file_accepts_gpu_hour_schema():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"vllm:my-model": {"gpu_hourly": 0.75, "gpus": 1}}, f)
        path = f.name
    loaded = load_pricing_file(path)
    assert loaded["vllm:my-model"]["gpu_hourly"] == 0.75


def test_cost_per_cell_gpu_hour_schema():
    """Cost computation must switch to latency-based when given GPU-hour pricing."""
    from depth_lens.adapters.base import ComputeLevel
    from depth_lens.metrics import ProbeResult

    result = ProbeResult(
        task_name="t",
        adapter_name="vllm:m",
        compute_axis="max_tokens",
        depths=[4],
        compute_grid=[ComputeLevel(256, "max_tokens=256"), ComputeLevel(1024, "max_tokens=1024")],
        n_per_cell=8,
        accuracy=[[0.9, 0.95]],
        latency_per_cell=[[0.5, 2.0]],  # seconds per call
    )
    cost = result.cost_per_cell({"gpu_hourly": 0.50, "gpus": 1})
    assert cost is not None
    # 0.5 sec × $0.50/hr / 3600 = $0.0000694
    # 2.0 sec × $0.50/hr / 3600 = $0.0002778
    assert abs(cost[0, 0] - 0.5 * 0.50 / 3600.0) < 1e-9
    assert abs(cost[0, 1] - 2.0 * 0.50 / 3600.0) < 1e-9


def test_cost_per_cell_returns_none_for_gpu_pricing_without_latency():
    from depth_lens.adapters.base import ComputeLevel
    from depth_lens.metrics import ProbeResult

    result = ProbeResult(
        task_name="t", adapter_name="vllm:m", compute_axis="max_tokens",
        depths=[4], compute_grid=[ComputeLevel(256, "max_tokens=256")],
        n_per_cell=8, accuracy=[[0.9]],
    )
    assert result.cost_per_cell({"gpu_hourly": 0.50, "gpus": 1}) is None


def test_load_pricing_file_rejects_partial_gpu_schema():
    import pytest

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"bad": {"gpus": 1}}, f)  # missing gpu_hourly
        path = f.name
    with pytest.raises(ValueError, match="must be"):
        load_pricing_file(path)
