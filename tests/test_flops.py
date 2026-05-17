"""Tests for the FLOPs estimator."""

from __future__ import annotations

import pytest

from depth_lens.flops import (
    API_PARAM_ESTIMATES,
    VLLM_PARAM_COUNTS,
    estimate_flops_per_call,
    paradigm_of,
)


def test_openmythos_basic():
    r = estimate_flops_per_call(
        "openmythos",
        input_tokens=10, output_tokens=1,
        openmythos_params=1_000_000, n_loops=4,
    )
    # 2 * 1e6 * 11 * 4 = 88e6
    assert r["flops"] == pytest.approx(8.8e7)
    assert r["params"] == 1_000_000
    assert r["n_loops"] == 4
    assert r["source"] == "measured"
    assert r["paradigm"] == "looped"


def test_openmythos_requires_params():
    with pytest.raises(ValueError, match="openmythos_params"):
        estimate_flops_per_call("openmythos", input_tokens=10, output_tokens=1)


def test_openmythos_n_loops_multiplier():
    """FLOPs should scale linearly with n_loops for OpenMythos."""
    common = dict(adapter_spec="openmythos", input_tokens=10,
                  output_tokens=1, openmythos_params=1_000_000)
    f1 = estimate_flops_per_call(**common, n_loops=1)["flops"]
    f4 = estimate_flops_per_call(**common, n_loops=4)["flops"]
    f16 = estimate_flops_per_call(**common, n_loops=16)["flops"]
    assert f4 == pytest.approx(4 * f1)
    assert f16 == pytest.approx(16 * f1)


def test_vllm_known_model():
    r = estimate_flops_per_call(
        "vllm:hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        input_tokens=100, output_tokens=200,
    )
    # 2 * 8.03e9 * 300 = 4.818e12
    assert r["flops"] == pytest.approx(2.0 * 8.03e9 * 300)
    assert r["source"] == "published"
    assert r["paradigm"] == "self_hosted"


def test_api_estimated():
    r = estimate_flops_per_call(
        "anthropic:claude-haiku-4-5",
        input_tokens=100, output_tokens=500,
    )
    assert r["flops"] == pytest.approx(2.0 * 8e9 * 600)
    assert r["source"] == "estimated"
    assert r["paradigm"] == "token_cot"
    # n_loops is ignored (always 1) for token-CoT APIs
    assert r["n_loops"] == 1


def test_unknown_spec_returns_none():
    r = estimate_flops_per_call("unknown:foo-bar", input_tokens=10, output_tokens=10)
    assert r["flops"] is None
    assert r["source"] == "unknown"


def test_paradigm_of():
    assert paradigm_of("openmythos") == "looped"
    assert paradigm_of("vllm:meta-llama/Llama-3-8B") == "self_hosted"
    assert paradigm_of("hf:Qwen/Qwen2.5-1.5B") == "self_hosted"
    assert paradigm_of("anthropic:claude-haiku-4-5") == "token_cot"
    assert paradigm_of("openai:gpt-5-mini") == "token_cot"
    assert paradigm_of("gemini:gemini-3.1-flash-lite") == "token_cot"


def test_api_param_estimates_cover_default_pricing():
    """Every API model in DEFAULT_PRICING should have a FLOPs estimate so
    cross-paradigm plots aren't missing data points."""
    from depth_lens.pricing import DEFAULT_PRICING
    for spec in DEFAULT_PRICING:
        assert spec in API_PARAM_ESTIMATES, (
            f"{spec} is in DEFAULT_PRICING but missing from API_PARAM_ESTIMATES"
        )


def test_vllm_param_counts_are_accurate_to_model_card():
    """Sanity: every vLLM spec we list has a plausible parameter count."""
    for spec, params in VLLM_PARAM_COUNTS.items():
        assert 1e8 < params < 1e12, f"{spec}: {params} params outside plausible range"
