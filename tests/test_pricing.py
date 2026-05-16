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
