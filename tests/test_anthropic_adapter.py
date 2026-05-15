"""Unit tests for the Anthropic adapter with a mocked SDK."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


def _install_fake_anthropic(monkeypatch, response_text: str, thinking_text: str = ""):
    """Install a fake `anthropic` module into sys.modules so the adapter imports cleanly."""
    fake = types.ModuleType("anthropic")

    class FakeRateLimitError(Exception):
        pass

    class FakeAPIStatusError(Exception):
        pass

    class _Block:
        def __init__(self, type_, **fields):
            self.type = type_
            for k, v in fields.items():
                setattr(self, k, v)

    class _Usage:
        def __init__(self):
            self.input_tokens = 10
            self.output_tokens = 20

    response = MagicMock()
    response.content = [
        _Block("thinking", thinking=thinking_text),
        _Block("text", text=response_text),
    ]
    response.usage = _Usage()

    class FakeClient:
        def __init__(self, *a, **kw):
            self.messages = MagicMock()
            self.messages.create = MagicMock(return_value=response)

    fake.Anthropic = FakeClient
    fake.RateLimitError = FakeRateLimitError
    fake.APIStatusError = FakeAPIStatusError
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    return fake


def test_anthropic_adapter_predict(monkeypatch):
    _install_fake_anthropic(
        monkeypatch,
        response_text="Let me think... Final answer: 7",
        thinking_text="(internal reasoning)",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")

    from depth_lens.adapters.anthropic_adapter import AnthropicAdapter
    from depth_lens.adapters.base import ComputeLevel

    adapter = AnthropicAdapter(model="claude-opus-4-7", task_name="k-hop")
    preds = adapter.predict(["3 add5 mul2 ="], ComputeLevel(2048, "think=2048"))

    assert len(preds) == 1
    # The adapter should have stripped to "7" via _extract_final_answer.
    assert preds[0].text == "7"
    assert preds[0].metadata["thinking_budget_tokens"] == 2048
    assert preds[0].metadata["thinking_chars"] == len("(internal reasoning)")


def test_anthropic_adapter_missing_key(monkeypatch):
    _install_fake_anthropic(monkeypatch, response_text="ok")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from depth_lens.adapters.anthropic_adapter import AnthropicAdapter

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicAdapter()


def test_anthropic_adapter_grid(monkeypatch):
    _install_fake_anthropic(monkeypatch, response_text="Final answer: 0")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")

    from depth_lens.adapters.anthropic_adapter import AnthropicAdapter

    adapter = AnthropicAdapter(compute_grid=[100, 200], task_name="parity")
    grid = adapter.default_compute_grid()
    assert [c.value for c in grid] == [100, 200]
    assert grid[0].label == "think=100"
    assert adapter.compute_axis_name == "thinking_budget_tokens"
