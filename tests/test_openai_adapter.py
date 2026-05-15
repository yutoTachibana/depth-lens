"""Unit tests for the OpenAI adapter with a mocked SDK."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


def _install_fake_openai(monkeypatch, response_text: str = "Final answer: 7"):
    fake = types.ModuleType("openai")

    class FakeRateLimitError(Exception):
        pass

    class FakeAPIStatusError(Exception):
        pass

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 20

    class _Message:
        content = response_text

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]
        usage = _Usage()

    response = _Response()

    class FakeClient:
        def __init__(self, *a, **kw):
            self.chat = MagicMock()
            self.chat.completions = MagicMock()
            self.chat.completions.create = MagicMock(return_value=response)

    fake.OpenAI = FakeClient
    fake.RateLimitError = FakeRateLimitError
    fake.APIStatusError = FakeAPIStatusError
    monkeypatch.setitem(sys.modules, "openai", fake)
    return fake


def test_openai_predict(monkeypatch):
    _install_fake_openai(monkeypatch, response_text="Let me think... Final answer: 7")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")

    from depth_lens.adapters.base import ComputeLevel
    from depth_lens.adapters.openai_adapter import OpenAIAdapter

    adapter = OpenAIAdapter(model="o4-mini", task_name="k-hop")
    preds = adapter.predict(["3 add5 ="], ComputeLevel(2, "effort=medium"))
    assert len(preds) == 1
    assert preds[0].text == "7"
    assert preds[0].metadata["reasoning_effort"] == "medium"


def test_openai_default_grid(monkeypatch):
    _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "fake")

    from depth_lens.adapters.openai_adapter import OpenAIAdapter

    adapter = OpenAIAdapter()
    grid = adapter.default_compute_grid()
    assert [c.label for c in grid] == ["effort=low", "effort=medium", "effort=high"]
    assert adapter.compute_axis_name == "reasoning_effort"


def test_openai_missing_key(monkeypatch):
    _install_fake_openai(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from depth_lens.adapters.openai_adapter import OpenAIAdapter

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIAdapter()
