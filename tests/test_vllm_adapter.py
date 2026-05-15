"""Unit tests for the vLLM adapter (reuses the OpenAI SDK)."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _install_fake_openai(monkeypatch, response_text: str = "Final answer: 1"):
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

    captured: dict = {}

    class FakeClient:
        def __init__(self, *a, **kw):
            captured["init_kwargs"] = kw
            self.chat = MagicMock()
            self.chat.completions = MagicMock()
            self.chat.completions.create = MagicMock(return_value=_Response())

    fake.OpenAI = FakeClient
    fake.RateLimitError = FakeRateLimitError
    fake.APIStatusError = FakeAPIStatusError
    monkeypatch.setitem(sys.modules, "openai", fake)
    return captured


def test_vllm_adapter_uses_base_url(monkeypatch):
    captured = _install_fake_openai(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from depth_lens.adapters.vllm_adapter import VLLMAdapter

    adapter = VLLMAdapter(
        model="Qwen/Qwen3-Thinking-32B",
        base_url="http://my-vllm:8000/v1",
        task_name="parity",
    )
    assert captured["init_kwargs"]["base_url"] == "http://my-vllm:8000/v1"
    assert adapter.name == "vllm:Qwen/Qwen3-Thinking-32B"


def test_vllm_adapter_predict_roundtrip(monkeypatch):
    _install_fake_openai(monkeypatch, response_text="Trace... Final answer: 1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from depth_lens.adapters.base import ComputeLevel
    from depth_lens.adapters.vllm_adapter import VLLMAdapter

    adapter = VLLMAdapter(model="m", task_name="parity")
    preds = adapter.predict(["1 0 1 0 parity"], ComputeLevel(2, "effort=medium"))
    assert len(preds) == 1
    assert preds[0].text == "1"
    assert preds[0].metadata["reasoning_effort"] == "medium"


def test_vllm_default_compute_axis(monkeypatch):
    _install_fake_openai(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from depth_lens.adapters.vllm_adapter import VLLMAdapter

    adapter = VLLMAdapter(model="m")
    assert adapter.compute_axis_name == "reasoning_effort"
    grid = adapter.default_compute_grid()
    assert [c.label for c in grid] == ["effort=low", "effort=medium", "effort=high"]
