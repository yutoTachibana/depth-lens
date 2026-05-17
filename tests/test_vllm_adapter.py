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


def test_vllm_max_tokens_axis_default_grid(monkeypatch):
    _install_fake_openai(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from depth_lens.adapters.vllm_adapter import VLLMAdapter

    adapter = VLLMAdapter(model="m", compute_axis="max_tokens")
    assert adapter.compute_axis_name == "max_tokens"
    grid = adapter.default_compute_grid()
    assert [c.label for c in grid] == ["max_tokens=256", "max_tokens=1024", "max_tokens=4096"]
    assert [c.value for c in grid] == [256, 1024, 4096]


def test_vllm_max_tokens_axis_predict_passes_max_tokens(monkeypatch):
    """When compute_axis='max_tokens', the request must NOT include
    reasoning_effort (which would error on Llama-3-8B-Instruct etc.) and
    must include max_tokens."""
    captured = _install_fake_openai(monkeypatch, response_text="Final answer: 7")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from depth_lens.adapters.base import ComputeLevel
    from depth_lens.adapters.vllm_adapter import VLLMAdapter

    adapter = VLLMAdapter(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        compute_axis="max_tokens",
        task_name="k-hop",
    )
    preds = adapter.predict(["3 add1 add5"], ComputeLevel(1024, "max_tokens=1024"))
    assert len(preds) == 1
    assert preds[0].text == "7"
    assert preds[0].metadata["max_tokens"] == 1024
    # The compute_axis name must propagate (used downstream by plots / cache).
    assert adapter.compute_axis_name == "max_tokens"
    # The init kwargs captured should NOT have anything about reasoning effort.
    init_kwargs = captured.get("init_kwargs", {})
    assert init_kwargs.get("base_url") == "http://localhost:8000/v1"


def test_vllm_max_tokens_grid_explicit(monkeypatch):
    _install_fake_openai(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from depth_lens.adapters.vllm_adapter import VLLMAdapter

    adapter = VLLMAdapter(model="m", compute_axis="max_tokens", compute_grid=[128, 512])
    grid = adapter.default_compute_grid()
    assert [c.label for c in grid] == ["max_tokens=128", "max_tokens=512"]
    assert [c.value for c in grid] == [128, 512]


def test_vllm_invalid_compute_axis_raises(monkeypatch):
    _install_fake_openai(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from depth_lens.adapters.vllm_adapter import VLLMAdapter

    import pytest

    with pytest.raises(ValueError, match="compute_axis"):
        VLLMAdapter(model="m", compute_axis="loops")
