"""Unit tests for the Gemini adapter with a mocked SDK."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


def _install_fake_gemini(monkeypatch, response_text: str = "Final answer: 7"):
    """Install a fake `google.genai` module so the adapter imports cleanly."""
    genai_module = types.ModuleType("google.genai")
    types_module = types.ModuleType("google.genai.types")
    google_pkg = types.ModuleType("google")
    google_pkg.genai = genai_module
    genai_module.types = types_module

    class FakeThinkingConfig:
        def __init__(self, thinking_budget):
            self.thinking_budget = thinking_budget

    class FakeGenerateContentConfig:
        def __init__(self, system_instruction=None, thinking_config=None):
            self.system_instruction = system_instruction
            self.thinking_config = thinking_config

    types_module.ThinkingConfig = FakeThinkingConfig
    types_module.GenerateContentConfig = FakeGenerateContentConfig

    response = MagicMock()
    response.text = response_text
    # Match google-genai's `usage_metadata` shape.
    usage_md = MagicMock()
    usage_md.prompt_token_count = 100
    usage_md.candidates_token_count = 50
    usage_md.thoughts_token_count = 200
    response.usage_metadata = usage_md

    class FakeClient:
        def __init__(self, *a, **kw):
            self.models = MagicMock()
            self.models.generate_content = MagicMock(return_value=response)

    genai_module.Client = FakeClient

    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)


def test_gemini_predict(monkeypatch):
    _install_fake_gemini(monkeypatch, response_text="Step by step ...\nFinal answer: 7")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")

    from depth_lens.adapters.base import ComputeLevel
    from depth_lens.adapters.gemini_adapter import GeminiAdapter

    adapter = GeminiAdapter(model="gemini-2.5-flash", task_name="k-hop")
    preds = adapter.predict(["3 add5 ="], ComputeLevel(1024, "think=1024"))
    assert len(preds) == 1
    assert preds[0].text == "7"
    assert preds[0].metadata["thinking_budget_tokens"] == 1024
    # Verify usage was captured from usage_metadata, with thinking folded
    # into output (Gemini bills thinking at the output rate).
    u = preds[0].metadata["usage"]
    assert u == {"input_tokens": 100, "output_tokens": 250, "thinking_tokens": 200}


def test_gemini_missing_key(monkeypatch):
    _install_fake_gemini(monkeypatch)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    from depth_lens.adapters.gemini_adapter import GeminiAdapter

    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        GeminiAdapter()


def test_gemini_default_grid(monkeypatch):
    _install_fake_gemini(monkeypatch)
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")

    from depth_lens.adapters.gemini_adapter import GeminiAdapter

    adapter = GeminiAdapter()
    grid = adapter.default_compute_grid()
    assert [c.value for c in grid] == [1024, 2048, 4096, 8192, 16384]
    assert grid[0].label == "think=1024"
    assert adapter.compute_axis_name == "thinking_budget_tokens"
