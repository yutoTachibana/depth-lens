"""
vLLM (OpenAI-compatible) adapter.

A thin wrapper around `OpenAIAdapter` that points at a local OpenAI-compatible
inference server — vLLM, SGLang, llama.cpp's server, TGI, etc. The compute
knob is `reasoning_effort` when the served model supports it (recent
DeepSeek-R1, Qwen-3-Thinking, etc.), or a no-op single-level grid otherwise.

Typical usage:
    1. Start a vLLM server:
       `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-Thinking-32B`
    2. Probe it:
       `depth-lens probe --model vllm:Qwen/Qwen3-Thinking-32B --task k-hop`
"""

from __future__ import annotations

from depth_lens.adapters.openai_adapter import OpenAIAdapter


class VLLMAdapter(OpenAIAdapter):
    """OpenAI-compatible local server (vLLM / SGLang / TGI / llama.cpp)."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        task_name: str | None = None,
        compute_grid: list[str] | None = None,
        adapter_label: str | None = None,
        **kwargs,
    ):
        # vLLM (and friends) accept any non-empty API key string when auth is off.
        super().__init__(
            model=model,
            task_name=task_name,
            compute_grid=compute_grid,
            api_key="EMPTY",
            base_url=base_url,
            adapter_label=adapter_label or f"vllm:{model}",
            **kwargs,
        )
        self.name = adapter_label or f"vllm:{model}"
