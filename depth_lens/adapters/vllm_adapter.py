"""
vLLM (and other OpenAI-compatible) adapter.

Targets a local OpenAI-compatible inference server — vLLM, SGLang, llama.cpp's
server, TGI, etc. Built on top of the shared `OpenAIAdapter` HTTP path.

Two compute axes are supported, selectable via `compute_axis`:

- ``reasoning_effort`` (default) — for *thinking* models that accept the
  OpenAI reasoning-effort knob (recent DeepSeek-R1 distills, Qwen3-Thinking,
  etc.). Behavior is identical to `OpenAIAdapter`.

- ``max_tokens`` — for *non-thinking* models like Llama-3-8B-Instruct. Sweeps
  ``max_completion_tokens`` so the Pareto plot shows the effect of allowing
  longer free-form CoT in the response itself, without depending on any
  thinking extension.

Typical usage::

    # Thinking model
    depth-lens probe --model vllm:deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
      --compute low,medium,high --task k-hop

    # Non-thinking model (max_tokens axis)
    depth-lens probe --model vllm:meta-llama/Meta-Llama-3-8B-Instruct \
      --compute-axis max_tokens --compute 256,1024,4096 --task k-hop
"""

from __future__ import annotations

import time

from depth_lens.adapters.base import ComputeLevel, Prediction
from depth_lens.adapters.openai_adapter import OpenAIAdapter, _extract_final_answer


class VLLMAdapter(OpenAIAdapter):
    """OpenAI-compatible local server (vLLM / SGLang / TGI / llama.cpp)."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        task_name: str | None = None,
        compute_axis: str = "reasoning_effort",
        compute_grid: list[str] | list[int] | None = None,
        adapter_label: str | None = None,
        **kwargs,
    ):
        if compute_axis not in ("reasoning_effort", "max_tokens"):
            raise ValueError(
                f"compute_axis must be 'reasoning_effort' or 'max_tokens', got {compute_axis!r}"
            )

        # Default grid depends on the chosen axis.
        if compute_grid is None:
            compute_grid = (
                ["low", "medium", "high"]
                if compute_axis == "reasoning_effort"
                else ["256", "1024", "4096"]
            )

        # vLLM (and friends) accept any non-empty API key string when auth is off.
        super().__init__(
            model=model,
            task_name=task_name,
            compute_grid=[str(c) for c in compute_grid],
            api_key="EMPTY",
            base_url=base_url,
            adapter_label=adapter_label or f"vllm:{model}",
            **kwargs,
        )
        self._compute_axis = compute_axis
        self.name = adapter_label or f"vllm:{model}"

    @property
    def compute_axis_name(self) -> str:
        return self._compute_axis

    def default_compute_grid(self) -> list[ComputeLevel]:
        if self._compute_axis == "max_tokens":
            return [
                ComputeLevel(int(t), f"max_tokens={t}") for t in self._compute_grid
            ]
        return super().default_compute_grid()

    def predict(self, prompts: list[str], compute: ComputeLevel) -> list[Prediction]:
        if self._compute_axis != "max_tokens":
            return super().predict(prompts, compute)

        from depth_lens.adapters._concurrency import parallel_map

        # Parse max_tokens from the label "max_tokens=<n>" or use compute.value.
        try:
            max_tokens = int(compute.value)
        except (TypeError, ValueError):
            max_tokens = int(compute.label.split("=", 1)[1])

        def one(prompt: str) -> Prediction:
            text, meta = self._max_tokens_call(prompt, max_tokens=max_tokens)
            if self._request_delay:
                time.sleep(self._request_delay)
            return Prediction(text=text, metadata=meta)

        return parallel_map(one, prompts, max_workers=self._max_concurrent)

    def _max_tokens_call(self, prompt: str, *, max_tokens: int) -> tuple[str, dict]:
        openai = self._openai
        retries = 0
        while True:
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": self._instructions},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                )
                text = resp.choices[0].message.content or ""
                final = _extract_final_answer(text)
                usage_obj = getattr(resp, "usage", None)
                return final or text, {
                    "max_tokens": max_tokens,
                    "model": self._model,
                    "raw_text": text,
                    "usage": usage_obj.__dict__ if usage_obj else None,
                }
            except (openai.RateLimitError, openai.APIStatusError):
                retries += 1
                if retries > self._max_retries:
                    raise
                wait = self._retry_seconds * (2 ** (retries - 1))
                time.sleep(wait)
