"""
OpenAI o-series adapter.

Wraps OpenAI reasoning models (o3, o3-mini, o4-mini, gpt-5, …) with
`reasoning_effort` as the compute knob. The OpenAI API exposes a coarse,
discrete grid rather than a continuous token budget like Anthropic.

Requires:
    pip install openai
    export OPENAI_API_KEY=...

Notes:
- Different models support different effort levels. The adapter accepts a list
  and silently skips levels that aren't supported by the chosen model (the API
  raises and we treat the prediction as empty / wrong).
- Like the Anthropic adapter, this is single-threaded; async batching is v1.0.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from depth_lens.adapters.base import ComputeLevel, ModelAdapter, Prediction

if TYPE_CHECKING:
    pass


_TASK_INSTRUCTIONS: dict[str, str] = {
    "k-hop": (
        "You are computing modular arithmetic on Z/23Z (all results in [0, 22]).\n"
        "Operators:\n"
        "  add1: x -> (x + 1) mod 23\n"
        "  add5: x -> (x + 5) mod 23\n"
        "  mul2: x -> (2 * x) mod 23\n"
        "  mul3: x -> (3 * x) mod 23\n"
        "Apply the operators left to right starting from the leading integer. "
        "On the final line write exactly `Final answer: <integer>`."
    ),
    "parity": (
        "Compute the parity (XOR) of the given binary string. "
        "On the final line write exactly `Final answer: 0` or `Final answer: 1`."
    ),
    "graph-reach": (
        "Decide whether the goal node is reachable from the start node in the directed graph. "
        "On the final line write exactly `Final answer: yes` or `Final answer: no`."
    ),
}


# Coarse effort grid. Treat label as the comparable value (mapped to a numeric
# rank under the hood for plot ordering).
_EFFORT_RANK = {"minimal": 0, "low": 1, "medium": 2, "high": 3}


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI reasoning models with reasoning_effort as compute knob."""

    def __init__(
        self,
        model: str = "o4-mini",
        task_name: str | None = None,
        compute_grid: list[str] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        retry_seconds: float = 5.0,
        max_retries: int = 4,
        request_delay: float = 0.0,
        max_concurrent: int = 8,
        adapter_label: str | None = None,
    ):
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "OpenAIAdapter requires `pip install openai`."
            ) from e

        # vLLM and other OpenAI-compatible local servers don't require a key —
        # they pass a sentinel "EMPTY". The base OpenAI adapter still requires
        # one; subclasses (VLLMAdapter) pass the sentinel explicitly.
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Provide --api-key or export OPENAI_API_KEY."
            )

        self._openai = openai
        # base_url override lets this same adapter target a vLLM / SGLang /
        # llama.cpp / TGI server that exposes an OpenAI-compatible API.
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self.name = adapter_label or f"openai:{model}"

        if task_name and task_name in _TASK_INSTRUCTIONS:
            self._instructions = _TASK_INSTRUCTIONS[task_name]
        else:
            self._instructions = (
                "Solve the problem. On the final line write exactly "
                "`Final answer: <answer>`."
            )

        self._compute_grid = compute_grid or ["low", "medium", "high"]
        self._retry_seconds = retry_seconds
        self._max_retries = max_retries
        self._request_delay = request_delay
        self._max_concurrent = max_concurrent

    @property
    def compute_axis_name(self) -> str:
        return "reasoning_effort"

    def default_compute_grid(self) -> list[ComputeLevel]:
        return [
            ComputeLevel(_EFFORT_RANK.get(e, -1), f"effort={e}")
            for e in self._compute_grid
        ]

    def predict(self, prompts: list[str], compute: ComputeLevel) -> list[Prediction]:
        from depth_lens.adapters._concurrency import parallel_map

        # Parse effort from the label "effort=<x>".
        effort = compute.label.split("=", 1)[1] if "=" in compute.label else "medium"

        def one(prompt: str) -> Prediction:
            text, meta = self._one_call(prompt, effort=effort)
            if self._request_delay:
                time.sleep(self._request_delay)
            return Prediction(text=text, metadata=meta)

        return parallel_map(one, prompts, max_workers=self._max_concurrent)

    def _one_call(self, prompt: str, *, effort: str) -> tuple[str, dict]:
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
                    reasoning_effort=effort,
                )
                text = resp.choices[0].message.content or ""
                final = _extract_final_answer(text)
                return final or text, {
                    "reasoning_effort": effort,
                    "model": self._model,
                    "raw_text": text,
                    "usage": getattr(resp, "usage", None).__dict__ if getattr(resp, "usage", None) else None,
                }
            except (openai.RateLimitError, openai.APIStatusError) as e:
                retries += 1
                if retries > self._max_retries:
                    raise
                wait = self._retry_seconds * (2 ** (retries - 1))
                time.sleep(wait)


def _extract_final_answer(text: str) -> str | None:
    import re

    pattern = re.compile(
        r"(?:final\s+answer|answer)\s*[:=]\s*([^\n]+)",
        re.IGNORECASE,
    )
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    return None
