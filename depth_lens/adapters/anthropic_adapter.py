"""
Anthropic extended-thinking adapter.

Wraps the Anthropic Messages API with extended thinking enabled. The compute
knob is `budget_tokens` — the number of internal "thinking" tokens the model
may emit before composing its final answer.

Requires:
    pip install anthropic
    export ANTHROPIC_API_KEY=...

This adapter is rate-limited and costs money — start with small `n_samples`
when probing. Concurrency is single-threaded in v0.5; v1.0 will add async
batching.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from depth_lens.adapters.base import ComputeLevel, ModelAdapter, Prediction

if TYPE_CHECKING:
    pass


# Task-specific instruction templates. Anthropic prefers natural-language framing.
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


class AnthropicAdapter(ModelAdapter):
    """Adapter for Claude with extended thinking budget as the compute knob."""

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        task_name: str | None = None,
        max_tokens: int = 1024,
        compute_grid: list[int] | None = None,
        api_key: str | None = None,
        retry_seconds: float = 5.0,
        max_retries: int = 4,
        request_delay: float = 0.0,
        max_concurrent: int = 8,
        adapter_label: str | None = None,
    ):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "AnthropicAdapter requires `pip install anthropic`. "
                "If you only want to inspect depth-lens without the SDK, omit this adapter."
            ) from e

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Provide --api-key or export ANTHROPIC_API_KEY."
            )

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self.name = adapter_label or f"anthropic:{model}"

        # Pick instruction template.
        if task_name and task_name in _TASK_INSTRUCTIONS:
            self._instructions = _TASK_INSTRUCTIONS[task_name]
        else:
            self._instructions = (
                "Solve the problem. On the final line write exactly "
                "`Final answer: <answer>`."
            )

        self._max_tokens = max_tokens
        # Extended-thinking budget grid (in tokens). Values must be ≤ max_tokens - some slack.
        self._compute_grid = compute_grid or [1024, 2048, 4096, 8192, 16384]
        self._retry_seconds = retry_seconds
        self._max_retries = max_retries
        self._request_delay = request_delay
        self._max_concurrent = max_concurrent

    @property
    def compute_axis_name(self) -> str:
        return "thinking_budget_tokens"

    def default_compute_grid(self) -> list[ComputeLevel]:
        return [ComputeLevel(v, f"think={v}") for v in self._compute_grid]

    def predict(self, prompts: list[str], compute: ComputeLevel) -> list[Prediction]:
        from depth_lens.adapters._concurrency import parallel_map

        budget = int(compute.value)
        max_out = max(self._max_tokens, budget + 512)

        def one(prompt: str) -> Prediction:
            text, meta = self._one_call(prompt, budget=budget, max_out=max_out)
            if self._request_delay:
                time.sleep(self._request_delay)
            return Prediction(text=text, metadata=meta)

        return parallel_map(one, prompts, max_workers=self._max_concurrent)

    def _one_call(self, prompt: str, *, budget: int, max_out: int) -> tuple[str, dict]:
        anthropic = self._anthropic
        retries = 0
        while True:
            try:
                resp = self._client.messages.create(
                    model=self._model,
                    max_tokens=max_out,
                    system=self._instructions,
                    messages=[{"role": "user", "content": prompt}],
                    thinking={"type": "enabled", "budget_tokens": budget},
                )
                text_block = "".join(
                    b.text for b in resp.content if getattr(b, "type", "") == "text"
                )
                thinking_block = "".join(
                    getattr(b, "thinking", "")
                    for b in resp.content
                    if getattr(b, "type", "") == "thinking"
                )
                # Adapter returns just the final-answer text (already parsed) so
                # the task's lenient scorer can extract it cleanly.
                final = _extract_final_answer(text_block)
                return final or text_block, {
                    "thinking_budget_tokens": budget,
                    "model": self._model,
                    "raw_text": text_block,
                    "thinking_chars": len(thinking_block),
                    "usage": getattr(resp, "usage", None).__dict__ if getattr(resp, "usage", None) else None,
                }
            except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
                retries += 1
                if retries > self._max_retries:
                    raise
                wait = self._retry_seconds * (2 ** (retries - 1))
                time.sleep(wait)


def _extract_final_answer(text: str) -> str | None:
    """Pull a 'Final answer: X' / 'Answer: X' line out of the model output."""
    import re

    pattern = re.compile(
        r"(?:final\s+answer|answer)\s*[:=]\s*([^\n]+)",
        re.IGNORECASE,
    )
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    return None
