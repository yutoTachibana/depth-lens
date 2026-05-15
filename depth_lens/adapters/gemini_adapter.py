"""
Google Gemini thinking-mode adapter.

Wraps `google-genai` (Gemini Developer API) with `thinking_budget` as the
compute knob. Like Anthropic, Gemini exposes a continuous integer budget in
internal tokens.

Requires:
    pip install google-genai
    export GOOGLE_API_KEY=...   (or GEMINI_API_KEY)
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


class GeminiAdapter(ModelAdapter):
    """Adapter for Gemini models with thinking_budget as compute knob."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        task_name: str | None = None,
        compute_grid: list[int] | None = None,
        api_key: str | None = None,
        retry_seconds: float = 5.0,
        max_retries: int = 4,
        request_delay: float = 0.0,
        adapter_label: str | None = None,
    ):
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as e:
            raise ImportError(
                "GeminiAdapter requires `pip install google-genai`."
            ) from e

        api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) not set."
            )

        self._genai = genai
        self._types = genai_types
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self.name = adapter_label or f"gemini:{model}"

        if task_name and task_name in _TASK_INSTRUCTIONS:
            self._instructions = _TASK_INSTRUCTIONS[task_name]
        else:
            self._instructions = (
                "Solve the problem. On the final line write exactly "
                "`Final answer: <answer>`."
            )

        # Gemini's thinking budgets (per docs, model-dependent):
        # gemini-2.5-flash: 0–24576, gemini-2.5-pro: 128–32768
        self._compute_grid = compute_grid or [1024, 2048, 4096, 8192, 16384]
        self._retry_seconds = retry_seconds
        self._max_retries = max_retries
        self._request_delay = request_delay

    @property
    def compute_axis_name(self) -> str:
        return "thinking_budget_tokens"

    def default_compute_grid(self) -> list[ComputeLevel]:
        return [ComputeLevel(v, f"think={v}") for v in self._compute_grid]

    def predict(self, prompts: list[str], compute: ComputeLevel) -> list[Prediction]:
        budget = int(compute.value)
        out: list[Prediction] = []
        for prompt in prompts:
            text, meta = self._one_call(prompt, budget=budget)
            out.append(Prediction(text=text, metadata=meta))
            if self._request_delay:
                time.sleep(self._request_delay)
        return out

    def _one_call(self, prompt: str, *, budget: int) -> tuple[str, dict]:
        types = self._types
        retries = 0
        while True:
            try:
                config = types.GenerateContentConfig(
                    system_instruction=self._instructions,
                    thinking_config=types.ThinkingConfig(thinking_budget=budget),
                )
                resp = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )
                text = getattr(resp, "text", "") or ""
                final = _extract_final_answer(text)
                return final or text, {
                    "thinking_budget_tokens": budget,
                    "model": self._model,
                    "raw_text": text,
                }
            except Exception:
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
