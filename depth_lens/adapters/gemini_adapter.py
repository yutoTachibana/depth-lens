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
    "mini-csp": (
        "Decide whether the small 2-SAT Boolean formula is satisfiable. The prompt "
        "lists Boolean variables and a conjunction of 2-literal clauses. Determine "
        "whether any assignment makes every clause true. On the final line write "
        "exactly `Final answer: yes` or `Final answer: no`."
    ),
    "dict-lookup": (
        "You are given a list of `key = value` pairs followed by `lookup <key>`. "
        "Return the value associated with the queried key. Keys are single letters; "
        "values are single digits 0-9. On the final line write exactly "
        "`Final answer: <digit>`."
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
        max_concurrent: int = 8,
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
                "GOOGLE_API_KEY (or GEMINI_API_KEY) not set. "
                "`export GOOGLE_API_KEY=...` (get one at "
                "https://aistudio.google.com/apikey) or pass api_key= to "
                "GeminiAdapter when using the Python API."
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
        self._max_concurrent = max_concurrent
        # Gemini 3.x replaces `thinking_budget` (int) with `thinking_level`
        # (enum: low/medium/high). Detect by model name; can be flipped at
        # runtime if the API rejects the legacy param.
        self._use_thinking_level = "gemini-3" in model

    @property
    def compute_axis_name(self) -> str:
        return "thinking_budget_tokens"

    def default_compute_grid(self) -> list[ComputeLevel]:
        return [ComputeLevel(v, f"think={v}") for v in self._compute_grid]

    def predict(self, prompts: list[str], compute: ComputeLevel) -> list[Prediction]:
        from depth_lens.adapters._concurrency import parallel_map

        budget = int(compute.value)

        def one(prompt: str) -> Prediction:
            text, meta = self._one_call(prompt, budget=budget)
            if self._request_delay:
                time.sleep(self._request_delay)
            return Prediction(text=text, metadata=meta)

        return parallel_map(one, prompts, max_workers=self._max_concurrent)

    def _one_call(self, prompt: str, *, budget: int) -> tuple[str, dict]:
        types = self._types
        retries = 0
        while True:
            try:
                if self._use_thinking_level:
                    level = _budget_to_level(budget)
                    thinking_cfg = types.ThinkingConfig(thinking_level=level)
                else:
                    thinking_cfg = types.ThinkingConfig(thinking_budget=budget)
                config = types.GenerateContentConfig(
                    system_instruction=self._instructions,
                    thinking_config=thinking_cfg,
                )
                resp = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )
                text = getattr(resp, "text", "") or ""
                final = _extract_final_answer(text)
                # google-genai exposes usage as `usage_metadata`, not `usage`.
                # Translate to the {input_tokens, output_tokens, thinking_tokens}
                # shape that depth_lens.metrics._extract_token_usage recognizes,
                # folding thinking tokens into output for cost calc (Gemini bills
                # thinking tokens at the output rate).
                usage_md = getattr(resp, "usage_metadata", None)
                usage = None
                if usage_md is not None:
                    prompt_tok = getattr(usage_md, "prompt_token_count", 0) or 0
                    cand_tok = getattr(usage_md, "candidates_token_count", 0) or 0
                    think_tok = getattr(usage_md, "thoughts_token_count", 0) or 0
                    usage = {
                        "input_tokens": int(prompt_tok),
                        "output_tokens": int(cand_tok) + int(think_tok),
                        "thinking_tokens": int(think_tok),
                    }
                meta = {
                    "thinking_budget_tokens": budget,
                    "model": self._model,
                    "raw_text": text,
                    "usage": usage,
                }
                if self._use_thinking_level:
                    meta["thinking_level"] = _budget_to_level(budget)
                return final or text, meta
            except TypeError as e:
                # Gemini-3 -> 2.5 fallback path (or vice versa) if SDK rejects
                # the unknown thinking param. Flip once and retry.
                msg = str(e)
                if "thinking_budget" in msg and not self._use_thinking_level:
                    self._use_thinking_level = True
                    continue
                if "thinking_level" in msg and self._use_thinking_level:
                    self._use_thinking_level = False
                    continue
                raise
            except Exception:
                retries += 1
                if retries > self._max_retries:
                    raise
                wait = self._retry_seconds * (2 ** (retries - 1))
                time.sleep(wait)


def _budget_to_level(budget: int) -> str:
    """Map a legacy budget_tokens value to a Gemini 3.x thinking_level enum."""
    if budget <= 2048:
        return "low"
    if budget <= 8192:
        return "medium"
    return "high"


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
