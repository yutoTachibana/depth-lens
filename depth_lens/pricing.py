"""
Default pricing data and cost-from-tokens / cost-from-latency helpers.

Two pricing schemas are supported, distinguished by their keys:

- **Token-based** (API models): ``{"input": 1.00, "output": 5.00}`` where
  values are USD per 1M tokens. Used for hosted APIs (Anthropic, OpenAI,
  Gemini). Extended-thinking tokens are billed as output tokens.

- **GPU-hour-based** (self-hosted vLLM / SGLang / TGI / OpenMythos):
  ``{"gpu_hourly": 0.50, "gpus": 1}`` where ``gpu_hourly`` is USD per
  GPU-hour and ``gpus`` is the number of GPUs the server uses. Cost per
  call is ``latency_seconds × gpu_hourly × gpus / 3600``. This makes
  self-hosted models comparable to API models on the same cost axis.

The dict keys match the adapter spec strings used elsewhere
(``anthropic:claude-haiku-4-5``, ``vllm:meta-llama/Meta-Llama-3-8B-Instruct``,
etc.) so a single lookup handles all vendors uniformly. For self-hosted
models the spec isn't known ahead of time — see ``gpu_hourly_fallback``
below for the convenience path.

Last reviewed: 2026-05-17.
"""

from __future__ import annotations

import json
from pathlib import Path

# (input USD/1M, output USD/1M) — output includes thinking tokens.
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    # ---- Anthropic ----
    "anthropic:claude-haiku-4-5":        {"input": 1.00,  "output": 5.00},
    "anthropic:claude-sonnet-4-6":       {"input": 3.00,  "output": 15.00},
    "anthropic:claude-opus-4-7":         {"input": 15.00, "output": 75.00},
    # Older Anthropic — still callable, useful for regression comparisons
    "anthropic:claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "anthropic:claude-opus-4-1-20250805": {"input": 15.00, "output": 75.00},
    # ---- OpenAI ----
    "openai:gpt-5":                      {"input": 1.25,  "output": 10.00},
    "openai:gpt-5-mini":                 {"input": 0.25,  "output": 2.00},
    "openai:o4-mini":                    {"input": 1.10,  "output": 4.40},
    "openai:o3":                         {"input": 2.00,  "output": 8.00},
    "openai:o3-mini":                    {"input": 1.10,  "output": 4.40},
    # ---- Google ----
    "gemini:gemini-2.5-flash":           {"input": 0.30,  "output": 2.50},
    "gemini:gemini-2.5-pro":             {"input": 1.25,  "output": 10.00},
    "gemini:gemini-3.1-flash-lite":      {"input": 0.10,  "output": 0.40},
    "gemini:gemini-3.1-pro-preview":     {"input": 1.25,  "output": 10.00},
}


# Convenience for self-hosted: if a `vllm:*` (or other self-hosted) spec
# has no explicit pricing entry, treat it as a single-GPU server at this
# hourly rate. The CLI's --gpu-hourly-rate flag overrides this value.
# Default chosen as a midpoint between AWS g5 spot (~$0.30/hr) and
# on-demand (~$1.00/hr) for "average cloud GPU".
DEFAULT_GPU_HOURLY_RATE: float = 0.50


def is_gpu_hour_pricing(pricing: dict | None) -> bool:
    """Return True if `pricing` is a GPU-hour schema (vs token schema)."""
    return pricing is not None and "gpu_hourly" in pricing


def get_pricing(model_spec: str, override: dict | None = None) -> dict | None:
    """
    Return pricing for a model spec.

    Lookup order:
      1. override dict if provided (full spec match)
      2. DEFAULT_PRICING
      3. None if the model isn't in either

    The returned dict is either token-based ({input, output}) or GPU-hour-based
    ({gpu_hourly, gpus}). Callers should branch on `is_gpu_hour_pricing()` or
    let `ProbeResult.cost_per_cell` handle it.
    """
    if override is not None and model_spec in override:
        return override[model_spec]
    return DEFAULT_PRICING.get(model_spec)


def gpu_hour_pricing(gpu_hourly: float, gpus: int = 1) -> dict:
    """Construct a GPU-hour pricing dict (convenience constructor)."""
    return {"gpu_hourly": float(gpu_hourly), "gpus": int(gpus)}


def maybe_gpu_hour_fallback(
    model_spec: str,
    pricing: dict | None,
    gpu_hourly_rate: float | None,
) -> dict | None:
    """
    If `pricing` is None and the spec looks self-hosted (`vllm:*`,
    `hf:*`, `openmythos`), fall back to a GPU-hour pricing dict using
    the supplied or default hourly rate. Returns the original `pricing`
    otherwise.
    """
    if pricing is not None:
        return pricing
    rate = gpu_hourly_rate if gpu_hourly_rate is not None else DEFAULT_GPU_HOURLY_RATE
    if model_spec.startswith(("vllm:", "hf:")) or model_spec == "openmythos":
        return gpu_hour_pricing(rate, gpus=1)
    return None


def load_pricing_file(path: str | Path) -> dict:
    """Load a JSON pricing override file. Validates that each entry is either
    a token-based ({input, output}) or GPU-hour-based ({gpu_hourly[, gpus]})
    schema."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for k, v in data.items():
        if not isinstance(v, dict):
            raise ValueError(f"Pricing entry for {k!r} must be a dict, got {v!r}")
        has_token = "input" in v and "output" in v
        has_gpu = "gpu_hourly" in v
        if not (has_token or has_gpu):
            raise ValueError(
                f"Pricing entry for {k!r} must be {{'input', 'output'}} "
                f"(token-based) or {{'gpu_hourly'[, 'gpus']}} (GPU-hour-based), "
                f"got {v!r}"
            )
    return data
