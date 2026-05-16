"""
Default pricing data and cost-from-tokens helpers.

Prices are USD per 1 million tokens. We treat extended-thinking tokens as
output tokens (the Anthropic billing model). Pricing changes; override via
`--pricing path/to/pricing.json` on the CLI when reality moves.

The dict keys match the adapter spec strings used elsewhere
(`anthropic:claude-haiku-4-5`, `openai:o4-mini`, etc.) so a single lookup
handles all vendors uniformly.

Last reviewed: 2026-05-16. Pricing for newer / preview models is sometimes
not yet posted; we use best-effort estimates and tag them.
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


def get_pricing(model_spec: str, override: dict | None = None) -> dict | None:
    """
    Return `{'input': ..., 'output': ...}` USD per 1M tokens for a model.

    Lookup order:
      1. override dict if provided (full spec match)
      2. DEFAULT_PRICING
      3. None if the model isn't in either
    """
    if override is not None and model_spec in override:
        return override[model_spec]
    return DEFAULT_PRICING.get(model_spec)


def load_pricing_file(path: str | Path) -> dict:
    """Load a JSON pricing override file. Same schema as DEFAULT_PRICING."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for k, v in data.items():
        if not isinstance(v, dict) or "input" not in v or "output" not in v:
            raise ValueError(
                f"Pricing entry for {k!r} must be {{'input': float, 'output': float}}, got {v!r}"
            )
    return data
