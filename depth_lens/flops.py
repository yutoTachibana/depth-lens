"""
Per-inference FLOPs estimation across the 3 inference-compute paradigms
depth-lens compares.

We use a single canonical formula and document its assumptions; the goal is
NOT vendor-defensible cost accounting (that's pricing.py) but a *fair
architectural-effort axis* so the Pareto plot for v2.0's scaling-law finding
isn't dominated by wall-clock differences between local GPU and network APIs.

Formula
-------
For a decoder-only transformer of N params decoding T_out tokens given
T_in input tokens, the standard "forward-pass FLOPs ≈ 2N per output token"
approximation (Kaplan 2020; Hoffmann/Chinchilla 2022) gives:

    flops_per_call ≈ 2 * N * (T_in + T_out)

We extend this with a `loops` multiplier for latent recursion (OpenMythos):

    flops_per_call ≈ 2 * N * (T_in + T_out) * n_loops

For self-hosted vLLM models this works directly with the known param count.
For OpenMythos we count parameters from the live model.
For closed APIs we use a per-vendor estimated N (published or community-
reported) with an explicit "approx" flag in the returned dict so downstream
plotting can mark these points.

What this is NOT
----------------
- It does not account for MoE sparsity (active vs total experts). For
  OpenMythos with `n_experts_per_tok=2` and `n_experts=16` the active
  param count per token is ~1/8 of total. We currently use total params
  (overestimates FLOPs for MoE). The bias is consistent within a paradigm
  so the SHAPE of the scaling curve is still informative.
- It does not account for attention quadratic terms (negligible for the
  short sequence lengths depth-lens probes).
- API estimates use generation-2 param counts from public model cards
  where available, with reasonable guesses for unannounced models. Errors
  of factor ~2× are possible. See `API_PARAM_ESTIMATES` for sources.
"""

from __future__ import annotations

#: Estimated total parameter counts for closed API models. Sources:
#: - openai:gpt-5-mini, openai:gpt-5: community estimates (no official #s);
#:   placeholder values chosen to put gpt-5 at a frontier-typical scale.
#: - openai:o3, o3-mini, o4-mini: similar — official architecture is opaque;
#:   we use rough community-replication numbers for plot-axis placement.
#: - anthropic:claude-*-{haiku,sonnet,opus}: Anthropic does not publish
#:   parameter counts. We use the typical "small/mid/large frontier tier"
#:   sizes that the per-token pricing suggests.
#: - gemini:*: similarly opaque; estimates based on observed cost tiers.
#:
#: ALL of these are estimates good to roughly a factor of 2×. They exist to
#: place API points on a FLOPs axis for comparison; do NOT cite as
#: authoritative.
API_PARAM_ESTIMATES: dict[str, float] = {
    # Anthropic — small / mid / large frontier
    "anthropic:claude-haiku-4-5":         8e9,    # ~8 B
    "anthropic:claude-sonnet-4-6":        70e9,   # ~70 B
    "anthropic:claude-opus-4-7":          400e9,  # ~400 B
    "anthropic:claude-sonnet-4-20250514": 70e9,
    "anthropic:claude-opus-4-1-20250805": 400e9,
    # OpenAI
    "openai:gpt-5":                       400e9,
    "openai:gpt-5-mini":                  20e9,
    "openai:o4-mini":                     20e9,
    "openai:o3":                          200e9,
    "openai:o3-mini":                     20e9,
    # Google Gemini
    "gemini:gemini-2.5-flash":            10e9,
    "gemini:gemini-2.5-pro":              200e9,
    "gemini:gemini-3.1-flash-lite":       3e9,
    "gemini:gemini-3.1-pro-preview":      200e9,
}


#: Known/published parameter counts for vLLM-served self-hosted models.
#: These are accurate (from the model cards) rather than estimates.
VLLM_PARAM_COUNTS: dict[str, float] = {
    "vllm:hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4": 8.03e9,
    "vllm:meta-llama/Meta-Llama-3.1-8B-Instruct":              8.03e9,
    "vllm:deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":          1.5e9,
    "vllm:deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":            7.6e9,
}


def estimate_flops_per_call(
    adapter_spec: str,
    input_tokens: int,
    output_tokens: int,
    *,
    openmythos_params: int | None = None,
    n_loops: int = 1,
) -> dict:
    """
    Return a FLOPs-per-inference estimate plus metadata about the source.

    Args:
        adapter_spec: full adapter spec string (e.g. "anthropic:claude-haiku-4-5",
            "vllm:meta-llama/Meta-Llama-3.1-8B-Instruct", "openmythos").
        input_tokens: prompt token count.
        output_tokens: response token count (includes thinking for APIs
            that emit them as separate tokens).
        openmythos_params: required when adapter_spec == "openmythos"; the
            number of parameters in the trained model.
        n_loops: latent-recursion loop count (OpenMythos n_loops or vLLM
            equivalents). Applied as a multiplier on top of the per-token
            FLOPs estimate.

    Returns:
        dict with keys:
            flops              : float, total FLOPs for the call
            params             : float, estimated/known parameter count used
            tokens             : int, total tokens (input + output)
            n_loops            : int, loop multiplier used
            source             : 'measured' | 'published' | 'estimated'
            paradigm           : 'token_cot' | 'self_hosted' | 'looped'
    """
    total_tokens = int(input_tokens) + int(output_tokens)

    if adapter_spec == "openmythos":
        if openmythos_params is None:
            raise ValueError("openmythos_params required for OpenMythos FLOPs")
        params = float(openmythos_params)
        return {
            "flops": 2.0 * params * total_tokens * max(1, int(n_loops)),
            "params": params,
            "tokens": total_tokens,
            "n_loops": int(n_loops),
            "source": "measured",
            "paradigm": "looped",
        }

    if adapter_spec in VLLM_PARAM_COUNTS:
        params = VLLM_PARAM_COUNTS[adapter_spec]
        return {
            "flops": 2.0 * params * total_tokens * max(1, int(n_loops)),
            "params": params,
            "tokens": total_tokens,
            "n_loops": int(n_loops),
            "source": "published",
            "paradigm": "self_hosted",
        }

    if adapter_spec in API_PARAM_ESTIMATES:
        params = API_PARAM_ESTIMATES[adapter_spec]
        return {
            "flops": 2.0 * params * total_tokens,
            "params": params,
            "tokens": total_tokens,
            "n_loops": 1,
            "source": "estimated",  # caveat: API param counts are estimates
            "paradigm": "token_cot",
        }

    # Unknown spec — return None-ish so plot code can skip cleanly.
    return {
        "flops": None,
        "params": None,
        "tokens": total_tokens,
        "n_loops": int(n_loops),
        "source": "unknown",
        "paradigm": "unknown",
    }


def paradigm_of(adapter_spec: str) -> str:
    """Quick classifier for plot grouping."""
    if adapter_spec == "openmythos":
        return "looped"
    if adapter_spec.startswith("vllm:") or adapter_spec.startswith("hf:"):
        return "self_hosted"
    return "token_cot"
