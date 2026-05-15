"""Built-in adapters."""

from depth_lens.adapters.base import ComputeLevel, ModelAdapter, Prediction

__all__ = ["ComputeLevel", "ModelAdapter", "Prediction"]


def get_adapter(name: str, **kwargs):
    """Lazy adapter loader — imports the implementation module on demand so
    optional dependencies (torch, open-mythos, anthropic, ...) aren't required
    just to inspect the package."""
    if name == "openmythos":
        from depth_lens.adapters.openmythos_adapter import OpenMythosAdapter

        return OpenMythosAdapter(**kwargs)
    if name.startswith("hf:"):
        from depth_lens.adapters.hf_adapter import HuggingFaceAdapter

        model_name = name[len("hf:"):]
        return HuggingFaceAdapter(model_name=model_name, **kwargs)
    if name.startswith("anthropic:"):
        from depth_lens.adapters.anthropic_adapter import AnthropicAdapter

        model = name[len("anthropic:"):]
        return AnthropicAdapter(model=model, **kwargs)
    if name.startswith("openai:"):
        from depth_lens.adapters.openai_adapter import OpenAIAdapter

        model = name[len("openai:"):]
        return OpenAIAdapter(model=model, **kwargs)
    if name.startswith("gemini:"):
        from depth_lens.adapters.gemini_adapter import GeminiAdapter

        model = name[len("gemini:"):]
        return GeminiAdapter(model=model, **kwargs)
    if name.startswith("vllm:"):
        from depth_lens.adapters.vllm_adapter import VLLMAdapter

        model = name[len("vllm:"):]
        return VLLMAdapter(model=model, **kwargs)
    raise KeyError(
        f"Unknown adapter {name!r}. Known: 'openmythos', 'hf:<model>', "
        "'anthropic:<model>', 'openai:<model>', 'gemini:<model>', 'vllm:<model>'"
    )
