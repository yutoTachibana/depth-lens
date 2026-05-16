"""
HuggingFace causal-LM adapter.

The compute knob is `max_thinking_tokens` — the number of new tokens the model
may emit before its output is parsed for an answer. For instruction-tuned
models, this maps directly to the chain-of-thought / "thinking" budget knob
exposed by most modern reasoning APIs.

The adapter wraps the abstract Task prompt with a short natural-language
instruction header so a generic causal LM can attempt the symbolic task
without task-specific training. This is intentionally simple in v0.1 — more
sophisticated prompt strategies belong in v0.5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from depth_lens.adapters.base import ComputeLevel, ModelAdapter, Prediction

if TYPE_CHECKING:
    pass


# A general instruction header. Tasks can override by setting `task.hf_instructions`.
DEFAULT_INSTRUCTIONS = (
    "You will solve a short symbolic problem. "
    "Compute step by step, then write the final answer on a line of the form "
    '`Final answer: <integer>`.'
)

# Task-specific instruction overrides (registered by task name).
# Kept here so adapters don't pollute the Task ABC. Extend as new tasks land.
_TASK_INSTRUCTIONS: dict[str, str] = {
    "k-hop": (
        "You are computing modular arithmetic on Z/23Z (i.e. all results are reduced mod 23, in [0, 22]).\n"
        "Operators:\n"
        "  add1: x -> (x + 1) mod 23\n"
        "  add5: x -> (x + 5) mod 23\n"
        "  mul2: x -> (2 * x) mod 23\n"
        "  mul3: x -> (3 * x) mod 23\n"
        "Apply the operators left to right starting from the leading integer. "
        "Show each intermediate value, then on a final line write exactly "
        '`Final answer: <integer>` where <integer> is in [0, 22].'
    ),
    "parity": (
        "You are computing the parity (XOR) of a binary string. "
        "The prompt is a space-separated list of bits followed by the word 'parity'. "
        "Compute the cumulative XOR left to right, showing intermediate values, then on a final line write exactly "
        '`Final answer: 0` or `Final answer: 1`.'
    ),
    "graph-reach": (
        "You are asked whether one node is reachable from another in a small directed graph. "
        "The prompt lists edges (`a -> b` means a points to b) separated by `;`, then asks `reach X -> Y ?`. "
        "Trace the reachable set forward from X step by step, then on a final line write exactly "
        '`Final answer: yes` or `Final answer: no`.'
    ),
    "state-tracking": (
        "You are tracking two counters C1 and C2 (both start at 0, all values mod 17, so in [0, 16]). "
        "Operators:\n"
        "  inc1 : C1 = (C1 + 1) mod 17\n"
        "  inc2 : C2 = (C2 + 1) mod 17\n"
        "  swap : (C1, C2) = (C2, C1)\n"
        "  add  : C1 = (C1 + C2) mod 17\n"
        "Apply the operators left to right. The prompt ends with `query <i>` (i is 1 or 2); "
        "report the final value of C<i>. On the final line write exactly `Final answer: <integer>`."
    ),
    "mini-csp": (
        "You are deciding whether a small 2-SAT Boolean formula is satisfiable. "
        "The prompt lists Boolean variables (single letters) and a conjunction of "
        "2-literal clauses written as `( a OR NOT b )`. Your job is to determine "
        "whether *any* assignment of true/false to the variables makes every "
        "clause true. Reason step by step (try assignments or use unit propagation), "
        "then on the final line write exactly `Final answer: yes` or `Final answer: no`."
    ),
}


class HuggingFaceAdapter(ModelAdapter):
    """Adapter around a HuggingFace causal LM with CoT-token-budget as compute knob."""

    def __init__(
        self,
        model_name: str,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        instructions: str | None = None,
        task_name: str | None = None,
        compute_grid: list[int] | None = None,
        adapter_label: str | None = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model_name = model_name
        self.name = adapter_label or f"hf:{model_name.split('/')[-1]}"

        device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        self.device = device
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Causal LMs must pad on the left for correct batched generation.
        self.tokenizer.padding_side = "left"

        # Use .to(device) manually so we don't pull `accelerate` in just for device_map.
        # `dtype` is the post-deprecation kwarg name (transformers ≥ 4.45). Falls
        # back to `torch_dtype` for older versions.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model = self.model.to(device).eval()

        # Pick instruction template
        if instructions:
            self._instructions = instructions
        elif task_name and task_name in _TASK_INSTRUCTIONS:
            self._instructions = _TASK_INSTRUCTIONS[task_name]
        else:
            self._instructions = DEFAULT_INSTRUCTIONS

        self._compute_grid = compute_grid or [16, 32, 64, 128, 256]

    @property
    def compute_axis_name(self) -> str:
        return "max_thinking_tokens"

    def default_compute_grid(self) -> list[ComputeLevel]:
        return [ComputeLevel(v, f"tokens={v}") for v in self._compute_grid]

    def _wrap_prompt(self, prompt: str) -> str:
        # Prefer the model's own chat template when available — gives much
        # better instruction following on small instruct models.
        sys = self._instructions
        user = f"Problem:\n{prompt}\n"
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Fallback: minimal text format
        return f"{sys}\n\n{user}\nSolution:\n"

    @torch.no_grad()
    def predict(self, prompts: list[str], compute: ComputeLevel) -> list[Prediction]:
        wrapped = [self._wrap_prompt(p) for p in prompts]
        enc = self.tokenizer(
            wrapped,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.device)

        gen = self.model.generate(
            **enc,
            max_new_tokens=int(compute.value),
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # Strip the input portion so we only score generated tokens.
        gen_only = gen[:, enc["input_ids"].shape[1]:]
        texts = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        out: list[Prediction] = []
        for t in texts:
            answer = _extract_answer(t)
            out.append(
                Prediction(
                    text=answer if answer is not None else t,
                    metadata={
                        "max_thinking_tokens": int(compute.value),
                        "model": self._model_name,
                        "raw_generation": t,
                        "final_answer_line": _find_final_answer(t),
                    },
                )
            )
        return out

    def teardown(self) -> None:
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _find_final_answer(text: str) -> str | None:
    """Try to locate a 'Final answer: N' line; return it verbatim if found."""
    for line in reversed(text.strip().splitlines()):
        s = line.strip()
        if "final answer" in s.lower():
            return s
    return None


def _extract_answer(text: str) -> str | None:
    """
    Pull just the answer integer out of a verbose CoT generation.

    Priority order:
      1. "Final answer: N" / "Answer: N" (case-insensitive)
      2. The last integer in the text (good fallback for "= 16" endings)
    """
    import re

    # 1. Look for an explicit "(final) answer: N" pattern (last occurrence).
    pattern = re.compile(r"(?:final\s+answer|answer)\s*[:=]\s*(-?\d+)", re.IGNORECASE)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]

    # 2. Fall back to the last integer anywhere in the text.
    ints = re.findall(r"-?\d+", text)
    if ints:
        return ints[-1]

    return None
