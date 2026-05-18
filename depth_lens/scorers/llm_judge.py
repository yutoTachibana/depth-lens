"""
LLM-as-judge scorer.

Calls a separate judge model to grade each (prompt, target, prediction)
triple against a named criterion (or a free-form rubric string). Returns
a binary score (0 or 1) so Wilson 95% CIs still apply cleanly.

Why this exists
---------------
The other bundled scorers (exact / first_int / contains / regex / yes_no)
cover classification and short-answer tasks. They're useless on:

  - "Is this summary faithful to the source document?"
  - "Did this customer-support reply correctly identify the issue and
     suggest the right action?"
  - "Is this generated code stylistically consistent with the codebase?"
  - "Multi-criterion: accuracy AND format AND length AND tone"

For those, you need another LLM to read the prediction and decide.
That's what this module does.

CLI form
--------
The scorer spec is `llm:<judge-model-spec>:<criterion>`. The judge model
spec uses the same `<vendor>:<model>` form that adapters use elsewhere,
so it contains a colon — we parse with `rpartition` so only the final
colon is the criterion separator:

    depth-lens recommend \\
        --task custom:./summaries.jsonl:llm:openai:gpt-5-mini:faithful \\
        ...

That picks `openai:gpt-5-mini` as the judge and `faithful` as the
built-in criterion. To use a custom rubric instead, prefix it with
`rubric:` and pass an arbitrary string:

    --task custom:./tickets.jsonl:llm:openai:gpt-5-mini:rubric:the
        reply must identify the device model AND propose a concrete next step

Cost note
---------
Every prediction triggers one judge API call. With n_samples × compute
levels × depths × models, the judge cost is multiplied. Pick a cheap
judge model (gemini-3.1-flash-lite or gpt-5-mini at effort=low) for
exploratory sweeps. depth-lens does not yet fold judge token usage into
`recommend`'s $/k-pred line — track judge spend separately by exporting
JUDGE_DEBUG_LOG=1 to print each judge call.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

#: Built-in criteria. The value is the criterion-specific section
#: of the judge prompt; the surrounding template is constant.
BUILTIN_CRITERIA: dict[str, str] = {
    "correct": (
        "Decide whether the prediction is semantically equivalent to the "
        "target answer. Minor surface differences (whitespace, formatting, "
        "synonyms, restating the question) are OK; the answer must be "
        "logically the same."
    ),
    "faithful": (
        "Decide whether the prediction is faithful to the source: it must "
        "not invent facts that the target/reference does not support. "
        "Missing details are acceptable; fabricated details are not."
    ),
    "helpful": (
        "Decide whether the prediction is helpful to the user — does it "
        "answer the question concretely and actionably, rather than being "
        "evasive, generic, or off-topic?"
    ),
    "concise": (
        "Decide whether the prediction is appropriately concise: it conveys "
        "the answer without unnecessary preamble, repetition, or filler."
    ),
    "format": (
        "Decide whether the prediction follows the required output format "
        "implied by the target (e.g., JSON keys present, expected fields "
        "filled, no extra commentary outside the structure)."
    ),
    "polite": (
        "Decide whether the prediction uses an appropriate, polite tone for "
        "a professional communication. Bluntness, dismissiveness, or curt "
        "phrasing fails."
    ),
}


@dataclass(frozen=True)
class JudgeSpec:
    """Parsed scorer spec: `llm:<judge-model>:<criterion-or-rubric>`."""

    judge_model_spec: str       # e.g., "openai:gpt-5-mini"
    criterion_key: str | None   # one of BUILTIN_CRITERIA keys, OR None if rubric
    rubric: str | None          # free-form rubric text (when not a built-in)

    @property
    def criterion_text(self) -> str:
        if self.criterion_key is not None:
            return BUILTIN_CRITERIA[self.criterion_key]
        assert self.rubric is not None
        return self.rubric


def parse_judge_spec(spec: str) -> JudgeSpec:
    """Parse an 'llm:<judge-model>:<criterion>' scorer spec.

    Accepts:
        llm:openai:gpt-5-mini:faithful
        llm:anthropic:claude-haiku-4-5:correct
        llm:openai:gpt-5-mini:rubric:the reply must mention X and propose Y

    Note: the judge model spec itself contains a colon (`vendor:model`),
    so we partition from the RIGHT — the criterion is whatever follows
    the LAST colon, unless that LAST segment is preceded by `rubric:`
    in which case everything from `rubric:` onward is the free-form
    rubric text.
    """
    if not spec.startswith("llm:"):
        raise ValueError(f"judge spec must start with 'llm:', got {spec!r}")
    body = spec[len("llm:"):]

    # Free-form rubric case: `rubric:<text>` somewhere in the body.
    # Everything before `:rubric:` is the judge model spec.
    if ":rubric:" in body:
        judge_model_spec, _, rubric = body.partition(":rubric:")
        if not judge_model_spec or not rubric.strip():
            raise ValueError(
                f"rubric form is `llm:<judge-model>:rubric:<rubric-text>`, "
                f"got {spec!r}"
            )
        return JudgeSpec(
            judge_model_spec=judge_model_spec,
            criterion_key=None,
            rubric=rubric.strip(),
        )

    # Built-in criterion case: last colon separates the criterion key.
    judge_model_spec, sep, criterion_key = body.rpartition(":")
    if not sep:
        raise ValueError(
            f"judge spec must include a criterion: `llm:<judge-model>:<criterion>`, "
            f"got {spec!r}"
        )
    if criterion_key not in BUILTIN_CRITERIA:
        raise ValueError(
            f"unknown criterion {criterion_key!r}. Known: "
            f"{sorted(BUILTIN_CRITERIA)}. For free-form criteria, use "
            f"the `rubric:<text>` form."
        )
    return JudgeSpec(
        judge_model_spec=judge_model_spec,
        criterion_key=criterion_key,
        rubric=None,
    )


_JUDGE_PROMPT_TEMPLATE = """\
You are an evaluation judge. Decide whether the model's prediction
satisfies the stated criterion, given the original task prompt and the
target/reference answer.

[Original task prompt]
{prompt}

[Target / reference answer]
{target}

[Model's prediction to be judged]
{prediction}

[Criterion to apply]
{criterion}

Reason briefly, then on the FINAL LINE write exactly:
    Score: 1     (if the prediction meets the criterion)
    Score: 0     (if it does not)
"""


_SCORE_RE = re.compile(r"score\s*[:=]\s*([01])", re.IGNORECASE)


@dataclass
class LLMJudgeScorer:
    """Callable scorer that delegates judgment to a separate LLM.

    Holds the parsed spec and lazily constructs the judge adapter on first
    call. Aggregates per-call judge metadata (raw text, score) so a
    downstream report can audit individual rulings.
    """

    spec: JudgeSpec
    _adapter: object = field(default=None, init=False, repr=False)
    _log: list[dict] = field(default_factory=list, init=False, repr=False)

    @classmethod
    def from_string(cls, raw_spec: str) -> "LLMJudgeScorer":
        return cls(spec=parse_judge_spec(raw_spec))

    def _build_adapter(self):
        if self._adapter is None:
            from depth_lens.adapters import get_adapter
            self._adapter = get_adapter(self.spec.judge_model_spec)
        return self._adapter

    def score_one(self, prompt: str, target: str, prediction: str) -> float:
        """Score a single (prompt, target, prediction) triple. Returns 0.0 or 1.0."""
        from depth_lens.adapters.base import ComputeLevel

        adapter = self._build_adapter()
        judge_prompt = _JUDGE_PROMPT_TEMPLATE.format(
            prompt=prompt,
            target=target,
            prediction=prediction,
            criterion=self.spec.criterion_text,
        )

        # Pick a sensible default compute level — judges don't need much
        # reasoning. For OpenAI, "low" effort. For Anthropic, a small
        # thinking budget. The adapter's default grid's first entry works.
        compute = adapter.default_compute_grid()[0]
        preds = adapter.predict([judge_prompt], compute)
        raw = preds[0].text if preds else ""

        # Parse the final "Score: 0|1" line.
        matches = _SCORE_RE.findall(raw)
        if not matches:
            # If the judge didn't comply with the format, count as fail (0).
            score = 0.0
            parse_status = "no-score-line"
        else:
            score = float(int(matches[-1]))
            parse_status = "ok"

        self._log.append({
            "prompt": prompt,
            "target": target,
            "prediction": prediction,
            "judge_raw": raw,
            "score": score,
            "parse_status": parse_status,
        })

        if os.environ.get("JUDGE_DEBUG_LOG"):
            print(f"  [judge] criterion={self.spec.criterion_key or 'rubric'} "
                  f"score={score} parse={parse_status}", flush=True)

        return score

    @property
    def log(self) -> list[dict]:
        return list(self._log)
