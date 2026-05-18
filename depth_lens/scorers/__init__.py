"""Pluggable scorers for custom tasks.

depth-lens's bundled scorers (in `depth_lens/tasks/custom.py`) cover
exact / first_int / last_int / yes_no / contains / regex — fine for
classification and short-answer tasks. The `llm_judge` scorer here
extends scoring to open-ended outputs (summaries, free-form Q&A,
code review, multi-criteria checks) by calling a separate "judge"
LLM to grade each prediction.
"""

from depth_lens.scorers.llm_judge import (
    BUILTIN_CRITERIA,
    LLMJudgeScorer,
    parse_judge_spec,
)

__all__ = ["LLMJudgeScorer", "BUILTIN_CRITERIA", "parse_judge_spec"]
