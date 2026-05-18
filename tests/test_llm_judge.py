"""Tests for the LLM-as-judge scorer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from depth_lens.scorers.llm_judge import (
    BUILTIN_CRITERIA,
    LLMJudgeScorer,
    parse_judge_spec,
)


def test_parse_builtin_criterion():
    spec = parse_judge_spec("llm:openai:gpt-5-mini:faithful")
    assert spec.judge_model_spec == "openai:gpt-5-mini"
    assert spec.criterion_key == "faithful"
    assert spec.rubric is None


def test_parse_anthropic_judge_model():
    spec = parse_judge_spec("llm:anthropic:claude-haiku-4-5:correct")
    assert spec.judge_model_spec == "anthropic:claude-haiku-4-5"
    assert spec.criterion_key == "correct"


def test_parse_rubric_form():
    raw = "llm:openai:gpt-5-mini:rubric:the reply must mention X and propose Y"
    spec = parse_judge_spec(raw)
    assert spec.judge_model_spec == "openai:gpt-5-mini"
    assert spec.criterion_key is None
    assert spec.rubric == "the reply must mention X and propose Y"


def test_parse_rubric_with_internal_colons():
    """Free-form rubric text can contain colons; they shouldn't trip the parser."""
    raw = "llm:openai:gpt-5-mini:rubric:must include format: JSON with keys A:B"
    spec = parse_judge_spec(raw)
    assert spec.judge_model_spec == "openai:gpt-5-mini"
    assert spec.rubric == "must include format: JSON with keys A:B"


def test_parse_rejects_missing_prefix():
    with pytest.raises(ValueError, match="must start with 'llm:'"):
        parse_judge_spec("openai:gpt-5-mini:faithful")


def test_parse_rejects_unknown_criterion():
    with pytest.raises(ValueError, match="unknown criterion"):
        parse_judge_spec("llm:openai:gpt-5-mini:bogus_criterion")


def test_parse_rejects_missing_criterion():
    with pytest.raises(ValueError, match="must include a criterion"):
        parse_judge_spec("llm:openai-gpt5-mini")


def test_parse_rejects_empty_rubric():
    with pytest.raises(ValueError, match="rubric form"):
        parse_judge_spec("llm:openai:gpt-5-mini:rubric:")


def test_criterion_text_uses_builtin():
    spec = parse_judge_spec("llm:openai:gpt-5-mini:faithful")
    assert spec.criterion_text == BUILTIN_CRITERIA["faithful"]


def test_criterion_text_uses_rubric():
    raw = "llm:openai:gpt-5-mini:rubric:my custom rubric"
    spec = parse_judge_spec(raw)
    assert spec.criterion_text == "my custom rubric"


def _mock_adapter(judge_output: str):
    """Build a minimal adapter mock returning a single Prediction with the
    given .text. Replaces the real get_adapter return value."""
    from depth_lens.adapters.base import ComputeLevel, Prediction

    adapter = MagicMock()
    adapter.default_compute_grid = MagicMock(return_value=[ComputeLevel(0, "default")])
    adapter.predict = MagicMock(return_value=[Prediction(text=judge_output, metadata={})])
    return adapter


def test_score_one_returns_1_when_judge_says_score_1(monkeypatch):
    scorer = LLMJudgeScorer.from_string("llm:openai:gpt-5-mini:faithful")
    fake_adapter = _mock_adapter("Reasoning: the summary preserves all facts.\nScore: 1")
    monkeypatch.setattr(
        "depth_lens.adapters.get_adapter", lambda *a, **k: fake_adapter
    )
    score = scorer.score_one(prompt="Summarize: text", target="reference", prediction="output")
    assert score == 1.0
    assert len(scorer.log) == 1
    entry = scorer.log[0]
    assert entry["score"] == 1.0
    assert entry["parse_status"] == "ok"


def test_score_one_returns_0_when_judge_says_score_0(monkeypatch):
    scorer = LLMJudgeScorer.from_string("llm:openai:gpt-5-mini:correct")
    fake_adapter = _mock_adapter("Reasoning: invented facts.\nScore: 0")
    monkeypatch.setattr(
        "depth_lens.adapters.get_adapter", lambda *a, **k: fake_adapter
    )
    assert scorer.score_one(prompt="Q", target="A", prediction="B") == 0.0


def test_score_one_returns_0_on_malformed_judge_output(monkeypatch):
    """If the judge ignores the format instruction, fail closed (0)."""
    scorer = LLMJudgeScorer.from_string("llm:openai:gpt-5-mini:correct")
    fake_adapter = _mock_adapter("This summary is fine, but I will not answer in the format.")
    monkeypatch.setattr(
        "depth_lens.adapters.get_adapter", lambda *a, **k: fake_adapter
    )
    assert scorer.score_one(prompt="Q", target="A", prediction="B") == 0.0
    assert scorer.log[-1]["parse_status"] == "no-score-line"


def test_score_uses_last_score_line(monkeypatch):
    """Judge may reason about partial scores in CoT, e.g. 'tempted to score 0
    but actually 1'. We must take the LAST Score: line, like Anthropic does
    for thinking blocks."""
    scorer = LLMJudgeScorer.from_string("llm:openai:gpt-5-mini:correct")
    fake_adapter = _mock_adapter(
        "Initial reaction: Score: 0. On second thought,\nScore: 1"
    )
    monkeypatch.setattr(
        "depth_lens.adapters.get_adapter", lambda *a, **k: fake_adapter
    )
    assert scorer.score_one(prompt="Q", target="A", prediction="B") == 1.0


def test_custom_task_uses_llm_judge_scorer(monkeypatch):
    """End-to-end: CustomTask with an `llm:` scorer routes through the
    judge and the result reaches .score()."""
    from depth_lens.tasks.custom import CustomTask

    # Set up a tiny JSONL bench file.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        f.write(json.dumps({"prompt": "Q?", "target": "A.", "depth": 1}) + "\n")
        path = f.name

    fake_adapter = _mock_adapter("Looks correct.\nScore: 1")
    monkeypatch.setattr(
        "depth_lens.adapters.get_adapter", lambda *a, **k: fake_adapter
    )

    task = CustomTask(path=path, scorer="llm:openai:gpt-5-mini:correct")
    insts = task.generate(depth=1, n_samples=3, seed=0)
    score = task.score(insts[0], "my prediction")
    assert score == 1.0
    # judge log should be populated
    log = task.llm_judge_log
    assert log is not None and len(log) == 1
    assert log[0]["prediction"] == "my prediction"


def test_get_task_parses_llm_scorer_from_custom_spec(monkeypatch):
    """`custom:./path.jsonl:llm:openai:gpt-5-mini:correct` should parse the
    `llm:...` tail as the scorer (not get confused by inner colons)."""
    from depth_lens.tasks import get_task

    # Make a temp JSONL so CustomTask constructor doesn't error.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        f.write(json.dumps({"prompt": "Q", "target": "A", "depth": 1}) + "\n")
        path = f.name

    spec = f"custom:{path}:llm:openai:gpt-5-mini:correct"
    task = get_task(spec)
    assert task._scorer == "llm:openai:gpt-5-mini:correct"
    assert task._llm_judge is not None
    assert task._llm_judge.spec.criterion_key == "correct"


def test_get_task_parses_path_with_simple_scorer_still_works():
    """Regression: existing `custom:./p.jsonl:first_int` form keeps working."""
    from depth_lens.tasks import get_task

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        f.write(json.dumps({"prompt": "1+1=", "target": "2", "depth": 1}) + "\n")
        path = f.name

    task = get_task(f"custom:{path}:first_int")
    assert task._scorer == "first_int"
    assert task._llm_judge is None


def test_get_task_parses_path_with_regex_scorer():
    """Regression: regex:<pattern> tail still parses."""
    from depth_lens.tasks import get_task

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        f.write(json.dumps({"prompt": "x", "target": "y", "depth": 1}) + "\n")
        path = f.name

    task = get_task(f"custom:{path}:regex:hello\\s+world")
    assert task._scorer == "regex:hello\\s+world"


def test_get_task_default_scorer_is_exact():
    """No scorer suffix → exact."""
    from depth_lens.tasks import get_task

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        f.write(json.dumps({"prompt": "x", "target": "y", "depth": 1}) + "\n")
        path = f.name

    task = get_task(f"custom:{path}")
    assert task._scorer == "exact"
