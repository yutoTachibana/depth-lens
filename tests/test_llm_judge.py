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


def test_judge_total_usage_accumulates(monkeypatch):
    """Each score_one call should add to total_usage() across all calls."""
    from depth_lens.adapters.base import Prediction

    captured: dict = {"usages": [
        {"input_tokens": 100, "output_tokens": 30},
        {"input_tokens": 110, "output_tokens": 25},
        {"input_tokens": 120, "output_tokens": 40},
    ]}

    fake_adapter = MagicMock()
    from depth_lens.adapters.base import ComputeLevel
    fake_adapter.default_compute_grid = MagicMock(return_value=[ComputeLevel(0, "default")])
    call_idx = {"n": 0}

    def fake_predict(prompts, compute):
        i = call_idx["n"]
        call_idx["n"] += 1
        return [Prediction(text="Score: 1", metadata={"usage": captured["usages"][i]})]

    fake_adapter.predict = fake_predict
    monkeypatch.setattr(
        "depth_lens.adapters.get_adapter", lambda *a, **k: fake_adapter
    )

    scorer = LLMJudgeScorer.from_string("llm:openai:gpt-5-mini:correct")
    for i in range(3):
        scorer.score_one(prompt=f"Q{i}", target="A", prediction="B")

    total = scorer.total_usage()
    # Sum across all 3 calls
    assert total["input"] == 100 + 110 + 120
    assert total["output"] == 30 + 25 + 40
    assert scorer.call_count() == 3


def test_custom_task_judge_summary(monkeypatch):
    """CustomTask.llm_judge_summary() exposes the spec, criterion, call count,
    total usage, and parse-failure count for the recommend CLI to surface."""
    from depth_lens.adapters.base import Prediction
    from depth_lens.tasks.custom import CustomTask

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        f.write(json.dumps({"prompt": "Q1", "target": "A1", "depth": 1}) + "\n")
        f.write(json.dumps({"prompt": "Q2", "target": "A2", "depth": 1}) + "\n")
        path = f.name

    # Two predictions: one scores 1 (ok parse), one has no Score line (fail closed)
    call_idx = {"n": 0}

    def fake_predict(prompts, compute):
        i = call_idx["n"]
        call_idx["n"] += 1
        text = "Score: 1" if i == 0 else "(no score line here)"
        return [Prediction(text=text, metadata={"usage": {"input_tokens": 50, "output_tokens": 10}})]

    fake_adapter = MagicMock()
    from depth_lens.adapters.base import ComputeLevel
    fake_adapter.default_compute_grid = MagicMock(return_value=[ComputeLevel(0, "default")])
    fake_adapter.predict = fake_predict
    monkeypatch.setattr(
        "depth_lens.adapters.get_adapter", lambda *a, **k: fake_adapter
    )

    task = CustomTask(path=path, scorer="llm:openai:gpt-5-mini:faithful")
    insts = task.generate(depth=1, n_samples=2, seed=0)
    task.score(insts[0], "pred1")
    task.score(insts[1], "pred2")

    summary = task.llm_judge_summary()
    assert summary is not None
    assert summary["judge_model_spec"] == "openai:gpt-5-mini"
    assert summary["criterion"] == "faithful"
    assert summary["call_count"] == 2
    assert summary["total_usage"]["input"] == 100  # 50 + 50
    assert summary["total_usage"]["output"] == 20  # 10 + 10
    assert summary["parse_failure_count"] == 1  # the second call had no Score line


def test_custom_task_judge_summary_none_for_non_judge_scorer():
    """Non-llm scorers should return None from llm_judge_summary()."""
    from depth_lens.tasks.custom import CustomTask

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        f.write(json.dumps({"prompt": "1+1=", "target": "2", "depth": 1}) + "\n")
        path = f.name

    task = CustomTask(path=path, scorer="first_int")
    assert task.llm_judge_summary() is None


def test_openai_adapter_free_form_skips_extraction(monkeypatch):
    """In free_form mode, the OpenAI adapter must NOT strip the response to a
    'Final answer:' line — the full reply is the answer."""
    import sys
    import types
    fake = types.ModuleType("openai")

    class FakeRateLimitError(Exception):
        pass

    class FakeAPIStatusError(Exception):
        pass

    class _Usage:
        prompt_tokens = 50
        completion_tokens = 80

    class _Message:
        content = "This is a multi-paragraph reply.\nNo 'Final answer:' marker here.\nThe model just answers directly."

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]
        usage = _Usage()

    class FakeClient:
        def __init__(self, *a, **kw):
            self.chat = MagicMock()
            self.chat.completions = MagicMock()
            self.chat.completions.create = MagicMock(return_value=_Response())

    fake.OpenAI = FakeClient
    fake.RateLimitError = FakeRateLimitError
    fake.APIStatusError = FakeAPIStatusError
    monkeypatch.setitem(sys.modules, "openai", fake)
    monkeypatch.setenv("OPENAI_API_KEY", "stub")

    from depth_lens.adapters.base import ComputeLevel
    from depth_lens.adapters.openai_adapter import OpenAIAdapter

    adapter = OpenAIAdapter(model="gpt-5-mini", task_name=None, free_form=True)
    preds = adapter.predict(["Write a paragraph."], ComputeLevel(1, "effort=low"))
    # Full text returned — no 'Final answer:' filter applied
    assert preds[0].text.startswith("This is a multi-paragraph reply.")
    assert "answers directly" in preds[0].text
