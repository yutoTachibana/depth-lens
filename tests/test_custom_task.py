"""Unit tests for CustomTask (bring-your-own JSONL)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from depth_lens.tasks import CustomTask, get_task


def _write(rows: list[dict]) -> Path:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.close()
    return Path(f.name)


def test_loads_jsonl():
    path = _write([
        {"prompt": "what is 1 + 1?", "target": "2", "depth": 1},
        {"prompt": "what is 2 + 3?", "target": "5", "depth": 1},
    ])
    t = CustomTask(path=path, scorer="first_int")
    assert t.available_depths() == [1]
    insts = t.generate(depth=1, n_samples=8, seed=0)
    assert len(insts) == 8
    # Sampling with replacement — but with 8 samples from 2 rows, both should appear.
    assert {i.target for i in insts} == {"2", "5"}


def test_missing_depth_defaults_to_1():
    path = _write([
        {"prompt": "p1", "target": "a"},
        {"prompt": "p2", "target": "b"},
    ])
    t = CustomTask(path=path)
    assert t.available_depths() == [1]


def test_groups_by_depth():
    path = _write([
        {"prompt": "p1", "target": "a", "depth": 2},
        {"prompt": "p2", "target": "b", "depth": 5},
        {"prompt": "p3", "target": "c", "depth": 5},
    ])
    t = CustomTask(path=path)
    assert t.available_depths() == [2, 5]
    d5 = t.generate(depth=5, n_samples=4, seed=0)
    assert all(i.depth == 5 for i in d5)
    assert all(i.target in {"b", "c"} for i in d5)


def test_unknown_depth_errors():
    path = _write([{"prompt": "p", "target": "t", "depth": 1}])
    t = CustomTask(path=path)
    with pytest.raises(KeyError, match="No rows at depth=3"):
        t.generate(depth=3, n_samples=1)


def test_scorer_first_int():
    path = _write([{"prompt": "p", "target": "42", "depth": 1}])
    t = CustomTask(path=path, scorer="first_int")
    inst = t.generate(1, 1, seed=0)[0]
    assert t.score(inst, "42") == 1.0
    assert t.score(inst, "the answer is 42, certainly") == 1.0
    assert t.score(inst, "the answer is 100, no wait, 42") == 0.0  # first int is 100


def test_scorer_last_int():
    path = _write([{"prompt": "p", "target": "42", "depth": 1}])
    t = CustomTask(path=path, scorer="last_int")
    inst = t.generate(1, 1, seed=0)[0]
    assert t.score(inst, "thinking 100... actually 42") == 1.0


def test_scorer_yes_no():
    path = _write([{"prompt": "p", "target": "yes", "depth": 1}])
    t = CustomTask(path=path, scorer="yes_no")
    inst = t.generate(1, 1, seed=0)[0]
    assert t.score(inst, "Final answer: yes") == 1.0
    assert t.score(inst, "True") == 1.0
    assert t.score(inst, "No") == 0.0


def test_scorer_contains():
    path = _write([{"prompt": "p", "target": "Paris", "depth": 1}])
    t = CustomTask(path=path, scorer="contains")
    inst = t.generate(1, 1, seed=0)[0]
    assert t.score(inst, "The capital of France is paris.") == 1.0
    assert t.score(inst, "Berlin") == 0.0


def test_scorer_regex():
    path = _write([{"prompt": "p", "target": "ignored", "depth": 1}])
    t = CustomTask(path=path, scorer="regex:\\d{4}-\\d{2}-\\d{2}")
    inst = t.generate(1, 1, seed=0)[0]
    assert t.score(inst, "Released 2026-05-15.") == 1.0
    assert t.score(inst, "no date here") == 0.0


def test_get_task_custom():
    path = _write([{"prompt": "p", "target": "1", "depth": 1}])
    t = get_task(f"custom:{path}:first_int")
    assert isinstance(t, CustomTask)
    assert t.score(t.generate(1, 1)[0], "1") == 1.0


def test_get_task_custom_default_scorer():
    path = _write([{"prompt": "p", "target": "hello", "depth": 1}])
    t = get_task(f"custom:{path}")
    assert isinstance(t, CustomTask)
    assert t.score(t.generate(1, 1)[0], "Hello") == 1.0  # exact (case-insensitive)
