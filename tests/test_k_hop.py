"""Unit tests for the K-hop task."""

from __future__ import annotations

import pytest

from depth_lens.tasks import KHopTask, get_task


def test_get_task():
    t = get_task("k-hop")
    assert isinstance(t, KHopTask)


def test_generate_shape():
    t = KHopTask()
    insts = t.generate(depth=4, n_samples=10, seed=42)
    assert len(insts) == 10
    for x in insts:
        assert x.depth == 4
        assert x.prompt.endswith(" =")
        tokens = x.prompt.split()
        # 1 state + 4 ops + 1 "="
        assert len(tokens) == 6
        assert tokens[-1] == "="
        # Last segment of prompt before "=" should be op names
        for op_name in tokens[1:-1]:
            assert op_name in t.op_names


def test_targets_in_range():
    t = KHopTask(modulus=23)
    insts = t.generate(depth=6, n_samples=50, seed=0)
    for x in insts:
        v = int(x.target)
        assert 0 <= v < 23


def test_deterministic_with_seed():
    a = KHopTask().generate(depth=5, n_samples=20, seed=7)
    b = KHopTask().generate(depth=5, n_samples=20, seed=7)
    for x, y in zip(a, b, strict=True):
        assert x.prompt == y.prompt
        assert x.target == y.target


def test_score_exact():
    t = KHopTask()
    insts = t.generate(depth=3, n_samples=5, seed=0)
    for x in insts:
        assert t.score(x, x.target) == 1.0


def test_score_lenient_extracts_first_int():
    t = KHopTask()
    inst = t.generate(depth=2, n_samples=1, seed=0)[0]
    truth = inst.target
    assert t.score(inst, f"The answer is {truth}.") == 1.0
    assert t.score(inst, f"= {truth}\n") == 1.0
    # Wrong answer
    wrong = (int(truth) + 1) % t.M
    assert t.score(inst, f"{wrong}") == 0.0
    # No integer present
    assert t.score(inst, "nope") == 0.0


def test_invalid_depth():
    t = KHopTask()
    with pytest.raises(ValueError):
        t.generate(depth=0, n_samples=1)
