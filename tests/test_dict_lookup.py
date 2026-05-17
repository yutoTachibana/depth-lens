"""Tests for the dict-lookup task."""

from __future__ import annotations

from depth_lens.tasks import get_task
from depth_lens.tasks.base import ProbeInstance
from depth_lens.tasks.dict_lookup import DictLookupTask


def test_get_task_dict_lookup():
    task = get_task("dict-lookup")
    assert isinstance(task, DictLookupTask)
    assert task.name == "dict-lookup"


def test_generate_shape():
    task = DictLookupTask()
    insts = task.generate(depth=5, n_samples=8, seed=42)
    assert len(insts) == 8
    for inst in insts:
        assert inst.depth == 5
        assert len(inst.metadata["keys"]) == 5
        assert len(set(inst.metadata["keys"])) == 5  # distinct keys
        assert inst.metadata["query_key"] in inst.metadata["keys"]
        # target is the digit paired with query_key
        idx = inst.metadata["keys"].index(inst.metadata["query_key"])
        assert inst.target == str(inst.metadata["values"][idx])


def test_prompt_format():
    """Prompt format must be tokenizable by the OpenMythos word-level vocab.

    Pairs separated by single spaces, equals sign as its own token, ending
    with 'lookup <key>'. This is the contract train_for_task relies on."""
    task = DictLookupTask()
    inst = task.generate(depth=3, n_samples=1, seed=0)[0]
    tokens = inst.prompt.split()
    # depth=3 → 3 pairs × 3 tokens each (k = v) + 'lookup' + key = 11 tokens
    assert len(tokens) == 11
    assert tokens[-2] == "lookup"
    assert tokens[-1] == inst.metadata["query_key"]
    # Equals signs at positions 1, 4, 7
    assert tokens[1] == "=" and tokens[4] == "=" and tokens[7] == "="


def test_score_extracts_first_digit():
    task = DictLookupTask()
    inst = ProbeInstance(
        prompt="a = 7 lookup a", target="7", depth=1, metadata={}
    )
    # Bare answer
    assert task.score(inst, "7") == 1.0
    # CoT trailing
    assert task.score(inst, "The value of a is 7.") == 1.0
    # Wrong digit
    assert task.score(inst, "Final answer: 3") == 0.0
    # No digit
    assert task.score(inst, "hello there") == 0.0


def test_score_ignores_multi_digit_distractors():
    """A multi-digit '23' in the response should NOT match target '2' or '3' —
    only standalone single digits 0-9 count, per the regex \\b[0-9]\\b."""
    task = DictLookupTask()
    inst = ProbeInstance(
        prompt="a = 5 lookup a", target="5", depth=1, metadata={}
    )
    assert task.score(inst, "the value is 23") == 0.0
    assert task.score(inst, "five = 5 (as in 5)") == 1.0


def test_vocab_seed():
    task = DictLookupTask()
    vocab = task.vocab_seed()
    # All 26 lowercase letters + 10 digits + = + lookup = 38 tokens
    assert len(vocab) == 38
    assert "=" in vocab
    assert "lookup" in vocab
    assert "a" in vocab and "z" in vocab
    assert "0" in vocab and "9" in vocab


def test_depth_bounds():
    """Depth must be in [1, 26]."""
    import pytest

    task = DictLookupTask()
    with pytest.raises(ValueError):
        task.generate(depth=0, n_samples=1)
    with pytest.raises(ValueError):
        task.generate(depth=27, n_samples=1)
    # Boundaries are valid:
    task.generate(depth=1, n_samples=1)
    task.generate(depth=26, n_samples=1)


def test_determinism():
    """Same seed → same instances."""
    task = DictLookupTask()
    a = task.generate(depth=4, n_samples=5, seed=123)
    b = task.generate(depth=4, n_samples=5, seed=123)
    for x, y in zip(a, b, strict=False):
        assert x.prompt == y.prompt
        assert x.target == y.target
