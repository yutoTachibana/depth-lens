"""Unit tests for the mini-CSP (2-SAT) task."""

from depth_lens.tasks import MiniCSPTask, get_task
from depth_lens.tasks.mini_csp import _is_sat


def test_get_task():
    assert isinstance(get_task("mini-csp"), MiniCSPTask)


def test_is_sat_basic():
    # (a OR b) -- SAT (e.g., a=T, b=T)
    assert _is_sat([((0, False), (1, False))], n=2) is True
    # (a OR b) AND (NOT a OR b) AND (a OR NOT b) AND (NOT a OR NOT b) -- UNSAT
    clauses = [
        ((0, False), (1, False)),
        ((0, True), (1, False)),
        ((0, False), (1, True)),
        ((0, True), (1, True)),
    ]
    assert _is_sat(clauses, n=2) is False


def test_generate_balance_sat_unsat():
    t = MiniCSPTask()
    insts = t.generate(depth=4, n_samples=20, seed=0)
    yeses = sum(1 for x in insts if x.target == "yes")
    nos = sum(1 for x in insts if x.target == "no")
    assert yeses == nos == 10


def test_generate_targets_match_brute_force():
    """Whatever the task says SAT/UNSAT should be confirmable by re-running _is_sat."""
    t = MiniCSPTask()
    for d in range(2, 7):
        insts = t.generate(depth=d, n_samples=8, seed=d * 7)
        for inst in insts:
            sat_truth = _is_sat(inst.metadata["clauses"], n=d)
            expected = "yes" if sat_truth else "no"
            assert inst.target == expected, (
                f"target mismatch for depth={d}, prompt={inst.prompt}"
            )


def test_prompt_is_whitespace_tokenizable():
    """All tokens in the prompt should appear in the vocab seed."""
    t = MiniCSPTask()
    seed = set(t.vocab_seed())
    for inst in t.generate(depth=5, n_samples=4, seed=11):
        for tok in inst.prompt.split():
            assert tok in seed, f"unexpected token {tok!r} in prompt {inst.prompt!r}"


def test_score_lenient():
    t = MiniCSPTask()
    inst = t.generate(2, 1, seed=0)[0]
    truth = inst.target
    # Plain answer
    assert t.score(inst, truth) == 1.0
    # Verbose CoT
    assert t.score(inst, f"trying assignment a=true b=false ... Final answer: {truth}") == 1.0
    # Synonyms
    syn = "satisfiable" if truth == "yes" else "unsatisfiable"
    assert t.score(inst, f"answer: {syn}") == 1.0
    # Wrong
    wrong = "no" if truth == "yes" else "yes"
    assert t.score(inst, f"Final answer: {wrong}") == 0.0
    # Empty
    assert t.score(inst, "") == 0.0


def test_invalid_depth():
    t = MiniCSPTask()
    import pytest
    with pytest.raises(ValueError):
        t.generate(depth=1, n_samples=1)
    with pytest.raises(ValueError):
        t.generate(depth=20, n_samples=1)
