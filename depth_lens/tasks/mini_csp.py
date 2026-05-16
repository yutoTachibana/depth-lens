"""
Mini-CSP — small 2-SAT satisfiability.

Each instance is a small 2-CNF formula over n variables. The model decides
whether the formula is satisfiable. This adds a *search* / *constraint
propagation* shape to the task suite — distinct from the forward composition
of k-hop / parity / state-tracking and the single-pass BFS of graph-reach.

The depth axis is the number of Boolean variables (n). Number of clauses
scales as ⌈1.5·n⌉ — enough to make the formula non-trivial without saturating.

Generation is balanced 50/50 between SAT and UNSAT instances, by rejection
sampling against an O(2ⁿ) brute-force satisfiability check (cheap for n ≤ 10).

Prompt format (whitespace-tokenizable for both API LMs and the OpenMythos
naïve tokenizer):

    vars : a b c d ;
    clauses : ( a OR NOT b ) AND ( b OR c ) AND ( NOT c OR d ) ;
    satisfiable ?

Target: `yes` (∃ assignment satisfying every clause) or `no` (UNSAT).
"""

from __future__ import annotations

import random
import string
from itertools import product

from depth_lens.tasks.base import ProbeInstance, Task

_MAX_VARS = 10  # safety cap — brute force is 2^n


def _is_sat(clauses: list[tuple[tuple[int, bool], tuple[int, bool]]], n: int) -> bool:
    """
    Brute-force satisfiability.

    `clauses` is a list of 2-clauses; each clause is two (var_idx, is_negated) pairs.
    A clause `((i, False), (j, True))` means `(x_i OR NOT x_j)`.
    """
    for assignment in product([False, True], repeat=n):
        ok = True
        for (i, ni), (j, nj) in clauses:
            li = (not assignment[i]) if ni else assignment[i]
            lj = (not assignment[j]) if nj else assignment[j]
            if not (li or lj):
                ok = False
                break
        if ok:
            return True
    return False


def _format_literal(var_name: str, negated: bool) -> str:
    return f"NOT {var_name}" if negated else var_name


def _format_clause(c: tuple[tuple[int, bool], tuple[int, bool]], var_names: list[str]) -> str:
    (i, ni), (j, nj) = c
    return f"( {_format_literal(var_names[i], ni)} OR {_format_literal(var_names[j], nj)} )"


class MiniCSPTask(Task):
    """Tiny 2-SAT satisfiability with a depth=#variables axis."""

    name = "mini-csp"
    description = (
        "Decide whether a small 2-SAT formula is satisfiable. depth = number of "
        "Boolean variables; clauses scale as ⌈1.5·depth⌉. SAT/UNSAT instances "
        "are balanced 50/50 via brute-force rejection sampling."
    )

    def __init__(self, max_attempts_per_instance: int = 200):
        self.max_attempts_per_instance = max_attempts_per_instance

    def vocab_seed(self) -> list[str]:
        return [
            *list(string.ascii_lowercase[:_MAX_VARS]),  # variable names a..j
            "vars", ":", ";", "clauses",
            "(", ")", "OR", "AND", "NOT",
            "satisfiable", "?", "yes", "no",
        ]

    def generate(self, depth: int, n_samples: int, seed: int = 0) -> list[ProbeInstance]:
        if not 2 <= depth <= _MAX_VARS:
            raise ValueError(f"depth (n_vars) must be in [2, {_MAX_VARS}], got {depth}")
        rng = random.Random(seed)
        # ~2× variables — enough to make UNSAT instances likely on rejection
        # sampling at depth=2 (where the SAT/UNSAT space is tiny).
        n_clauses = max(3, 2 * depth)
        out: list[ProbeInstance] = []
        for i in range(n_samples):
            want_sat = (i % 2 == 0)
            inst = self._gen_one(rng, depth, n_clauses, want_sat)
            out.append(inst)
        return out

    def _gen_one(
        self,
        rng: random.Random,
        n: int,
        n_clauses: int,
        want_sat: bool,
    ) -> ProbeInstance:
        var_names = list(string.ascii_lowercase[:n])
        for _ in range(self.max_attempts_per_instance):
            clauses: list[tuple[tuple[int, bool], tuple[int, bool]]] = []
            for _ in range(n_clauses):
                i = rng.randrange(n)
                j = rng.randrange(n)
                while j == i:  # avoid trivial tautologies / contradictions
                    j = rng.randrange(n)
                ni = bool(rng.getrandbits(1))
                nj = bool(rng.getrandbits(1))
                clauses.append(((i, ni), (j, nj)))
            sat = _is_sat(clauses, n)
            if sat == want_sat:
                clause_strs = [_format_clause(c, var_names) for c in clauses]
                vars_str = " ".join(var_names)
                clauses_str = " AND ".join(clause_strs)
                prompt = (
                    f"vars : {vars_str} ; "
                    f"clauses : {clauses_str} ; "
                    f"satisfiable ?"
                )
                target = "yes" if want_sat else "no"
                return ProbeInstance(
                    prompt=prompt,
                    target=target,
                    depth=n,
                    metadata={
                        "n_vars": n,
                        "n_clauses": n_clauses,
                        "want_sat": want_sat,
                        "clauses": clauses,
                    },
                )
        raise RuntimeError(
            f"Could not synthesize a depth={n}, want_sat={want_sat} 2-SAT "
            f"instance in {self.max_attempts_per_instance} attempts."
        )

    def score(self, instance: ProbeInstance, prediction: str) -> float:
        """
        Lenient yes/no scoring: extract the last yes/no/true/false token in the
        prediction (preferring an explicit `Final answer: …` line if present).
        """
        pred = _first_yesno(prediction)
        if pred is None:
            return 0.0
        return float(pred == instance.target)


def _first_yesno(s: str) -> str | None:
    import re

    pattern = re.compile(
        r"(?:final\s+answer|answer)\s*[:=]\s*(yes|no|true|false|sat|unsat|satisfiable|unsatisfiable)",
        re.IGNORECASE,
    )
    matches = pattern.findall(s)
    if matches:
        return _yesno_map(matches[-1])
    tokens = re.findall(
        r"\b(yes|no|true|false|sat|unsat|satisfiable|unsatisfiable)\b",
        s, re.IGNORECASE,
    )
    if tokens:
        return _yesno_map(tokens[-1])
    return None


def _yesno_map(t: str) -> str:
    t = t.lower()
    if t in ("yes", "true", "sat", "satisfiable"):
        return "yes"
    return "no"
