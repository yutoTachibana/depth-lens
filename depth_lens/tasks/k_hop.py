"""
K-hop modular composition task.

Apply K permutation operators on Z/MZ sequentially to a starting state s_0,
predict the resulting state. Used by the looped-transformer literature
(e.g., Saunshi et al. 2025) as a standard latent-CoT probe.

Operators are chosen so the generated group is non-abelian:
    add1, add5, mul2, mul3   (all in Z/23Z, all bijections)

Prompt format: "3 add5 mul2 add1 ="
Target:        "<integer answer in [0, M)>"
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from depth_lens.tasks.base import ProbeInstance, Task


@dataclass(frozen=True)
class _Op:
    name: str
    apply: callable  # type: ignore[type-arg]


def _build_ops(M: int) -> list[_Op]:
    return [
        _Op("add1", lambda x: (x + 1) % M),
        _Op("add5", lambda x: (x + 5) % M),
        _Op("mul2", lambda x: (2 * x) % M),
        _Op("mul3", lambda x: (3 * x) % M),
    ]


class KHopTask(Task):
    """K-hop modular composition over Z/MZ."""

    name = "k-hop"
    description = "K-hop modular composition: apply K permutation operators sequentially in Z/MZ."

    def __init__(self, modulus: int = 23):
        if modulus < 2:
            raise ValueError("modulus must be ≥ 2")
        self.M = modulus
        self.ops = _build_ops(modulus)

    @property
    def op_names(self) -> list[str]:
        return [o.name for o in self.ops]

    def vocab_seed(self) -> list[str]:
        # All state values 0..M-1, all operator names, and the "=" delimiter.
        return [str(i) for i in range(self.M)] + self.op_names + ["="]

    def generate(self, depth: int, n_samples: int, seed: int = 0) -> list[ProbeInstance]:
        if depth < 1:
            raise ValueError("depth (K) must be ≥ 1")
        rng = random.Random(seed)
        out: list[ProbeInstance] = []
        for _ in range(n_samples):
            s = rng.randrange(self.M)
            op_idxs = [rng.randrange(len(self.ops)) for _ in range(depth)]
            tokens = [str(s)] + [self.ops[i].name for i in op_idxs] + ["="]
            prompt = " ".join(tokens)
            state = s
            for i in op_idxs:
                state = self.ops[i].apply(state)
            out.append(
                ProbeInstance(
                    prompt=prompt,
                    target=str(state),
                    depth=depth,
                    metadata={"s0": s, "ops": [self.ops[i].name for i in op_idxs], "M": self.M},
                )
            )
        return out

    def score(self, instance: ProbeInstance, prediction: str) -> float:
        """
        Lenient numeric scoring: extract the first integer token from prediction
        and compare modulo M. This is important for API models that produce
        verbose outputs like "The answer is 7." or "= 7\n".
        """
        pred = _first_int(prediction)
        if pred is None:
            return 0.0
        return float((pred % self.M) == int(instance.target))


def _first_int(s: str) -> int | None:
    """Return the first integer in s, or None. Handles signed integers."""
    import re

    m = re.search(r"-?\d+", s)
    return int(m.group(0)) if m else None
