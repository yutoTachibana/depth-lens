"""
Two-counter state-tracking task.

The model tracks two integer counters C1 and C2 (mod M, default M=17) through
a sequence of K instructions, then is asked the final value of one counter.

Operators:
    inc1 : C1 = (C1 + 1) mod M
    inc2 : C2 = (C2 + 1) mod M
    swap : (C1, C2) = (C2, C1)
    add  : C1 = (C1 + C2) mod M

The task is intentionally distinct from K-hop:

  - State is **vector-valued** (two registers), not scalar.
  - `swap` mixes the registers, so the model can't just track a single sum.
  - The query at the end picks which register to read out — the model has to
    keep BOTH alive throughout the chain.

This is closer to register-machine emulation than to function composition,
and exposes a different failure mode than K-hop (where a single state suffices).

Prompt format: "inc1 inc2 swap add inc1 query 1"   (final integer = which register to read)
Target:        "<integer in [0, M)>"
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
        _Op("inc1", lambda c1, c2: ((c1 + 1) % M, c2)),
        _Op("inc2", lambda c1, c2: (c1, (c2 + 1) % M)),
        _Op("swap", lambda c1, c2: (c2, c1)),
        _Op("add", lambda c1, c2: ((c1 + c2) % M, c2)),
    ]


class StateTrackingTask(Task):
    """Two-counter modular state-tracking."""

    name = "state-tracking"
    description = (
        "Track two counters (mod M) through K instructions including increments, "
        "swap, and add. The query at the end picks which counter to read."
    )

    def __init__(self, modulus: int = 17):
        if modulus < 2:
            raise ValueError("modulus must be ≥ 2")
        self.M = modulus
        self.ops = _build_ops(modulus)

    @property
    def op_names(self) -> list[str]:
        return [o.name for o in self.ops]

    def vocab_seed(self) -> list[str]:
        # All counter values 0..M-1, all operator names, query and indices.
        return [str(i) for i in range(self.M)] + self.op_names + ["query", "1", "2"]

    def generate(self, depth: int, n_samples: int, seed: int = 0) -> list[ProbeInstance]:
        if depth < 1:
            raise ValueError("depth (K) must be ≥ 1")
        rng = random.Random(seed)
        out: list[ProbeInstance] = []
        for _ in range(n_samples):
            op_idxs = [rng.randrange(len(self.ops)) for _ in range(depth)]
            query = rng.choice([1, 2])

            c1, c2 = 0, 0
            for i in op_idxs:
                c1, c2 = self.ops[i].apply(c1, c2)
            answer = c1 if query == 1 else c2

            tokens = [self.ops[i].name for i in op_idxs] + ["query", str(query)]
            prompt = " ".join(tokens)
            out.append(
                ProbeInstance(
                    prompt=prompt,
                    target=str(answer),
                    depth=depth,
                    metadata={
                        "ops": [self.ops[i].name for i in op_idxs],
                        "query": query,
                        "M": self.M,
                        "final": (c1, c2),
                    },
                )
            )
        return out

    def score(self, instance: ProbeInstance, prediction: str) -> float:
        """
        Lenient numeric scoring: extract the *last* integer from prediction
        and compare mod M. The last-int rule is robust to CoT outputs that
        include intermediate values.
        """
        pred = _last_int(prediction)
        if pred is None:
            return 0.0
        return float((pred % self.M) == int(instance.target))


def _last_int(s: str) -> int | None:
    import re

    # Prefer an explicit `Final answer: N` / `Answer: N` line if present.
    pattern = re.compile(r"(?:final\s+answer|answer)\s*[:=]\s*(-?\d+)", re.IGNORECASE)
    matches = pattern.findall(s)
    if matches:
        return int(matches[-1])
    # Fall back to the last integer anywhere.
    ints = re.findall(r"-?\d+", s)
    if ints:
        return int(ints[-1])
    return None
