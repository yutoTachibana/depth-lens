"""
Binary parity task.

Given a bit string b_1 b_2 ... b_n, predict its parity (XOR of all bits).
Parity is the canonical example of a problem where reasoning depth must scale
with input length — each new bit requires one additional composition step.

A 1-layer transformer cannot solve parity exactly for arbitrary n (well-known
expressivity bound). Looped / recurrent depth lets the *same* parameters
handle n=4 and n=16 by running more iterations.

Prompt format: "1 0 1 1 0 1 parity"
Target:        "0" or "1"
"""

from __future__ import annotations

import random

from depth_lens.tasks.base import ProbeInstance, Task


class ParityTask(Task):
    """Binary parity over n bits — classic depth-extrapolation probe."""

    name = "parity"
    description = "Compute the XOR (parity) of a length-n bit string."

    def vocab_seed(self) -> list[str]:
        return ["0", "1", "parity"]

    def generate(self, depth: int, n_samples: int, seed: int = 0) -> list[ProbeInstance]:
        if depth < 1:
            raise ValueError("depth (number of bits) must be ≥ 1")
        rng = random.Random(seed)
        out: list[ProbeInstance] = []
        for _ in range(n_samples):
            bits = [rng.randrange(2) for _ in range(depth)]
            prompt = " ".join(str(b) for b in bits) + " parity"
            target = str(sum(bits) % 2)
            out.append(
                ProbeInstance(
                    prompt=prompt,
                    target=target,
                    depth=depth,
                    metadata={"bits": bits},
                )
            )
        return out

    def score(self, instance: ProbeInstance, prediction: str) -> float:
        """
        Extract the first 0/1 in the prediction. Lenient enough to handle
        CoT outputs like '... so the parity is 1.' or 'Final answer: 1'.
        """
        pred = _first_bit(prediction)
        if pred is None:
            return 0.0
        return float(str(pred) == instance.target)


def _first_bit(s: str) -> int | None:
    """Return the *last* 0/1 in s (last is more robust on CoT outputs)."""
    import re

    matches = re.findall(r"\b[01]\b", s)
    if not matches:
        return None
    return int(matches[-1])
