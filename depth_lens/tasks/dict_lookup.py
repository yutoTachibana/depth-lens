"""
Dict-lookup task — the canonical "structured input field extraction" probe.

Given a key-value list and a query key, return the matching value. This is
the bounded-depth analog of common business workloads:

  - "User provided 5 fields in a form. Pull out the email."
  - "Extract the price from a JSON response containing 8 attributes."
  - "Retrieve the user_id from a session object with 12 keys."

The depth axis is the number of key-value pairs in the list, which controls
how many distractor pairs the model must skip past before locating the
target. Single-token prediction, no chain-of-thought required when the
model gets it right.

Prompt format (tokenizable by OpenMythos's word-level vocab):
    "c = 3 a = 1 b = 2 lookup b"
Target:
    "2"

Vocab: lowercase letters a-z (keys), digits 0-9 (values), '=' and 'lookup'.
Max depth = 26 (one per letter); typical sweeps run depth 2-12.
"""

from __future__ import annotations

import random
import re

from depth_lens.tasks.base import ProbeInstance, Task


class DictLookupTask(Task):
    """N-pair associative lookup — the structured-input extraction probe."""

    name = "dict-lookup"
    description = (
        "Given a list of `key = value` pairs and a query key, return the "
        "matching value. Depth = number of pairs in the list (more pairs = "
        "more distractors to skip)."
    )

    KEYS = "abcdefghijklmnopqrstuvwxyz"

    def vocab_seed(self) -> list[str]:
        return list(self.KEYS) + [str(i) for i in range(10)] + ["=", "lookup"]

    def generate(self, depth: int, n_samples: int, seed: int = 0) -> list[ProbeInstance]:
        if depth < 1 or depth > len(self.KEYS):
            raise ValueError(f"depth must be in [1, {len(self.KEYS)}], got {depth}")
        rng = random.Random(seed)
        out: list[ProbeInstance] = []
        for _ in range(n_samples):
            # Pick `depth` distinct keys and shuffle their presentation order.
            keys = list(self.KEYS[:depth])
            rng.shuffle(keys)
            # Each key gets an independent digit 0-9 (collisions OK — there's
            # only one (key, value) pair per key).
            values = [rng.randrange(10) for _ in range(depth)]
            query_idx = rng.randrange(depth)
            pairs = [f"{k} = {v}" for k, v in zip(keys, values, strict=False)]
            prompt = " ".join(pairs) + " lookup " + keys[query_idx]
            target = str(values[query_idx])
            out.append(
                ProbeInstance(
                    prompt=prompt,
                    target=target,
                    depth=depth,
                    metadata={
                        "keys": keys,
                        "values": values,
                        "query_key": keys[query_idx],
                    },
                )
            )
        return out

    def score(self, instance: ProbeInstance, prediction: str) -> float:
        """Extract the first standalone digit 0-9 in the prediction.

        Lenient enough for CoT outputs like 'so the value is 7' or
        'Final answer: 7'."""
        m = re.search(r"\b([0-9])\b", prediction)
        if m is None:
            return 0.0
        return float(m.group(1) == instance.target)
