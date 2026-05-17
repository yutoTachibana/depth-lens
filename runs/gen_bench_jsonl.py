"""
Generate a *graded-difficulty* K-hop benchmark.

The `depth` field is no longer literal K — it's a difficulty *tier*. Each
tier combines length, modulus, and operator complexity so the axis represents
how hard the row is, not just how long.

Tier 1 (easy)      : mod 13,  K=3,  ops = {+1, +2, +5}                  — Haiku ceiling
Tier 2 (moderate)  : mod 23,  K=6,  ops = {+1, +5, *2, *3}              — Sonnet starts here
Tier 3 (hard)      : mod 47,  K=10, ops = {+3, +11, -7, *2, *5}         — Opus starts here
Tier 4 (very hard) : mod 97,  K=14, ops = {+7, +41, -11, *3, *5, *11}   — frontier
"""

import json
import random
from pathlib import Path

TIERS = [
    {
        "tier": 1,
        "label": "easy",
        "M": 13,
        "K": 3,
        "ops": [("add 1", lambda x, M=13: (x + 1) % M),
                ("add 2", lambda x, M=13: (x + 2) % M),
                ("add 5", lambda x, M=13: (x + 5) % M)],
    },
    {
        "tier": 2,
        "label": "moderate",
        "M": 23,
        "K": 6,
        "ops": [("add 1", lambda x, M=23: (x + 1) % M),
                ("add 5", lambda x, M=23: (x + 5) % M),
                ("multiply by 2", lambda x, M=23: (2 * x) % M),
                ("multiply by 3", lambda x, M=23: (3 * x) % M)],
    },
    {
        "tier": 3,
        "label": "hard",
        "M": 47,
        "K": 10,
        "ops": [("add 3", lambda x, M=47: (x + 3) % M),
                ("add 11", lambda x, M=47: (x + 11) % M),
                ("subtract 7", lambda x, M=47: (x - 7) % M),
                ("multiply by 2", lambda x, M=47: (2 * x) % M),
                ("multiply by 5", lambda x, M=47: (5 * x) % M)],
    },
    {
        "tier": 4,
        "label": "very_hard",
        "M": 97,
        "K": 14,
        "ops": [("add 7", lambda x, M=97: (x + 7) % M),
                ("add 41", lambda x, M=97: (x + 41) % M),
                ("subtract 11", lambda x, M=97: (x - 11) % M),
                ("multiply by 3", lambda x, M=97: (3 * x) % M),
                ("multiply by 5", lambda x, M=97: (5 * x) % M),
                ("multiply by 11", lambda x, M=97: (11 * x) % M)],
    },
]

N_PER_TIER = 16
OUT = Path(__file__).parent / "bench.jsonl"

rng = random.Random(2026)
total = 0
with OUT.open("w", encoding="utf-8") as f:
    for spec in TIERS:
        M, K = spec["M"], spec["K"]
        ops = spec["ops"]
        for _ in range(N_PER_TIER):
            s0 = rng.randrange(M)
            picks = [rng.randrange(len(ops)) for _ in range(K)]
            state = s0
            for i in picks:
                state = ops[i][1](state)
            op_descs = [ops[i][0] for i in picks]
            prompt = (
                f"All arithmetic is modulo {M} (results are in [0, {M-1}]).\n"
                f"Starting from {s0}, apply each operation in order:\n"
                + "\n".join(f"  {j+1}. {d}" for j, d in enumerate(op_descs))
                + f"\nWhat is the final result? Show your work, then write `Final answer: <integer>`."
            )
            f.write(json.dumps({
                "prompt": prompt,
                "target": str(state),
                "depth": spec["tier"],
                "metadata": {
                    "label": spec["label"],
                    "M": M, "K": K, "s0": s0, "ops": op_descs,
                },
            }) + "\n")
            total += 1

print(f"wrote {OUT} with {total} rows across {len(TIERS)} difficulty tiers")
for spec in TIERS:
    print(f"  tier {spec['tier']} ({spec['label']:9s}): M={spec['M']:3d}, K={spec['K']:2d}, ops={len(spec['ops'])}")
