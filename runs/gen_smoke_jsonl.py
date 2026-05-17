"""Generate a small natural-language K-hop arithmetic dataset for the API smoke."""

import json
import random
from pathlib import Path

M = 23
OPS = [
    ("add 1", lambda x: (x + 1) % M),
    ("add 5", lambda x: (x + 5) % M),
    ("multiply by 2", lambda x: (2 * x) % M),
    ("multiply by 3", lambda x: (3 * x) % M),
]

DEPTHS = [2, 4, 6, 8]
N_PER_DEPTH = 16
OUT = Path(__file__).parent / "smoke.jsonl"

rng = random.Random(42)
with OUT.open("w", encoding="utf-8") as f:
    for K in DEPTHS:
        for _ in range(N_PER_DEPTH):
            s = rng.randrange(M)
            ops = [rng.randrange(len(OPS)) for _ in range(K)]
            state = s
            for i in ops:
                state = OPS[i][1](state)
            op_descs = [OPS[i][0] for i in ops]
            prompt = (
                f"All arithmetic is modulo {M} (results are in [0, {M-1}]).\n"
                f"Starting from {s}, apply each operation in order:\n"
                + "\n".join(f"  {j+1}. {d}" for j, d in enumerate(op_descs))
                + "\nWhat is the final result? Show your work, then write `Final answer: <integer>`."
            )
            f.write(json.dumps({
                "prompt": prompt,
                "target": str(state),
                "depth": K,
                "metadata": {"s0": s, "ops": op_descs},
            }) + "\n")

print(f"wrote {OUT} with {len(DEPTHS) * N_PER_DEPTH} rows")
