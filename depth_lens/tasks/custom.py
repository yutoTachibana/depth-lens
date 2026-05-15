"""
CustomTask — load your own (prompt, target) pairs from a JSONL file.

This is the most practical task in the suite for real-world use: it lets you
probe a reasoning model's compute-scaling profile on *your* problem, not on
a synthetic benchmark.

JSONL schema (one object per line):
    {"prompt": "...", "target": "...", "depth": 4, "metadata": {...}}

    - `prompt`  (required): the input text shown to the model
    - `target`  (required): the canonical correct answer
    - `depth`   (optional): integer depth axis. If omitted, all rows are
                            treated as depth=1 (single-row probes still
                            work — you just lose the depth-extrapolation
                            axis and only sweep the compute axis).
    - `metadata`(optional): freeform dict passed through to ProbeInstance.

Scorer is pluggable:
    "exact"      — case-insensitive exact match after stripping
    "first_int"  — extract the first integer; compare to int(target)
    "last_int"   — extract the last integer; compare to int(target)
    "yes_no"     — first yes/no token; compare to lowercased target
    "contains"   — target appears anywhere in prediction (case-insensitive)
    "regex:<p>"  — pattern <p> matches prediction
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from depth_lens.tasks.base import ProbeInstance, Task


class CustomTask(Task):
    """Bring-your-own-data task loaded from a JSONL file."""

    description = "Probe a model on (prompt, target) pairs loaded from your own JSONL file."

    def __init__(
        self,
        path: str | Path,
        scorer: str = "exact",
        name: str | None = None,
    ):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"JSONL not found: {self.path}")
        self.name = name or f"custom:{self.path.stem}"
        self._scorer = scorer

        rows: list[dict] = []
        for i, line in enumerate(self.path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" not in obj or "target" not in obj:
                raise ValueError(
                    f"Line {i+1} of {self.path} missing 'prompt' or 'target'."
                )
            rows.append(obj)
        if not rows:
            raise ValueError(f"{self.path} contains no rows.")
        self._rows = rows

        # Index by depth for fast generate() lookups. Missing depth → 1.
        self._by_depth: dict[int, list[dict]] = {}
        for r in rows:
            d = int(r.get("depth", 1))
            self._by_depth.setdefault(d, []).append(r)

    def available_depths(self) -> list[int]:
        """Sorted list of depths present in the JSONL."""
        return sorted(self._by_depth.keys())

    def generate(self, depth: int, n_samples: int, seed: int = 0) -> list[ProbeInstance]:
        pool = self._by_depth.get(depth)
        if pool is None:
            raise KeyError(
                f"No rows at depth={depth} in {self.path}. "
                f"Available: {self.available_depths()}"
            )
        rng = random.Random(seed)
        # Sample with replacement so callers can request more than len(pool).
        # (Probes pin n_samples per cell; for small custom datasets this is
        # the natural behaviour.)
        picks = [rng.choice(pool) for _ in range(n_samples)]
        return [
            ProbeInstance(
                prompt=r["prompt"],
                target=str(r["target"]),
                depth=depth,
                metadata=r.get("metadata", {}),
            )
            for r in picks
        ]

    def score(self, instance: ProbeInstance, prediction: str) -> float:
        return _SCORERS_DISPATCH(self._scorer, instance.target, prediction)


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


def _SCORERS_DISPATCH(spec: str, target: str, pred: str) -> float:
    if spec.startswith("regex:"):
        pattern = spec[len("regex:"):]
        return float(bool(re.search(pattern, pred)))
    scorer = _SCORERS.get(spec)
    if scorer is None:
        raise ValueError(
            f"Unknown scorer {spec!r}. Known: {sorted(_SCORERS)} or 'regex:<pattern>'."
        )
    return scorer(target, pred)


def _exact(target: str, pred: str) -> float:
    return float(target.strip().lower() == pred.strip().lower())


def _first_int(target: str, pred: str) -> float:
    m = re.search(r"-?\d+", pred)
    if not m:
        return 0.0
    try:
        return float(int(m.group(0)) == int(target))
    except ValueError:
        return 0.0


def _last_int(target: str, pred: str) -> float:
    matches = re.findall(r"-?\d+", pred)
    if not matches:
        return 0.0
    try:
        return float(int(matches[-1]) == int(target))
    except ValueError:
        return 0.0


def _yes_no(target: str, pred: str) -> float:
    # Prefer "answer: yes/no" line, else first yes/no token.
    m = re.findall(
        r"(?:final\s+answer|answer)\s*[:=]\s*(yes|no|true|false)",
        pred,
        flags=re.IGNORECASE,
    )
    cand = (m[-1] if m else None)
    if cand is None:
        tokens = re.findall(r"\b(yes|no|true|false)\b", pred, flags=re.IGNORECASE)
        if tokens:
            cand = tokens[-1]
    if cand is None:
        return 0.0
    cand = "yes" if cand.lower() in ("yes", "true") else "no"
    tgt = target.strip().lower()
    tgt = "yes" if tgt in ("yes", "true") else "no"
    return float(cand == tgt)


def _contains(target: str, pred: str) -> float:
    return float(target.strip().lower() in pred.lower())


_SCORERS = {
    "exact": _exact,
    "first_int": _first_int,
    "last_int": _last_int,
    "yes_no": _yes_no,
    "contains": _contains,
}
