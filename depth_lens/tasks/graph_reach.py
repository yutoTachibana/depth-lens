"""
Graph reachability task.

Given a small directed acyclic graph encoded as a list of edges and two
nodes (start, goal), predict whether goal is reachable from start.

The instance is constructed so that *the unique reachability path has length
exactly `depth`*. This makes "depth" a faithful axis for compositional depth:
solving a depth-K instance requires K chained inferences.

To eliminate trivial memorization, we mix positive (reachable) and negative
(unreachable) instances 1:1, with the negative instance built by perturbing
one edge in the positive path so reachability genuinely fails — meaning a
shallow heuristic ("does the goal appear at all?") gets 50% at best.

Prompt format (whitespace-separable so adapters with naive tokenizers can
build a small vocab):
    "edges : a -> b ; b -> c ; c -> d ; reach a -> d ?"
Target:
    "yes" or "no"
"""

from __future__ import annotations

import random
import string

from depth_lens.tasks.base import ProbeInstance, Task


class GraphReachabilityTask(Task):
    """Multi-hop yes/no reachability on a small DAG."""

    name = "graph-reach"
    description = (
        "Determine whether `goal` is reachable from `start` in a small DAG. "
        "Instances are constructed so the unique forward path has length "
        "exactly `depth`, with 50/50 yes/no balance."
    )

    NODES = list(string.ascii_lowercase)  # 26 nodes available

    def vocab_seed(self) -> list[str]:
        return list(self.NODES) + ["edges", ":", ";", "reach", "->", "?", "yes", "no"]

    def generate(self, depth: int, n_samples: int, seed: int = 0) -> list[ProbeInstance]:
        if depth < 1:
            raise ValueError("depth (path length) must be ≥ 1")
        if depth + 4 > len(self.NODES):
            raise ValueError(f"depth too large ({depth}); max ~{len(self.NODES)-4}")

        rng = random.Random(seed)
        out: list[ProbeInstance] = []
        for i in range(n_samples):
            is_positive = (i % 2 == 0)
            inst = self._gen_one(rng, depth, is_positive)
            out.append(inst)
        return out

    def _gen_one(self, rng: random.Random, depth: int, is_positive: bool) -> ProbeInstance:
        # Sample depth+1 distinct nodes for the chain plus 3 distractor nodes.
        n_chain = depth + 1
        n_extra = 3
        pool = list(self.NODES)
        rng.shuffle(pool)
        chain = pool[:n_chain]
        extras = pool[n_chain : n_chain + n_extra]

        # Build the chain edges
        edges: list[tuple[str, str]] = [(chain[i], chain[i + 1]) for i in range(depth)]

        # Add distractor edges among extras + non-path connections, never
        # creating a shorter or alternative path to chain[-1] from chain[0].
        # Simplest: distractor edges only among `extras` and from extras into
        # the middle of the chain (not back into chain[0] or out from chain[-1]).
        n_distractors = max(2, depth)
        attempts = 0
        while len(edges) < depth + n_distractors and attempts < 50:
            attempts += 1
            src = rng.choice(extras + chain[1:-1])
            dst = rng.choice(extras + chain[1:])
            if src == dst:
                continue
            if (src, dst) in edges:
                continue
            # Avoid creating an alternative path into chain[0] (since we want
            # the path from chain[0] to chain[-1] to be unique).
            if dst == chain[0]:
                continue
            edges.append((src, dst))

        start, goal = chain[0], chain[-1]

        if not is_positive:
            # Remove the last chain edge so the goal becomes unreachable from
            # the chain side, then strip any other incoming edge to goal
            # (distractors may have added some). Finally add a same-size
            # extras-only distractor so the no-instance has the same edge count
            # as the yes-instance.
            edges.remove((chain[depth - 1], chain[depth]))
            edges = [(a, b) for (a, b) in edges if b != chain[depth]]
            for _ in range(20):
                src = rng.choice(extras)
                dst = rng.choice([n for n in extras if n != src])
                if (src, dst) not in edges:
                    edges.append((src, dst))
                    break

        rng.shuffle(edges)

        edge_str = " ; ".join(f"{a} -> {b}" for a, b in edges)
        prompt = f"edges : {edge_str} ; reach {start} -> {goal} ?"
        target = "yes" if is_positive else "no"
        return ProbeInstance(
            prompt=prompt,
            target=target,
            depth=depth,
            metadata={"edges": edges, "start": start, "goal": goal, "is_positive": is_positive},
        )

    def score(self, instance: ProbeInstance, prediction: str) -> float:
        """
        Pull a yes/no answer out of the prediction. Looks for the first
        explicit yes/no token, treating 'true'/'false' as synonyms.
        """
        pred = _first_yesno(prediction)
        if pred is None:
            return 0.0
        return float(pred == instance.target)


def _first_yesno(s: str) -> str | None:
    import re

    # Prefer explicit "answer: yes/no" lines if present (last occurrence).
    pattern = re.compile(
        r"(?:final\s+answer|answer)\s*[:=]\s*(yes|no|true|false)",
        re.IGNORECASE,
    )
    m = pattern.findall(s)
    if m:
        return _yesno_map(m[-1])

    # Fall back to last standalone yes/no token.
    tokens = re.findall(r"\b(yes|no|true|false)\b", s, re.IGNORECASE)
    if tokens:
        return _yesno_map(tokens[-1])
    return None


def _yesno_map(t: str) -> str:
    t = t.lower()
    return "yes" if t in ("yes", "true") else "no"
