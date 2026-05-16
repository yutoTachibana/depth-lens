"""
Task abstraction.

A Task is a generator of probe instances at a given difficulty level (`depth`)
plus a scorer. depth is intentionally generic — for K-hop it's K; for parity
it's the number of bits; for graph reachability it's the path length.

Tasks return *prompts* (strings) and *targets* (strings) so they can be
consumed by any Adapter (logits-based for white-box models, generation-based
for API models). White-box adapters that need token IDs can re-tokenize the
prompt themselves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeInstance:
    """A single (prompt, target, metadata) triple ready to be scored."""

    prompt: str
    target: str
    depth: int  # The intrinsic difficulty depth of this instance.
    metadata: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


class Task(ABC):
    """A reasoning probe task with a controllable depth axis."""

    #: Stable identifier, used in CLIs and result files.
    name: str = "abstract-task"
    #: Human-readable single-line description.
    description: str = ""

    @abstractmethod
    def generate(self, depth: int, n_samples: int, seed: int = 0) -> list[ProbeInstance]:
        """Generate `n_samples` instances at the given depth."""
        ...

    def vocab_seed(self) -> list[str]:
        """
        Optional: return a list of canonical tokens that should appear in any
        vocabulary built for this task. Useful when target values exist that
        cannot be reached at low training depths (e.g., a counter task with
        modulus 17 whose target 8 only appears at K ≥ ~5). Adapters that build
        their own vocab on the fly (OpenMythos) consult this list so test-time
        targets at higher depths don't trip a KeyError.

        Default: empty (the adapter falls back to scanning generated samples).
        """
        return []

    def score(self, instance: ProbeInstance, prediction: str) -> float:
        """
        Default scorer: exact-match after light normalization.

        Subclasses can override (e.g., numeric tolerance, set equality).
        Returns 1.0 for correct, 0.0 for incorrect.
        """
        return float(_normalize(prediction) == _normalize(instance.target))

    def score_batch(
        self, instances: Sequence[ProbeInstance], predictions: Sequence[str]
    ) -> list[float]:
        assert len(instances) == len(predictions), "instances/predictions length mismatch"
        return [self.score(i, p) for i, p in zip(instances, predictions, strict=True)]


def _normalize(s: str) -> str:
    return s.strip().lower()
