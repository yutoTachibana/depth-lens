"""
Adapter abstraction.

A ModelAdapter exposes a single primitive: given a list of prompts and a
*compute level* (a model-specific knob — n_loops for OpenMythos,
thinking_budget for Claude, etc.), return predictions.

The Adapter is responsible for translating the abstract compute level into
whatever the underlying model accepts. depth-lens treats compute levels as
opaque comparable values within a single adapter, but adapters may report a
canonical name for each level (used for plot legends).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeLevel:
    """An opaque compute setting interpretable by a specific adapter."""

    value: int | float
    label: str  # e.g., "loops=4", "thinking=4k tokens"

    def __str__(self) -> str:
        return self.label


@dataclass(frozen=True)
class Prediction:
    """A single model output plus optional diagnostic metadata."""

    text: str
    metadata: dict


class ModelAdapter(ABC):
    """Adapter over a reasoning system with a controllable compute knob."""

    #: Adapter identifier for CLI / results.
    name: str = "abstract-adapter"

    @property
    @abstractmethod
    def compute_axis_name(self) -> str:
        """Short label for the compute axis (e.g., 'n_loops', 'thinking_budget_tokens')."""
        ...

    @abstractmethod
    def default_compute_grid(self) -> list[ComputeLevel]:
        """Sensible default sweep for this adapter."""
        ...

    @abstractmethod
    def predict(self, prompts: list[str], compute: ComputeLevel) -> list[Prediction]:
        """Run the model at the given compute level on a batch of prompts."""
        ...

    def teardown(self) -> None:
        """Release model resources (GPU memory, sessions, etc.). Default: no-op."""
