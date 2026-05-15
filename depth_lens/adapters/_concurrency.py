"""
Concurrency helpers for API adapters.

The Anthropic / OpenAI / Gemini / vLLM adapters issue one HTTP call per
prompt. depth-lens probes are dominated by these calls — a single
1000-sample probe is 1000 sequential round trips, which is slow and a
massive cost-of-walltime even when the per-call cost is fine.

`parallel_map` runs `fn(prompt)` across a thread pool. It's the simplest
correctness-preserving way to parallelize IO-bound work; we avoid async
because:
  - depth-lens calls into adapters from synchronous user code (CLI, notebook).
  - SDK error-handling and retries are simpler in sync code.
  - The bottleneck is wall-clock, not CPU.

Order is preserved: result[i] corresponds to prompts[i].
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")
U = TypeVar("U")


def parallel_map(
    fn: Callable[[T], U],
    items: list[T],
    *,
    max_workers: int = 8,
) -> list[U]:
    """Apply `fn` to each item via a thread pool, preserving order."""
    if max_workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(fn, items))
