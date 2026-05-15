"""Built-in tasks."""

from depth_lens.tasks.base import ProbeInstance, Task
from depth_lens.tasks.graph_reach import GraphReachabilityTask
from depth_lens.tasks.k_hop import KHopTask
from depth_lens.tasks.parity import ParityTask
from depth_lens.tasks.state_tracking import StateTrackingTask


def get_task(name: str) -> Task:
    """Look up a built-in task by name."""
    registry: dict[str, type[Task]] = {
        KHopTask.name: KHopTask,
        ParityTask.name: ParityTask,
        GraphReachabilityTask.name: GraphReachabilityTask,
        StateTrackingTask.name: StateTrackingTask,
    }
    if name not in registry:
        raise KeyError(f"Unknown task {name!r}. Known: {sorted(registry)}")
    return registry[name]()


__all__ = [
    "GraphReachabilityTask",
    "KHopTask",
    "ParityTask",
    "ProbeInstance",
    "StateTrackingTask",
    "Task",
    "get_task",
]
