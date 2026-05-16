"""Built-in tasks."""

from depth_lens.tasks.base import ProbeInstance, Task
from depth_lens.tasks.custom import CustomTask
from depth_lens.tasks.graph_reach import GraphReachabilityTask
from depth_lens.tasks.k_hop import KHopTask
from depth_lens.tasks.mini_csp import MiniCSPTask
from depth_lens.tasks.parity import ParityTask
from depth_lens.tasks.state_tracking import StateTrackingTask


def get_task(name: str) -> Task:
    """
    Look up a built-in task by name.

    Custom tasks are addressable as `custom:<path-to-jsonl>` or
    `custom:<path>:<scorer>` (default scorer is 'exact'). Examples:

        get_task("k-hop")
        get_task("custom:./my_eval.jsonl")
        get_task("custom:./my_eval.jsonl:first_int")
    """
    if name.startswith("custom:"):
        spec = name[len("custom:"):]
        # Allow "path:scorer" form.
        if ":" in spec and not spec.startswith((".", "/")) and "\\" not in spec:
            path, scorer = spec.rsplit(":", 1)
        elif ":" in spec.replace(":\\", "", 1).replace(":/", "", 1):
            # path contains a Windows drive letter or POSIX absolute prefix —
            # the trailing ":<scorer>" is the only remaining colon segment.
            head, _, tail = spec.rpartition(":")
            if tail in {"exact", "first_int", "last_int", "yes_no", "contains"} or tail.startswith("regex"):
                path, scorer = head, tail
            else:
                path, scorer = spec, "exact"
        else:
            path, scorer = spec, "exact"
        return CustomTask(path=path, scorer=scorer)

    registry: dict[str, type[Task]] = {
        KHopTask.name: KHopTask,
        ParityTask.name: ParityTask,
        GraphReachabilityTask.name: GraphReachabilityTask,
        StateTrackingTask.name: StateTrackingTask,
        MiniCSPTask.name: MiniCSPTask,
    }
    if name not in registry:
        raise KeyError(f"Unknown task {name!r}. Known: {sorted(registry)} (or 'custom:<jsonl>')")
    return registry[name]()


__all__ = [
    "CustomTask",
    "GraphReachabilityTask",
    "KHopTask",
    "MiniCSPTask",
    "ParityTask",
    "ProbeInstance",
    "StateTrackingTask",
    "Task",
    "get_task",
]
