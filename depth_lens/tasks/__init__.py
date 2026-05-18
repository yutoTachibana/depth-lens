"""Built-in tasks."""

from depth_lens.tasks.base import ProbeInstance, Task
from depth_lens.tasks.custom import CustomTask
from depth_lens.tasks.dict_lookup import DictLookupTask
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
        # The scorer suffix may itself contain colons (`llm:openai:gpt-5-mini:faithful`,
        # `regex:foo:bar`). To split path from scorer we look for a scorer-prefix
        # keyword and split there, instead of blindly rsplit'ing on the last colon.
        SCORER_PREFIXES = ("llm:", "regex:")
        SCORER_SIMPLE = {"exact", "first_int", "last_int", "yes_no", "contains"}
        path, scorer = spec, "exact"
        for prefix in SCORER_PREFIXES:
            marker = f":{prefix}"
            if marker in spec:
                idx = spec.index(marker)
                path = spec[:idx]
                scorer = spec[idx + 1:]
                break
        else:
            # No prefixed scorer — look for `:<simple-name>` at the end.
            head, sep, tail = spec.rpartition(":")
            if sep and tail in SCORER_SIMPLE:
                path, scorer = head, tail
        return CustomTask(path=path, scorer=scorer)

    registry: dict[str, type[Task]] = {
        KHopTask.name: KHopTask,
        ParityTask.name: ParityTask,
        GraphReachabilityTask.name: GraphReachabilityTask,
        StateTrackingTask.name: StateTrackingTask,
        MiniCSPTask.name: MiniCSPTask,
        DictLookupTask.name: DictLookupTask,
    }
    if name not in registry:
        raise KeyError(f"Unknown task {name!r}. Known: {sorted(registry)} (or 'custom:<jsonl>')")
    return registry[name]()


__all__ = [
    "CustomTask",
    "DictLookupTask",
    "GraphReachabilityTask",
    "KHopTask",
    "MiniCSPTask",
    "ParityTask",
    "ProbeInstance",
    "StateTrackingTask",
    "Task",
    "get_task",
]
