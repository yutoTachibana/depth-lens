"""depth-lens — reasoning-depth measurement toolkit."""

from depth_lens.metrics import ProbeResult, probe
from depth_lens.tasks.base import Task

__all__ = ["Task", "ProbeResult", "probe"]
__version__ = "1.0.0"
