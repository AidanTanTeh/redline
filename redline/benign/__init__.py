"""Benign tasks: normal requests a target should still complete."""

from redline.benign.loader import TaskSet, load_tasks
from redline.core.yaml_files import CorpusError

__all__ = [
    "CorpusError",
    "TaskSet",
    "load_tasks",
]
