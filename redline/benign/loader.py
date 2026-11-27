"""Load and validate the YAML benign tasks."""

from pathlib import Path
from typing import Literal

from redline.core import BenignTask, RedlineModel
from redline.core.yaml_files import read_yaml_dir

BUNDLED_TASKS_DIR = Path(__file__).parent / "tasks"


class _TaskFile(RedlineModel):
    schema_version: Literal[1]
    tasks: tuple[BenignTask, ...]


class TaskSet(RedlineModel):
    """All loaded tasks plus a fingerprint of the files they came from."""

    tasks: tuple[BenignTask, ...]
    content_hash: str


def load_tasks(path: Path | None = None) -> TaskSet:
    """Load every `*.yaml` file in `path` (default: the bundled tasks).

    Raises CorpusError for the same problems as `load_corpus`.
    """
    tasks, content_hash = read_yaml_dir(
        path if path is not None else BUNDLED_TASKS_DIR,
        _TaskFile,
        lambda file: file.tasks,
        "task",
    )
    return TaskSet(tasks=tasks, content_hash=content_hash)
