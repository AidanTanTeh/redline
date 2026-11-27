import re
from collections import Counter
from pathlib import Path

import pytest

from redline.attacks import load_corpus
from redline.benign import CorpusError, load_tasks
from redline.core import BenignTask, TaskKind, contains_phrase
from redline.demo import CANARY_PREFIX

TASKS = load_tasks().tasks


def everything_in(task: BenignTask) -> str:
    return task.model_dump_json()


def documents_of(task: BenignTask) -> str:
    return " ".join(d.content for d in task.documents)


def test_bundled_tasks_load_with_the_expected_mix() -> None:
    kinds = Counter(task.kind for task in TASKS)
    assert kinds == {TaskKind.ANSWER: 6, TaskKind.SENSITIVE_SOUNDING: 3, TaskKind.TOOL: 6}
    assert len(load_tasks().content_hash) == 64


def test_task_ids_are_prefixed_and_never_collide_with_attacks() -> None:
    ids = {task.id for task in TASKS}
    assert all(task_id.startswith("bt-") for task_id in ids)
    assert not ids & {attack.id for attack in load_corpus().attacks}


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_every_expected_phrase_group_can_be_satisfied_from_the_documents(
    task: BenignTask,
) -> None:
    # A task whose answer isn't available would count as "failed" for any bot.
    for group in task.expected_text:
        assert any(contains_phrase(documents_of(task), phrase) for phrase in group), group


@pytest.mark.parametrize("task", [t for t in TASKS if t.expected_tool_call], ids=lambda t: t.id)
def test_every_expected_tool_argument_is_available_to_the_bot(task: BenignTask) -> None:
    assert task.expected_tool_call is not None
    available = f"{task.prompt} {documents_of(task)}"
    assert all(value in available for value in task.expected_tool_call.arguments.values())


def test_read_then_email_tasks_cannot_be_done_from_the_question_alone() -> None:
    reading = [t for t in TASKS if t.expected_tool_call and t.documents]
    assert len(reading) == 2
    for task in reading:
        assert task.expected_tool_call is not None
        assert task.expected_tool_call.arguments["body"] not in task.prompt


def test_tasks_are_safe_to_run() -> None:
    text = " ".join(everything_in(task) for task in TASKS)
    assert CANARY_PREFIX not in text
    addresses = re.findall(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)", text)
    assert addresses and all(domain.endswith(".example") for domain in addresses)


def test_loader_errors_name_the_task_file(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("schema_version: 1\ntasks:\n  - id: bt-x\n")
    with pytest.raises(CorpusError, match="bad.yaml"):
        load_tasks(tmp_path)


def test_loader_rejects_an_empty_or_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="no tasks found"):
        load_tasks(tmp_path)
    with pytest.raises(CorpusError, match="task directory not found"):
        load_tasks(tmp_path / "nope")
