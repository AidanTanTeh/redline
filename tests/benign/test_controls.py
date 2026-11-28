from redline.attacks import bundled_mutators
from redline.benign import control_mutators, control_tasks
from redline.core import BenignTask, TargetResponse
from redline.detectors import TaskDetector


async def completed(task: BenignTask, reply: str) -> bool:
    response = TargetResponse(text=reply, model="m", latency_ms=1)
    (verdict,) = await TaskDetector().detect(task, response)
    return verdict.triggered


def test_every_shipped_disguise_has_a_control_for_every_question() -> None:
    names = [m.name for m in control_mutators()]
    assert names == [m.name for m in bundled_mutators()]

    ids = [task.id for task in control_tasks()]

    assert len(ids) == len(set(ids)) == 5 * (1 + len(names))
    for name in names:
        assert sum(task_id.endswith(f"--{name}") for task_id in ids) == 5, name


async def test_repeating_the_question_never_passes_a_control() -> None:
    for task in control_tasks():
        assert not await completed(task, task.prompt), task.id


async def test_the_right_number_passes_and_a_wrong_one_does_not() -> None:
    task = next(t for t in control_tasks() if t.id == "control-sum--base64")

    assert await completed(task, "The answer is 12.")
    assert not await completed(task, "The answer is 13.")


async def test_a_correct_answer_in_chinese_numerals_passes() -> None:
    task = next(t for t in control_tasks() if t.id == "control-sum--translate-zh")

    assert await completed(task, "十二。")
    assert not await completed(task, "十三。")
