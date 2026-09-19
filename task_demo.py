"""Deterministic Day 13 scenario, including a real child-process restart."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from task_commands import task_status
from task_llm import PlanResult, ValidationResult
from task_models import TaskQuestion, TaskStage, TaskStep
from task_repository import JsonTaskRepository
from task_service import TaskService

GOAL = "Подготовить краткое описание вакансии Android-разработчика."


class DemoTaskBackend:
    """An explicitly offline model double for a repeatable teaching example."""
    def __init__(self):
        self.executed: list[int] = []
        self.plan_calls = 0
        self.validation_calls = 0

    def plan(self, task):
        self.plan_calls += 1
        if not any(q.key == "developer_level" and q.answer for q in task.questions):
            return PlanResult([], [], TaskQuestion("developer_level", "Какой уровень Android-разработчика требуется?"))
        return PlanResult([
            TaskStep(1, "Собрать требования", "Определить уровень, стек и обязательные разделы вакансии"),
            TaskStep(2, "Создать черновик", "Написать краткую вакансию с требованиями, включая опыт"),
            TaskStep(3, "Добавить обязанности", "Сформулировать задачи разработчика"),
        ], ["Указан уровень Senior", "Есть стек Kotlin и Compose", "Есть требования к опыту", "Есть обязанности"])

    def execute(self, task, step):
        self.executed.append(step.id)
        if step.id == 1:
            level = next(q.answer for q in task.questions if q.key == "developer_level")
            return f"Уровень: {level}. Стек: Kotlin, Android SDK, Jetpack Compose, Coroutines. Нужны опыт и обязанности."
        if step.id == 2:
            result = "Ищем Senior Android-разработчика.\nСтек: Kotlin, Android SDK, Jetpack Compose, Coroutines."
            if task.validation_issues:
                result += "\nОпыт: от 5 лет Android-разработки, проектирование приложений и сопровождение релизов."
            return result
        return "Обязанности: разрабатывать функции приложения, проводить code review и улучшать стабильность."

    def validate(self, task):
        self.validation_calls += 1
        draft = next(s.result for s in task.steps if s.id == 2) or ""
        if "Опыт:" not in draft:
            return ValidationResult(["В описании отсутствуют требования к опыту кандидата"], [2])
        duties = next(s.result for s in task.steps if s.id == 3)
        return ValidationResult([], [], f"{draft}\n{duties}")


def finish(directory: Path, task_id: str) -> dict:
    repository = JsonTaskRepository(directory)
    backend = DemoTaskBackend()
    service = TaskService(repository, backend)
    saved = repository.get(task_id)
    first_result = saved.steps[0].result
    assert saved.stage == TaskStage.PAUSED and saved.current_step_id == 2
    print("\nНовый процесс: загружен JSON. Первый шаг сохранён, повторного вопроса нет.")
    task = service.resume_task(task_id)
    print("Продолжаю со второго шага: «Создать черновик». Первый шаг повторяться не будет.")
    for _ in range(10):
        task = service.continue_task(task_id)
        print("\n" + task_status(task))
        if task.stage == TaskStage.DONE:
            break
    assert task.stage == TaskStage.DONE
    assert backend.executed == [2, 3, 2], backend.executed
    assert backend.plan_calls == 0
    assert task.steps[0].result == first_result
    report = {"task_id": task_id, "executed_after_restart": backend.executed,
              "plan_calls_after_restart": backend.plan_calls, "state": task.to_dict()}
    (directory / "reports").mkdir(exist_ok=True)
    (directory / "reports" / f"{task_id}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print("\nЖУРНАЛ ПЕРЕХОДОВ")
    for entry in task.transition_history:
        print(f"{entry.from_stage.value} → {entry.to_stage.value}: {entry.reason}")
    print("\nИТОГОВОЕ СОСТОЯНИЕ")
    print(json.dumps(task.to_dict(), ensure_ascii=False, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="scenario always uses the offline model double")
    parser.add_argument("--directory", type=Path, default=Path("data/demo-tasks"))
    parser.add_argument("--phase", choices=("start", "finish"), default="start")
    parser.add_argument("--task-id")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.phase == "finish":
        finish(directory, args.task_id)
        return 0
    service = TaskService(JsonTaskRepository(directory), DemoTaskBackend())
    task = service.create_task(GOAL)
    print(task_status(task), flush=True)
    task = service.continue_task(task.task_id)
    print("\n" + task_status(task), flush=True)
    task = service.answer_question(task.task_id, "Senior")
    print("\nПользователь: Senior\n" + task_status(task), flush=True)
    task = service.execute_current_step(task.task_id)
    task = service.start_current_step(task.task_id)
    task = service.pause_task(task.task_id, "Демонстрация перезапуска во время второго шага")
    print("\n" + task_status(task), flush=True)
    print("\nСостояние записано. Запускается новый процесс без объектов предыдущего агента.", flush=True)
    subprocess.run([
        sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()),
        "--phase", "finish", "--directory", str(directory), "--task-id", task.task_id,
    ], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
