"""Persistent task data. The state, not the chat transcript, owns progress."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import re
from typing import Any


class TaskError(ValueError):
    """An actionable task error suitable for CLI and web users."""


class TaskStage(str, Enum):
    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    PAUSED = "paused"
    FAILED = "failed"
    DONE = "done"


class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass
class TaskStep:
    id: int
    title: str
    description: str
    status: StepStatus = StepStatus.PENDING
    result: str | None = None
    error: str | None = None


@dataclass
class ExpectedAction:
    actor: str
    action_type: str
    description: str

    def __post_init__(self) -> None:
        if self.actor not in {"user", "agent", "tool"}:
            raise TaskError("actor должен быть user, agent или tool.")
        if not all(isinstance(x, str) and x.strip() for x in (self.action_type, self.description)):
            raise TaskError("Действие должно иметь тип и описание.")


@dataclass
class PauseInfo:
    previous_stage: TaskStage
    reason: str
    paused_at: str
    previous_action: ExpectedAction | None = None


@dataclass
class StateTransition:
    from_stage: TaskStage
    to_stage: TaskStage
    reason: str
    timestamp: str


@dataclass
class TaskQuestion:
    key: str
    question: str
    answer: str | None = None


@dataclass
class TaskState:
    task_id: str
    goal: str
    stage: TaskStage
    current_step_id: int | None
    steps: list[TaskStep]
    expected_action: ExpectedAction | None
    pause_info: PauseInfo | None
    validation_issues: list[str]
    final_result: str | None
    created_at: str
    updated_at: str
    transition_history: list[StateTransition] = field(default_factory=list)
    questions: list[TaskQuestion] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    last_error: str | None = None
    operation_id: str | None = None
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TaskState:
        try:
            data = dict(raw)
            if data.get("schema_version", 1) != 1:
                raise ValueError("неподдерживаемая версия состояния")
            data["stage"] = TaskStage(data["stage"])
            data["steps"] = [
                TaskStep(**{**item, "status": StepStatus(item["status"])})
                for item in data["steps"]
            ]
            if data.get("expected_action") is not None:
                data["expected_action"] = ExpectedAction(**data["expected_action"])
            if data.get("pause_info") is not None:
                pause = dict(data["pause_info"])
                pause["previous_stage"] = TaskStage(pause["previous_stage"])
                if pause.get("previous_action") is not None:
                    pause["previous_action"] = ExpectedAction(**pause["previous_action"])
                data["pause_info"] = PauseInfo(**pause)
            data["transition_history"] = [
                StateTransition(**{
                    **item, "from_stage": TaskStage(item["from_stage"]),
                    "to_stage": TaskStage(item["to_stage"]),
                }) for item in data.get("transition_history", [])
            ]
            data["questions"] = [TaskQuestion(**item) for item in data.get("questions", [])]
            task = cls(**data)
            task.check()
            return task
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise TaskError(f"Некорректное состояние задачи: {error}") from error

    def check(self) -> None:
        if not isinstance(self.task_id, str) or not re.fullmatch(r"task-[a-zA-Z0-9-]{1,80}", self.task_id):
            raise TaskError("Некорректный task_id.")
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise TaskError("Цель задачи не может быть пустой.")
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)) or any(type(value) is not int or value < 1 for value in ids):
            raise TaskError("ID шагов должны быть уникальными положительными числами.")
        if self.current_step_id is not None and self.current_step_id not in ids:
            raise TaskError("Текущий шаг отсутствует в плане.")
        for step in self.steps:
            if not isinstance(step.title, str) or not step.title.strip():
                raise TaskError("Шаг должен иметь название.")
            if not isinstance(step.description, str) or not step.description.strip():
                raise TaskError("Шаг должен иметь описание.")
            for value in (step.result, step.error):
                if value is not None and not isinstance(value, str):
                    raise TaskError("Результат и ошибка шага должны быть строками.")
        for values in (self.validation_issues, self.acceptance_criteria):
            if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
                raise TaskError("Критерии и замечания должны быть списками строк.")
        keys = [q.key for q in self.questions]
        if len(keys) != len(set(keys)):
            raise TaskError("Идентификатор вопроса повторяется.")
        for question in self.questions:
            if not all(isinstance(x, str) and x.strip() for x in (question.key, question.question)):
                raise TaskError("Вопрос должен иметь идентификатор и текст.")
            if question.answer is not None and (not isinstance(question.answer, str) or not question.answer.strip()):
                raise TaskError("Сохранённый ответ должен быть непустой строкой.")
        if self.final_result is not None and not isinstance(self.final_result, str):
            raise TaskError("Итоговый результат должен быть строкой.")
        if self.stage == TaskStage.PAUSED and self.pause_info is None:
            raise TaskError("Для paused требуется pause_info.")
        if self.stage == TaskStage.DONE:
            if not self.steps or any(s.status != StepStatus.COMPLETED for s in self.steps):
                raise TaskError("Завершённая задача содержит незавершённые шаги.")
            if not self.final_result or self.validation_issues:
                raise TaskError("Для done требуется проверенный итог без замечаний.")
            if self.current_step_id is not None or self.expected_action is not None:
                raise TaskError("У завершённой задачи нет текущего шага или ожидаемого действия.")
            if any(not (s.result or "").strip() or s.error for s in self.steps):
                raise TaskError("У завершённых шагов должны быть результаты без ошибок.")
