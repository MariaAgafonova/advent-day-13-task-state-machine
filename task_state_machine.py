"""The only component allowed to change the task stage."""
from __future__ import annotations

from copy import deepcopy
import logging

from task_models import StateTransition, StepStatus, TaskError, TaskStage, TaskState, utc_now
from task_repository import TaskRepository

ALLOWED_TRANSITIONS = {
    TaskStage.PLANNING: {TaskStage.EXECUTION, TaskStage.PAUSED, TaskStage.FAILED},
    TaskStage.EXECUTION: {TaskStage.VALIDATION, TaskStage.PAUSED, TaskStage.FAILED},
    TaskStage.VALIDATION: {TaskStage.EXECUTION, TaskStage.DONE, TaskStage.PAUSED, TaskStage.FAILED},
    TaskStage.PAUSED: {TaskStage.PLANNING, TaskStage.EXECUTION, TaskStage.VALIDATION, TaskStage.FAILED},
    TaskStage.FAILED: {TaskStage.EXECUTION, TaskStage.PAUSED},
    TaskStage.DONE: set(),
}


class InvalidTaskTransitionError(TaskError):
    pass


class TaskStateMachine:
    def __init__(self, repository: TaskRepository) -> None:
        self.repository = repository

    def transition(self, task: TaskState, target_stage: TaskStage, reason: str = "") -> TaskState:
        with self.repository.locked(task.task_id):
            stored = self.repository.get(task.task_id)
            if stored.stage != task.stage or stored.updated_at != task.updated_at:
                raise InvalidTaskTransitionError("Состояние изменилось; загрузите задачу повторно.")
            source = task.stage
            if target_stage not in ALLOWED_TRANSITIONS[source]:
                raise InvalidTaskTransitionError(f"Недопустимый переход: {source.value} → {target_stage.value}.")
            if source == TaskStage.PAUSED and target_stage != TaskStage.FAILED:
                if task.pause_info is None or target_stage != task.pause_info.previous_stage:
                    raise InvalidTaskTransitionError("Из паузы можно вернуться только в сохранённый этап.")
            if target_stage == TaskStage.EXECUTION and source != TaskStage.PAUSED and (
                not task.steps or task.current_step_id is None
            ):
                raise InvalidTaskTransitionError("Для выполнения требуется план и текущий шаг.")
            if target_stage == TaskStage.VALIDATION and source != TaskStage.PAUSED and (
                not task.steps or any(s.status != StepStatus.COMPLETED for s in task.steps)
            ):
                raise InvalidTaskTransitionError("Перед валидацией завершите все шаги.")
            updated = deepcopy(task)
            updated.stage = target_stage
            updated.updated_at = utc_now()
            if source == TaskStage.PAUSED:
                updated.pause_info = None
            updated.transition_history.append(StateTransition(source, target_stage, reason, updated.updated_at))
            self.repository.save(updated)
            logging.getLogger(__name__).info("Task %s: %s -> %s (%s)", task.task_id, source.value, target_stage.value, reason)
            return updated
