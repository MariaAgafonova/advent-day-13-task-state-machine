"""Task orchestration. Every significant operation is persisted immediately."""
from __future__ import annotations

from copy import deepcopy
import logging
from threading import RLock
from time import perf_counter
from uuid import uuid4

from task_llm import TaskBackend, ValidationResult
from task_models import ExpectedAction, PauseInfo, StepStatus, TaskError, TaskRequestLog, TaskStage, TaskState, utc_now
from task_repository import TaskBusyError, TaskRepository
from task_state_machine import TaskStateMachine


class TaskService:
    _active: set[tuple[str, str]] = set()
    _active_lock = RLock()

    def __init__(self, repository: TaskRepository, backend: TaskBackend) -> None:
        self.repository = repository
        self.backend = backend
        self.machine = TaskStateMachine(repository)

    def _key(self, task_id: str) -> tuple[str, str]:
        return (str(getattr(self.repository, "directory", id(self.repository))), task_id)

    def is_busy(self, task_id: str) -> bool:
        with self._active_lock:
            return self._key(task_id) in self._active

    def delete_task(self, task_id: str) -> None:
        with self.repository.locked(task_id):
            if self.is_busy(task_id):
                raise TaskBusyError("Задача выполняет запрос. Дождитесь его завершения перед удалением.")
            self.repository.delete(task_id)

    def clear_tasks(self) -> list[str]:
        with self.repository.collection_locked():
            task_ids = self.repository.list_task_ids()
            if any(self.is_busy(task_id) for task_id in task_ids):
                raise TaskBusyError("Есть выполняющийся запрос. Дождитесь его завершения перед очисткой задач.")
            deleted = []
            for task_id in task_ids:
                try:
                    self.repository.delete(task_id)
                except TaskError as error:
                    raise TaskError(
                        f"Удалено {len(deleted)} из {len(task_ids)} задач. {error}"
                    ) from error
                deleted.append(task_id)
            return deleted

    def _save(self, task: TaskState) -> TaskState:
        task.updated_at = utc_now()
        self.repository.save(task)
        return task

    @staticmethod
    def _effective_stage(task: TaskState) -> TaskStage:
        return task.pause_info.previous_stage if task.pause_info else task.stage

    @staticmethod
    def _action(task: TaskState, action: ExpectedAction | None) -> None:
        if task.stage == TaskStage.PAUSED:
            task.pause_info.previous_action = action
        else:
            task.expected_action = action

    @staticmethod
    def _step(task: TaskState):
        return next((s for s in task.steps if s.id == task.current_step_id), None)

    def _next(self, task: TaskState) -> None:
        step = next((s for s in task.steps if s.status != StepStatus.COMPLETED), None)
        task.current_step_id = step.id if step else None
        self._action(task, ExpectedAction(
            "agent", "fix_validation_issue" if step and task.validation_issues else
            "execute_step" if step else "validate_result",
            f"Выполнить шаг {step.id}: {step.title}" if step else "Проверить результат по цели и критериям",
        ))

    def create_task(self, goal: str, profile_id: str | None = None) -> TaskState:
        if not isinstance(goal, str) or not goal.strip() or len(goal) > 3000:
            raise TaskError("Укажите цель задачи: от 1 до 3000 символов.")
        selected_profile = profile_id if profile_id is not None else getattr(self.backend, "profile_id", None)
        if selected_profile is not None:
            if not isinstance(selected_profile, str) or not selected_profile.strip() or len(selected_profile) > 200:
                raise TaskError("profile_id должен быть непустой строкой до 200 символов.")
            selected_profile = selected_profile.strip()
        now = utc_now()
        task = TaskState(
            task_id=f"task-{uuid4().hex[:16]}", goal=goal.strip(),
            stage=TaskStage.PLANNING, current_step_id=None, steps=[],
            expected_action=ExpectedAction("agent", "create_plan", "Проанализировать цель и составить план"),
            pause_info=None, validation_issues=[], final_result=None, created_at=now, updated_at=now,
            profile_id=selected_profile,
        )
        self.repository.create(task)
        return task

    def _claim(self, task: TaskState) -> TaskState:
        key = self._key(task.task_id)
        with self._active_lock:
            if key in self._active:
                raise TaskError("Действие этой задачи уже выполняется. Можно поставить его на паузу.")
            self._active.add(key)
        task.operation_id = uuid4().hex
        if task.profile_id is None:
            task.profile_id = getattr(self.backend, "profile_id", None)
        for record in task.request_logs:
            if record.status == "running":
                record.status = "interrupted"
                record.finished_at = utc_now()
                record.error = "Предыдущий процесс завершился до сохранения ответа."
        task.request_logs.append(TaskRequestLog(
            request_id=task.operation_id,
            operation={TaskStage.PLANNING: "plan", TaskStage.EXECUTION: "execute", TaskStage.VALIDATION: "validate"}[task.stage],
            stage=task.stage.value, step_id=task.current_step_id, started_at=utc_now(),
            mode=getattr(self.backend, "mode", "demo"), profile_id=task.profile_id,
        ))
        task.request_logs = task.request_logs[-100:]
        try:
            return self._save(task)
        except Exception:
            self._release(task.task_id)
            raise

    def _release(self, task_id: str) -> None:
        with self._active_lock:
            self._active.discard(self._key(task_id))

    def _latest(self, snapshot: TaskState) -> TaskState:
        task = self.repository.get(snapshot.task_id)
        if task.operation_id != snapshot.operation_id:
            raise TaskError("Ответ относится к устаревшей операции; сохранённое состояние не изменено.")
        record = next((r for r in snapshot.request_logs if r.request_id == snapshot.operation_id), None)
        if record:
            task.request_logs = [
                deepcopy(record) if r.request_id == record.request_id else r for r in task.request_logs
            ]
        return task

    def _call(self, snapshot: TaskState, method, *args):
        record = next(r for r in snapshot.request_logs if r.request_id == snapshot.operation_id)
        started = perf_counter()
        try:
            result = method(snapshot, *args)
            record.status = "success"
            return result
        except Exception as error:
            record.status, record.error = "error", str(error)
            raise
        finally:
            record.finished_at = utc_now()
            record.elapsed_seconds = round(perf_counter() - started, 3)
            logging.getLogger(__name__).info(
                "Task request task_id=%s operation=%s step=%s status=%s profile=%s profile_used=%s tokens=%s elapsed=%.3fs",
                snapshot.task_id, record.operation, record.step_id, record.status,
                record.profile_id, record.profile_used, record.metrics.get("total_tokens", 0), record.elapsed_seconds,
            )

    def _failure(self, snapshot: TaskState, error: Exception) -> TaskState:
        with self.repository.locked(snapshot.task_id):
            task = self._latest(snapshot)
            task.last_error = str(error)
            record = next((r for r in task.request_logs if r.request_id == snapshot.operation_id), None)
            if record:
                record.status, record.error = "error", str(error)
                record.finished_at = record.finished_at or utc_now()
            task.operation_id = None
            stage = self._effective_stage(task)
            if stage == TaskStage.EXECUTION:
                step = self._step(task)
                if step:
                    step.status, step.error = StepStatus.FAILED, str(error)
                self._action(task, ExpectedAction("user", "retry_step", "Повторить незавершённый шаг командой продолжения"))
                if task.stage != TaskStage.PAUSED:
                    return self.machine.transition(task, TaskStage.FAILED, f"Ошибка шага: {error}")
            else:
                # Planning/validation transport errors retain their stage:
                # failed -> planning and failed -> validation are forbidden.
                self._action(task, ExpectedAction("agent", "retry_operation", "Повторить действие после ошибки"))
            return self._save(task)

    def plan_task(self, task_id: str) -> TaskState:
        with self.repository.locked(task_id):
            task = self.repository.get(task_id)
            if task.stage != TaskStage.PLANNING:
                raise TaskError("Планирование доступно только на этапе planning.")
            if any(question.answer is None for question in task.questions):
                return task
            if task.steps:
                self._next(task)
                return self.machine.transition(task, TaskStage.EXECUTION, "Сохранённый план готов")
            snapshot = deepcopy(self._claim(task))
        try:
            plan = self._call(snapshot, self.backend.plan)
            with self.repository.locked(task_id):
                task = self._latest(snapshot)
                task.operation_id, task.last_error = None, None
                if plan.question:
                    normalized = plan.question.question.strip().casefold()
                    if any(q.key == plan.question.key or q.question.strip().casefold() == normalized for q in task.questions):
                        raise TaskError("Модель повторила уже заданный вопрос; ответ сохранён. Повторите планирование.")
                    if len(task.questions) >= 12:
                        raise TaskError("Достигнут лимит уточнений; уточните цель новой задачи.")
                    task.questions.append(plan.question)
                    self._action(task, ExpectedAction("user", f"answer:{plan.question.key}", plan.question.question))
                    return self._save(task)
                if not plan.steps or not plan.criteria:
                    raise TaskError("План должен содержать шаги и критерии готовности.")
                task.steps, task.acceptance_criteria = plan.steps, plan.criteria
                self._next(task)
                if task.stage == TaskStage.PAUSED:
                    return self._save(task)
                return self.machine.transition(task, TaskStage.EXECUTION, "План сформирован")
        except Exception as error:
            return self._failure(snapshot, error)
        finally:
            self._release(task_id)

    def answer_question(self, task_id: str, answer: str) -> TaskState:
        if not isinstance(answer, str) or not answer.strip() or len(answer) > 6000:
            raise TaskError("Ответ должен содержать от 1 до 6000 символов.")
        with self.repository.locked(task_id):
            task = self.repository.get(task_id)
            if task.stage != TaskStage.PLANNING:
                raise TaskError("Сначала возобновите planning для ответа на вопрос.")
            question = next((q for q in task.questions if q.answer is None), None)
            if question is None:
                raise TaskError("Задача сейчас не ожидает ответа пользователя.")
            question.answer = answer.strip()
            task.expected_action = ExpectedAction("agent", "create_plan", "Продолжить планирование с сохранённым ответом")
            self._save(task)
        return self.plan_task(task_id)

    def start_current_step(self, task_id: str) -> TaskState:
        """Persist a start independently, allowing a checkpoint before the call."""
        with self.repository.locked(task_id):
            task = self.repository.get(task_id)
            if task.stage != TaskStage.EXECUTION:
                raise TaskError("Выполнение доступно только на этапе execution.")
            self._next(task)
            step = self._step(task)
            if step is None:
                return self.machine.transition(task, TaskStage.VALIDATION, "Все шаги завершены")
            step.status, step.error = StepStatus.IN_PROGRESS, None
            return self._save(task)

    def execute_current_step(self, task_id: str) -> TaskState:
        with self.repository.locked(task_id):
            with self._active_lock:
                if self._key(task_id) in self._active:
                    raise TaskError("Действие этой задачи уже выполняется.")
            task = self.start_current_step(task_id)
            if task.stage == TaskStage.VALIDATION:
                return task
            snapshot = deepcopy(self._claim(task))
        try:
            result = self._call(snapshot, self.backend.execute, self._step(snapshot))
            if not isinstance(result, str) or not result.strip():
                raise TaskError("Шаг не вернул результат.")
            with self.repository.locked(task_id):
                task = self._latest(snapshot)
                step = next(s for s in task.steps if s.id == snapshot.current_step_id)
                step.status, step.result, step.error = StepStatus.COMPLETED, result.strip(), None
                task.operation_id, task.last_error = None, None
                self._next(task)
                if task.current_step_id is None and task.stage != TaskStage.PAUSED:
                    return self.machine.transition(task, TaskStage.VALIDATION, "Все шаги завершены")
                return self._save(task)
        except Exception as error:
            return self._failure(snapshot, error)
        finally:
            self._release(task_id)

    def pause_task(self, task_id: str, reason: str = "Команда пользователя") -> TaskState:
        with self.repository.locked(task_id):
            task = self.repository.get(task_id)
            if task.stage == TaskStage.PAUSED:
                return task
            if task.stage == TaskStage.DONE:
                raise TaskError("Задача уже завершена. Пауза не требуется.")
            task.pause_info = PauseInfo(task.stage, reason, utc_now(), deepcopy(task.expected_action))
            task.expected_action = ExpectedAction("user", "resume_task", "Продолжить выполнение задачи")
            return self.machine.transition(task, TaskStage.PAUSED, reason)

    def resume_task(self, task_id: str) -> TaskState:
        with self.repository.locked(task_id):
            task = self.repository.get(task_id)
            if task.stage == TaskStage.DONE:
                return task
            if task.stage != TaskStage.PAUSED:
                raise TaskError("Задача не находится на паузе. Используйте /continue-task.")
            previous = task.pause_info.previous_stage
            task.expected_action = task.pause_info.previous_action
            return self.machine.transition(task, previous, "Пользователь продолжил задачу")

    def validate_task(self, task_id: str) -> TaskState:
        with self.repository.locked(task_id):
            task = self.repository.get(task_id)
            if task.stage != TaskStage.VALIDATION:
                raise TaskError("Проверка доступна только на этапе validation.")
            snapshot = deepcopy(self._claim(task))
        try:
            broken = [s for s in snapshot.steps if s.status != StepStatus.COMPLETED or not (s.result or "").strip() or s.error]
            if broken:
                check = ValidationResult(
                    [f"Шаг {s.id}: отсутствует результат или осталась ошибка" for s in broken],
                    [s.id for s in broken],
                )
                record = snapshot.request_logs[-1]
                record.mode, record.status, record.finished_at = "local", "success", utc_now()
            else:
                check = self._call(snapshot, self.backend.validate)
            with self.repository.locked(task_id):
                task = self._latest(snapshot)
                task.operation_id, task.last_error = None, None
                task.validation_issues = check.issues
                if check.issues:
                    if not check.step_ids or any(x not in {s.id for s in task.steps} for x in check.step_ids):
                        raise TaskError("Проверка не указала корректные шаги для исправления.")
                    for step in task.steps:
                        if step.id in check.step_ids:
                            step.status, step.error = StepStatus.PENDING, None
                    task.final_result = None
                    self._next(task)
                    if task.stage == TaskStage.PAUSED:
                        self._action(task, ExpectedAction("agent", "apply_validation", "Применить сохранённые замечания"))
                        return self._save(task)
                    return self.machine.transition(task, TaskStage.EXECUTION, "Валидация обнаружила замечания")
                if not isinstance(check.final_result, str) or not check.final_result.strip():
                    raise TaskError("Проверка не вернула итоговый результат.")
                task.final_result = check.final_result.strip()
                task.current_step_id = None
                self._action(task, None)
                if task.stage == TaskStage.PAUSED:
                    self._action(task, ExpectedAction("agent", "apply_validation", "Сохранить успешный итог проверки"))
                    return self._save(task)
                return self.machine.transition(task, TaskStage.DONE, "Результат проверен")
        except Exception as error:
            return self._failure(snapshot, error)
        finally:
            self._release(task_id)

    def continue_task(self, task_id: str, answer: str | None = None) -> TaskState:
        if answer is not None:
            return self.answer_question(task_id, answer)
        task = self.repository.get(task_id)
        if task.stage in {TaskStage.PAUSED, TaskStage.DONE}:
            return task
        if task.stage == TaskStage.PLANNING:
            return self.plan_task(task_id)
        if task.stage == TaskStage.FAILED:
            with self.repository.locked(task_id):
                task = self.repository.get(task_id)
                if task.current_step_id is None:
                    raise TaskError("Для повтора отсутствует незавершённый шаг.")
                self.machine.transition(task, TaskStage.EXECUTION, "Повтор неуспешного шага")
            return self.execute_current_step(task_id)
        if task.stage == TaskStage.EXECUTION:
            return self.execute_current_step(task_id)
        if task.expected_action and task.expected_action.action_type == "apply_validation":
            with self.repository.locked(task_id):
                task = self.repository.get(task_id)
                if task.validation_issues:
                    self._next(task)
                    return self.machine.transition(task, TaskStage.EXECUTION, "Применены сохранённые замечания")
                task.expected_action = None
                return self.machine.transition(task, TaskStage.DONE, "Применена сохранённая успешная проверка")
        return self.validate_task(task_id)
