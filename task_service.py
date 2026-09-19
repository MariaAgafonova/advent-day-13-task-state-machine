"""Task orchestration. Every significant operation is persisted immediately."""
from __future__ import annotations

from copy import deepcopy
from threading import RLock
from uuid import uuid4

from task_llm import TaskBackend, ValidationResult
from task_models import ExpectedAction, PauseInfo, StepStatus, TaskError, TaskStage, TaskState, utc_now
from task_repository import TaskRepository
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

    def create_task(self, goal: str) -> TaskState:
        if not isinstance(goal, str) or not goal.strip() or len(goal) > 3000:
            raise TaskError("Укажите цель задачи: от 1 до 3000 символов.")
        now = utc_now()
        task = TaskState(
            task_id=f"task-{uuid4().hex[:16]}", goal=goal.strip(),
            stage=TaskStage.PLANNING, current_step_id=None, steps=[],
            expected_action=ExpectedAction("agent", "create_plan", "Проанализировать цель и составить план"),
            pause_info=None, validation_issues=[], final_result=None, created_at=now, updated_at=now,
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
        return task

    def _failure(self, snapshot: TaskState, error: Exception) -> TaskState:
        with self.repository.locked(snapshot.task_id):
            task = self._latest(snapshot)
            task.last_error = str(error)
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
            plan = self.backend.plan(snapshot)
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
            result = self.backend.execute(snapshot, self._step(snapshot))
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
            else:
                check = self.backend.validate(snapshot)
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
