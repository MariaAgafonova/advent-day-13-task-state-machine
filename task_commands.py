"""One command vocabulary for the CLI and Flask chat."""
from dataclasses import asdict, dataclass
import json

from task_models import StepStatus, TaskError, TaskStage, TaskState
from task_service import TaskService

COMMANDS = {"/new-task", "/task-status", "/pause-task", "/resume-task", "/continue-task", "/task-logs", "/list-tasks"}


def task_status(task: TaskState) -> str:
    count = sum(s.status == StepStatus.COMPLETED for s in task.steps)
    lines = [f"Задача: {task.task_id}", f"Цель: {task.goal}",
             f"Этап: {task.stage.value}", f"Выполнено: {count} из {len(task.steps)} шагов"]
    lines.append(f"Профиль задачи: {task.profile_id or 'будет выбран при первом запросе'}")
    current = next((s for s in task.steps if s.id == task.current_step_id), None)
    if current:
        lines.append(f"Текущий шаг: {current.id} — {current.title}")
    if task.pause_info:
        lines.append(f"Пауза на этапе {task.pause_info.previous_stage.value}: {task.pause_info.reason}")
    if task.expected_action:
        lines.append(f"Ожидается ({task.expected_action.actor}): {task.expected_action.description}")
    if task.last_error:
        lines.append(f"Ошибка: {task.last_error}")
    if task.validation_issues:
        lines.append("Замечания: " + "; ".join(task.validation_issues))
    if task.stage == TaskStage.DONE:
        lines.append("Задача уже завершена. Продолжение не требуется.")
        lines.append(task.final_result or "")
    if task.request_logs:
        record = task.request_logs[-1]
        lines.append(
            f"[лог] {record.operation}: {record.status}, {record.elapsed_seconds:.3f} с, "
            f"токены={record.metrics.get('total_tokens', 0)}, "
            f"профиль={record.profile_id or '—'}, включён в промпт={record.profile_used}"
        )
        if record.profile_used:
            settings = record.profile_settings
            lines.append(
                f"[профиль] язык={settings.get('language')}, стиль={settings.get('responseStyle')}, "
                f"формат={settings.get('preferredFormat')}, объём={settings.get('responseLength')}"
            )
    return "\n".join(lines)


@dataclass
class CommandResult:
    message: str
    task: TaskState | None = None


class TaskCommands:
    def __init__(self, service: TaskService) -> None:
        self.service = service
        self.active_task_id: str | None = None

    def handle(self, message: str) -> CommandResult | None:
        if not isinstance(message, str):
            raise TaskError("Команда должна быть строкой.")
        parts = message.strip().split(maxsplit=1)
        if not parts or parts[0].lower() not in COMMANDS:
            return None
        command = parts[0].lower()
        argument = parts[1].strip() if len(parts) > 1 else ""
        if command == "/list-tasks":
            if argument:
                raise TaskError("Использование: /list-tasks")
            tasks = self.service.repository.list_tasks()
            return CommandResult("\n\n".join(task_status(t) for t in tasks) or "Задач пока нет.")
        if command == "/new-task":
            task = self.service.create_task(argument)
        else:
            args = argument.split(maxsplit=1)
            if not args:
                raise TaskError(f"Использование: {command} <task_id>")
            task_id = args[0]
            extra = args[1] if len(args) > 1 else None
            if extra and command not in {"/continue-task", "/pause-task"}:
                raise TaskError(f"Использование: {command} <task_id>")
            if command in {"/task-status", "/task-logs"}:
                task = self.service.repository.get(task_id)
                if command == "/task-logs":
                    self.active_task_id = task_id
                    return CommandResult(json.dumps(
                        [asdict(record) for record in task.request_logs],
                        ensure_ascii=False, indent=2,
                    ), task)
            elif command == "/pause-task":
                task = self.service.pause_task(task_id, extra or "Команда пользователя")
            elif command == "/resume-task":
                task = self.service.resume_task(task_id)
                if task.stage != TaskStage.DONE and not self.service.is_busy(task_id):
                    task = self.service.continue_task(task_id)
            else:
                task = self.service.continue_task(task_id, extra)
        self.active_task_id = task.task_id
        return CommandResult(task_status(task), task)

    def answer_if_waiting(self, message: str) -> CommandResult | None:
        if self.active_task_id is None or message.startswith("/"):
            return None
        task = self.service.repository.get(self.active_task_id)
        if task.stage != TaskStage.PLANNING or not any(q.answer is None for q in task.questions):
            return None
        task = self.service.answer_question(task.task_id, message)
        return CommandResult(task_status(task), task)
