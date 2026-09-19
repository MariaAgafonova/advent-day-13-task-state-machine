"""Typed LLM boundary: the model proposes content, Python controls transitions."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Protocol
from copy import deepcopy

from profile import InMemoryProfileRepository, UserProfile, parse_request_overrides
from task_context import TaskContextBuilder
from task_models import TaskError, TaskQuestion, TaskState, TaskStep


@dataclass
class PlanResult:
    steps: list[TaskStep]
    criteria: list[str]
    question: TaskQuestion | None = None


@dataclass
class ValidationResult:
    issues: list[str]
    step_ids: list[int]
    final_result: str | None = None


class TaskBackend(Protocol):
    def plan(self, task: TaskState) -> PlanResult: ...
    def execute(self, task: TaskState, step: TaskStep) -> str: ...
    def validate(self, task: TaskState) -> ValidationResult: ...


def text_field(value, name: str, maximum: int = 12000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise TaskError(f"Ответ модели: {name} должен быть непустой строкой до {maximum} символов.")
    return value.strip()


def string_list(value, name: str) -> list[str]:
    if not isinstance(value, list) or len(value) > 12:
        raise TaskError(f"Ответ модели: {name} должен быть списком до 12 элементов.")
    return [text_field(item, name, 1000) for item in value]


class LLMTaskBackend:
    mode = "llm"

    def __init__(self, agent) -> None:
        self.agent = agent
        self.profile_id = agent.user_id
        self.context = TaskContextBuilder()
        self.last_request = None

    def _request(self, task: TaskState, instruction: str) -> dict:
        from agent import ChatAgent

        profile_id = task.profile_id or self.agent.user_id
        stored_profile = self.agent.profile_repository.get(profile_id)
        profile = stored_profile or UserProfile.defaults(profile_id)
        overrides = parse_request_overrides(task.goal)
        effective = profile.with_updates(overrides)
        record = next((r for r in task.request_logs if r.request_id == task.operation_id), None)
        if record:
            record.model = self.agent.config.model
            record.profile_id = profile_id
            record.profile_loaded = stored_profile is not None
            record.profile_settings = effective.to_dict()
            record.profile_overrides = overrides.copy()
        # Task calls have independent transient chats. Persisted task state owns progress.
        isolated = ChatAgent(
            client=self.agent._client,
            config=replace(self.agent.config, strategy="sliding_window", memory_storage="in_memory", max_tokens=3072),
            profile_repository=InMemoryProfileRepository({profile_id: stored_profile} if stored_profile else {}),
            user_id=profile_id,
        )
        try:
            result = isolated.ask_with_metadata(
                instruction + "\nReturn one JSON object only, without Markdown fences. "
                "Use the effective user profile language, style, level of detail and preferences for all "
                "human-facing text inside JSON. Explicit preferences in the task goal override the profile. "
                "The outer JSON schema and its field names remain mandatory.",
                memory_target="none",
                task_context=self.context.build(task),
                profile_overrides=overrides,
            )
            if record:
                record.response = result.answer
            self.last_request = result.request
        finally:
            if record and isolated.logs:
                log = isolated.logs[-1]
                record.request = deepcopy({
                    key: value for key, value in log.request.items()
                    if key in {"model", "messages", "max_tokens", "temperature", "extra_body"}
                })
                record.profile_used = any(
                    message.get("role") == "system" and "[USER PROFILE]" in message.get("content", "")
                    for message in record.request.get("messages", [])
                )
                metrics = log.token_metrics or {}
                record.metrics = {
                    key: metrics[key] for key in (
                        "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd",
                        "tokens_source", "request_sent", "context_characters", "context_messages",
                    ) if key in metrics
                }
        content = result.answer.strip()
        if content.startswith(chr(96) * 3) and content.endswith(chr(96) * 3):
            content = "\n".join(content.splitlines()[1:-1])
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as error:
            raise TaskError("Модель вернула некорректный JSON. Прогресс сохранён; повторите действие.") from error
        if not isinstance(raw, dict):
            raise TaskError("Ответ модели должен быть JSON-объектом.")
        return raw

    def plan(self, task: TaskState) -> PlanResult:
        raw = self._request(task, (
            "Plan the task. If essential information is missing, ask ONE concrete question. "
            "Never ask an answered question again, even under another key. Infer reasonable defaults otherwise. "
            'Question schema: {"question":{"key":"stable_snake_case_key","text":"question"}}. '
            "If information is sufficient, create 1–8 sequential steps and explicit readiness criteria. "
            "The steps produce text artifacts, not real-world actions. "
            'Plan schema: {"steps":[{"title":"...","description":"..."}],'
            '"acceptance_criteria":["..."]}. Do not include a question with a plan.'
        ))
        if "question" in raw:
            question = raw["question"]
            if not isinstance(question, dict) or "steps" in raw:
                raise TaskError("Планировщик должен вернуть вопрос или план.")
            return PlanResult([], [], TaskQuestion(
                text_field(question.get("key"), "question.key", 80),
                text_field(question.get("text"), "question.text", 1000),
            ))
        steps = raw.get("steps")
        if not isinstance(steps, list) or not 1 <= len(steps) <= 8:
            raise TaskError("План должен содержать от 1 до 8 шагов.")
        parsed = []
        for index, item in enumerate(steps, 1):
            if not isinstance(item, dict):
                raise TaskError("Шаг плана должен быть объектом.")
            parsed.append(TaskStep(index, text_field(item.get("title"), "title", 200),
                                   text_field(item.get("description"), "description", 1000)))
        criteria = string_list(raw.get("acceptance_criteria"), "acceptance_criteria")
        if not criteria:
            raise TaskError("В плане отсутствуют критерии готовности.")
        return PlanResult(parsed, criteria)

    def execute(self, task: TaskState, step: TaskStep) -> str:
        raw = self._request(task, (
            f"Execute ONLY step {step.id}: {step.title}. Use saved requirements and previous results. "
            "If validation found issues, correct this step's saved result. Do not discard useful content. "
            "Do not claim to send emails, modify files or call tools: this agent generates text artifacts. "
            'Return {"result":"complete result of this step, preferably under 1800 characters"}.'
        ))
        return text_field(raw.get("result"), "result")

    def validate(self, task: TaskState) -> ValidationResult:
        raw = self._request(task, (
            "Check the artifacts against the original goal and EVERY acceptance criterion. "
            "Check missing sections, contradictions and unresolved errors. "
            'For problems return {"issues":[{"step_id":2,"description":"specific problem to fix"}]}. '
            'On success return {"issues":[],"final_result":"the complete final artifact, not a status message"}.'
        ))
        issues = raw.get("issues")
        if not isinstance(issues, list) or len(issues) > 12:
            raise TaskError("Валидатор должен вернуть список issues.")
        descriptions, ids = [], []
        valid_ids = {step.id for step in task.steps}
        for issue in issues:
            if not isinstance(issue, dict) or type(issue.get("step_id")) is not int or issue["step_id"] not in valid_ids:
                raise TaskError("Валидатор указал неизвестный шаг.")
            descriptions.append(text_field(issue.get("description"), "issue", 1000))
            ids.append(issue["step_id"])
        final = None if issues else text_field(raw.get("final_result"), "final_result")
        return ValidationResult(descriptions, list(dict.fromkeys(ids)), final)
