"""Context-management strategies reused by the Day 11 memory laboratory.

The strategies deliberately do not create summaries.  They only decide which
parts of the original conversation are presented to the model on the next
request.  The agent owns the API calls; this keeps the strategies deterministic
and easy to test.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import re
from typing import Any, Callable, Literal, Protocol


Message = dict[str, str]
StrategyName = Literal["sliding_window", "facts", "branching", "retrieval"]


@dataclass(frozen=True)
class FactUpdateResult:
    facts: dict[str, Any]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: str | None = None


FactUpdater = Callable[[dict[str, Any], str], FactUpdateResult]


class ContextStrategy(Protocol):
    name: StrategyName

    @property
    def history(self) -> list[Message]: ...

    def add_user_message(self, content: str) -> None: ...
    def add_assistant_message(self, content: str) -> None: ...
    def build_context(self) -> list[Message]: ...
    def snapshot(self) -> dict[str, Any]: ...
    def restore(self, state: dict[str, Any]) -> None: ...
    def import_history(self, messages: list[Message]) -> None: ...
    def reset(self) -> None: ...


def _message(role: str, content: str) -> Message:
    if role not in {"user", "assistant"}:
        raise ValueError("role must be user or assistant")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content must be a non-empty string")
    return {"role": role, "content": content.strip()}


def _copy_messages(messages: list[Message]) -> list[Message]:
    return [message.copy() for message in messages]


@dataclass
class SlidingWindowStrategy:
    recent_limit: int = 10
    name: StrategyName = "sliding_window"
    _history: list[Message] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.recent_limit, bool) or self.recent_limit <= 0:
            raise ValueError("recent_limit must be a positive integer")

    @property
    def history(self) -> list[Message]:
        return _copy_messages(self._history)

    def add_user_message(self, content: str) -> None:
        self._history.append(_message("user", content))

    def add_assistant_message(self, content: str) -> None:
        self._history.append(_message("assistant", content))

    def build_context(self) -> list[Message]:
        return _copy_messages(self._history[-self.recent_limit :])

    def snapshot(self) -> dict[str, Any]:
        return {
            "strategy": self.name,
            "recent_limit": self.recent_limit,
            "history": self.history,
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.recent_limit = int(state.get("recent_limit", self.recent_limit))
        self._history = _copy_messages(state.get("history", []))

    def import_history(self, messages: list[Message]) -> None:
        self._history = _copy_messages(messages)

    def reset(self) -> None:
        self._history.clear()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "window_messages": len(self.build_context()),
            "discarded_messages": max(0, len(self._history) - self.recent_limit),
            "facts_count": 0,
            "retrieved_documents": [],
            "active_branch": None,
        }


@dataclass
class FactsStrategy:
    recent_limit: int = 10
    fact_updater: FactUpdater | None = None
    name: StrategyName = "facts"
    _history: list[Message] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    fact_updates: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.recent_limit, bool) or self.recent_limit <= 0:
            raise ValueError("recent_limit must be a positive integer")

    @property
    def history(self) -> list[Message]:
        return _copy_messages(self._history)

    def add_user_message(self, content: str) -> None:
        message = _message("user", content)
        self._history.append(message)
        if self.fact_updater is None:
            result = FactUpdateResult(_heuristic_facts(self.facts, content))
        else:
            try:
                result = self.fact_updater(deepcopy(self.facts), content)
            except Exception as error:  # facts must not break the dialog
                result = FactUpdateResult(deepcopy(self.facts), error=str(error))
        self.facts = _normalize_facts(result.facts)
        self.fact_updates.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "message": content,
                "facts": deepcopy(self.facts),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "error": result.error,
            },
        )

    def add_assistant_message(self, content: str) -> None:
        self._history.append(_message("assistant", content))

    def build_context(self) -> list[Message]:
        facts_json = json.dumps(self.facts, ensure_ascii=False, indent=2)
        facts_message = {
            "role": "system",
            "content": (
                "[STICKY FACTS]\n"
                "These are user-confirmed facts from the conversation. Use them when relevant.\n"
                f"{facts_json}"
            ),
        }
        return [facts_message, *_copy_messages(self._history[-self.recent_limit :])]

    def snapshot(self) -> dict[str, Any]:
        return {
            "strategy": self.name,
            "recent_limit": self.recent_limit,
            "history": self.history,
            "facts": deepcopy(self.facts),
            "fact_updates": deepcopy(self.fact_updates),
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.recent_limit = int(state.get("recent_limit", self.recent_limit))
        self._history = _copy_messages(state.get("history", []))
        self.facts = _normalize_facts(state.get("facts", {}))
        self.fact_updates = deepcopy(state.get("fact_updates", []))

    def import_history(self, messages: list[Message]) -> None:
        self._history = _copy_messages(messages)

    def reset(self) -> None:
        self._history.clear()
        self.facts.clear()
        self.fact_updates.clear()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "window_messages": min(len(self._history), self.recent_limit),
            "discarded_messages": max(0, len(self._history) - self.recent_limit),
            "facts_count": _facts_count(self.facts),
            "retrieved_documents": [],
            "active_branch": None,
        }


def _facts_count(facts: dict[str, Any]) -> int:
    count = 0
    for value in facts.values():
        if isinstance(value, list):
            count += len(value)
        elif value not in (None, "", {}):
            count += 1
    return count


def _normalize_facts(facts: Any) -> dict[str, Any]:
    if not isinstance(facts, dict):
        return {}
    normalized: dict[str, Any] = {}
    allowed = {"goal", "constraints", "preferences", "decisions", "agreements", "open_questions"}
    for key, value in facts.items():
        if key not in allowed:
            continue
        if isinstance(value, list):
            normalized[key] = [str(item).strip() for item in value if str(item).strip()]
        elif isinstance(value, (str, int, float, bool)):
            normalized[key] = value
    return normalized


def _heuristic_facts(previous: dict[str, Any], text: str) -> dict[str, Any]:
    """Small deterministic fallback used when no LLM fact updater is available."""

    facts = deepcopy(previous)
    lowered = text.lower()
    if any(marker in lowered for marker in ("цель", "нужно создать", "хочу сделать", "goal")):
        facts["goal"] = text
    bucket = None
    if any(marker in lowered for marker in ("огранич", "нельзя", "лимит", "constraint")):
        bucket = "constraints"
    elif any(marker in lowered for marker in ("предпочита", "нравится", "хочу", "предпочтение")):
        bucket = "preferences"
    elif any(marker in lowered for marker in ("решили", "решение", "договорились", "decision")):
        bucket = "decisions"
    if bucket:
        values = list(facts.get(bucket, []))
        if text not in values:
            values.append(text)
        facts[bucket] = values
    return _normalize_facts(facts)


@dataclass(frozen=True)
class Checkpoint:
    id: str
    name: str
    messages: list[Message]
    created_at: str


@dataclass
class Branch:
    id: str
    name: str
    parent_checkpoint_id: str | None = None
    messages: list[Message] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


@dataclass
class BranchingStrategy:
    name: StrategyName = "branching"
    branches: dict[str, Branch] = field(default_factory=dict)
    checkpoints: dict[str, Checkpoint] = field(default_factory=dict)
    active_branch_id: str = "main"
    _next_checkpoint: int = 1
    _next_branch: int = 1

    def __post_init__(self) -> None:
        self.branches.setdefault("main", Branch(id="main", name="main"))

    @property
    def history(self) -> list[Message]:
        return _copy_messages(self.branches[self.active_branch_id].messages)

    @property
    def active_branch(self) -> Branch:
        return self.branches[self.active_branch_id]

    def add_user_message(self, content: str) -> None:
        self.active_branch.messages.append(_message("user", content))

    def add_assistant_message(self, content: str) -> None:
        self.active_branch.messages.append(_message("assistant", content))

    def build_context(self) -> list[Message]:
        return self.history

    def create_checkpoint(self, name: str | None = None) -> Checkpoint:
        checkpoint_id = f"checkpoint-{self._next_checkpoint}"
        self._next_checkpoint += 1
        checkpoint = Checkpoint(
            id=checkpoint_id,
            name=name or checkpoint_id,
            messages=self.history,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.checkpoints[checkpoint_id] = checkpoint
        return checkpoint

    def create_branch(self, name: str, checkpoint_id: str) -> Branch:
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"unknown checkpoint: {checkpoint_id}")
        branch_id = f"branch-{self._next_branch}"
        self._next_branch += 1
        branch = Branch(
            id=branch_id,
            name=name.strip() or branch_id,
            parent_checkpoint_id=checkpoint_id,
            messages=_copy_messages(self.checkpoints[checkpoint_id].messages),
        )
        self.branches[branch_id] = branch
        # A newly created branch is the branch the user intends to continue.
        # This prevents follow-up messages from accidentally being appended to
        # the parent/main branch before the user explicitly switches elsewhere.
        self.active_branch_id = branch_id
        return branch

    def switch_branch(self, branch_id: str) -> None:
        if branch_id not in self.branches:
            raise ValueError(f"unknown branch: {branch_id}")
        self.active_branch_id = branch_id

    def snapshot(self) -> dict[str, Any]:
        return {
            "strategy": self.name,
            "active_branch_id": self.active_branch_id,
            "next_checkpoint": self._next_checkpoint,
            "next_branch": self._next_branch,
            "checkpoints": {key: asdict(value) for key, value in self.checkpoints.items()},
            "branches": {key: asdict(value) for key, value in self.branches.items()},
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.active_branch_id = str(state.get("active_branch_id", "main"))
        self._next_checkpoint = int(state.get("next_checkpoint", 1))
        self._next_branch = int(state.get("next_branch", 1))
        self.checkpoints = {
            key: Checkpoint(
                id=value["id"],
                name=value["name"],
                messages=_copy_messages(value.get("messages", [])),
                created_at=value.get("created_at", ""),
            )
            for key, value in state.get("checkpoints", {}).items()
        }
        self.branches = {
            key: Branch(
                id=value["id"],
                name=value["name"],
                parent_checkpoint_id=value.get("parent_checkpoint_id"),
                messages=_copy_messages(value.get("messages", [])),
                created_at=value.get("created_at", ""),
            )
            for key, value in state.get("branches", {}).items()
        }
        self.__post_init__()
        if self.active_branch_id not in self.branches:
            self.active_branch_id = "main"

    def import_history(self, messages: list[Message]) -> None:
        self.branches = {"main": Branch(id="main", name="main", messages=_copy_messages(messages))}
        self.checkpoints.clear()
        self.active_branch_id = "main"

    def reset(self) -> None:
        self.branches = {"main": Branch(id="main", name="main")}
        self.checkpoints.clear()
        self.active_branch_id = "main"
        self._next_checkpoint = 1
        self._next_branch = 1

    def diagnostics(self) -> dict[str, Any]:
        return {
            "window_messages": len(self.history),
            "discarded_messages": 0,
            "facts_count": 0,
            "retrieved_documents": [],
            "active_branch": self.active_branch_id,
            "branch_count": len(self.branches),
            "checkpoint_count": len(self.checkpoints),
        }


def strategy_diagnostics(strategy: ContextStrategy) -> dict[str, Any]:
    diagnostics = getattr(strategy, "diagnostics", None)
    return diagnostics() if diagnostics else {}


def strategy_state(strategy: ContextStrategy) -> dict[str, Any]:
    return deepcopy(strategy.snapshot())


def _validate_import_messages(messages: list[Message]) -> list[Message]:
    return [_message(message["role"], message["content"]) for message in messages]
