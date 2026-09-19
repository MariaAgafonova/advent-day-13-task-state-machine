"""DeepSeek agent with context strategies and explicit memory layers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import logging
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Literal

from openai import OpenAI

from analytics import Analytics
from context_strategies import (
    BranchingStrategy,
    ContextStrategy,
    FactUpdateResult,
    FactsStrategy,
    SlidingWindowStrategy,
    StrategyName,
    strategy_diagnostics,
)
from pricing import cost_for_tokens
from retrieval import RetrievalStrategy
from token_counter import TokenCounter
from memory_layers import (
    LayeredMemory,
    MemoryContextLayer,
    MemoryStorageMode,
    MemoryTarget,
    normalise_context_layer,
    normalise_memory_target,
)
from profile import (
    DEFAULT_USER_ID,
    JsonProfileRepository,
    ProfileRepository,
    UserProfile,
    normalise_overrides,
    parse_request_overrides,
)
from storage import JsonMemoryStore


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CONTEXT_LIMIT = 32768
DEFAULT_RECENT_LIMIT = 10
DEFAULT_RETRIEVAL_TOP_K = 5
DEFAULT_RETRIEVAL_MIN_SCORE = 0.1
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and accurately."
PERSONALIZED_MEMORY_ORDER: tuple[str, ...] = ("long_term", "working", "short_term")
THINKING_TOGGLE_MODELS = {
    "deepseek-flash",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-v4-flash-vision-exp",
}
logger = logging.getLogger(__name__)


class AgentError(RuntimeError):
    """An error raised when the agent cannot complete a request."""


@dataclass(frozen=True)
class AgentConfig:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    context_limit_tokens: int = DEFAULT_CONTEXT_LIMIT
    strategy: StrategyName = "sliding_window"
    recent_limit: int = DEFAULT_RECENT_LIMIT
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K
    retrieval_min_score: float = DEFAULT_RETRIEVAL_MIN_SCORE
    thinking_enabled: bool = False
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    memory_storage: MemoryStorageMode = "in_memory"
    memory_short_term_limit: int = 6
    memory_short_term_ttl_seconds: int = 86_400
    memory_working_limit: int = 30
    memory_long_term_limit: int = 100
    memory_long_term_top_k: int = 8
    user_id: str = DEFAULT_USER_ID


@dataclass(frozen=True)
class AgentResult:
    answer: str
    request: dict[str, Any]
    token_metrics: dict[str, Any]


@dataclass(frozen=True)
class AgentLog:
    request: dict[str, Any]
    elapsed_seconds: float
    token_metrics: dict[str, Any] | None = None
    error: str | None = None
    auxiliary_requests: list[dict[str, Any]] = field(default_factory=list)


class ChatAgent:
    """Stateful chat agent with switchable context and memory policies."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        config: AgentConfig | None = None,
        client: Any | None = None,
        strategy: ContextStrategy | None = None,
        memory_path: str | Path | None = None,
        profile_repository: ProfileRepository | None = None,
        profile_path: str | Path | None = None,
        user_id: str | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self._validate_config(self.config)
        if client is None and api_key:
            client = OpenAI(
                api_key=api_key,
                base_url=self.config.base_url,
                max_retries=0,
            )
        self._client = client
        self.profile_repository = profile_repository or JsonProfileRepository(
            profile_path or Path("data/profiles"),
        )
        selected_user_id = user_id or self.config.user_id
        self.user_id = selected_user_id.strip() if isinstance(selected_user_id, str) and selected_user_id.strip() else DEFAULT_USER_ID
        self.last_profile: UserProfile = UserProfile.defaults(self.user_id)
        self.last_effective_profile: UserProfile = self.last_profile
        self.last_profile_loaded = False
        self.last_profile_overrides: dict[str, Any] = {}
        self.token_counter = TokenCounter(self.config.model)
        self.strategy = strategy or self._new_strategy(self.config)
        self.memory_store = JsonMemoryStore(memory_path or Path("data/memory/default.json"))
        self.layered_memory = LayeredMemory(
            storage_mode=self.config.memory_storage,
            persistence=self.memory_store,
            short_term_limit=self.config.memory_short_term_limit,
            short_term_ttl_seconds=self.config.memory_short_term_ttl_seconds,
            working_limit=self.config.memory_working_limit,
            long_term_limit=self.config.memory_long_term_limit,
            long_term_top_k=self.config.memory_long_term_top_k,
        )
        self.logs: list[AgentLog] = []
        self.analytics = Analytics()
        self._pending_auxiliary: list[dict[str, Any]] = []
        self.last_memory_target: MemoryTarget = "short_term"
        self.last_context_layer: MemoryContextLayer = "short_term"

    def _new_strategy(self, config: AgentConfig) -> ContextStrategy:
        if config.strategy == "sliding_window":
            return SlidingWindowStrategy(recent_limit=config.recent_limit)
        if config.strategy == "facts":
            return FactsStrategy(
                recent_limit=config.recent_limit,
                fact_updater=self._extract_facts,
            )
        if config.strategy == "branching":
            return BranchingStrategy()
        if config.strategy == "retrieval":
            return RetrievalStrategy(
                recent_limit=config.recent_limit,
                top_k=config.retrieval_top_k,
                min_score=config.retrieval_min_score,
            )
        raise ValueError(f"unknown strategy: {config.strategy}")

    @property
    def strategy_name(self) -> StrategyName:
        return self.config.strategy

    @property
    def history(self) -> list[dict[str, str]]:
        return self.strategy.history

    @property
    def memory(self) -> ContextStrategy:
        """Compatibility alias for integrations that used Day 9's memory field."""

        return self.strategy

    def build_context(
        self,
        context_layer: str | None = None,
        *,
        user_id: str | None = None,
        profile_overrides: dict[str, Any] | None = None,
        task_context: str | None = None,
    ) -> list[dict[str, str]]:
        selected_layer = (
            normalise_context_layer(context_layer)
            if context_layer is not None else self.last_context_layer
        )
        current_question = next(
            (item["content"] for item in reversed(self.strategy.history) if item["role"] == "user"),
            None,
        )
        selected_user_id = self._select_user_id(user_id)
        profile = self._load_profile(selected_user_id)
        # Task requests contain internal protocol instructions. Their words
        # (e.g. "steps") must not override the user's actual task preferences.
        detected_overrides = {} if task_context else parse_request_overrides(current_question or "")
        explicit_overrides = normalise_overrides(profile_overrides)
        overrides = {**detected_overrides, **explicit_overrides}
        effective_profile = profile.with_updates(overrides) if overrides else profile
        self.last_effective_profile = effective_profile
        self.last_profile_overrides = overrides
        memory_context = self.layered_memory.build_context(
            current_question,
            selected_layer,
            exclude_content=current_question if selected_layer == "short_term" else None,
            layer_order=PERSONALIZED_MEMORY_ORDER,
        )
        strategy_context = self.strategy.build_context() if selected_layer == "all" else []
        strategy_system = [
            message["content"]
            for message in strategy_context
            if message["role"] == "system"
        ]
        strategy_history = [
            message.copy()
            for message in strategy_context
            if message["role"] != "system"
        ]
        if current_question:
            for index in range(len(strategy_history) - 1, -1, -1):
                if (
                    strategy_history[index]["role"] == "user"
                    and strategy_history[index]["content"] == current_question
                ):
                    strategy_history.pop(index)
                    break
        memory_text = memory_context[0]["content"] if memory_context else ""
        profile_text = (
            "[USER PROFILE]\n"
            f"{effective_profile.prompt_block()}\n"
            "Follow these preferences unless they conflict with system safety "
            "rules or an explicit instruction in the current request."
        )
        override_text = ""
        if overrides:
            override_text = (
                "[CURRENT REQUEST OVERRIDES]\n"
                "The current request has priority over the profile for these settings: "
                + ", ".join(f"{key}={value}" for key, value in overrides.items())
                + "."
            )
        system_sections = [self.config.system_prompt, profile_text]
        if override_text:
            system_sections.append(override_text)
        if memory_text:
            system_sections.append(memory_text)
        system_sections.extend(strategy_system)
        if task_context:
            system_sections.append(task_context)
        if current_question:
            strategy_history.append({"role": "user", "content": current_question})
        return [
            {"role": "system", "content": "\n\n".join(system_sections)},
            *strategy_history,
        ]

    def _completion_request(
        self,
        context_layer: str | None = None,
        *,
        profile_overrides: dict[str, Any] | None = None,
        task_context: str | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": self.build_context(
                context_layer,
                profile_overrides=profile_overrides,
                task_context=task_context,
            ),
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if self.config.model in THINKING_TOGGLE_MODELS:
            request["extra_body"] = {
                "thinking": {
                    "type": "enabled" if self.config.thinking_enabled else "disabled",
                },
            }
        return request

    def ask(
        self,
        question: str,
        memory_target: str = "auto",
        context_layer: str | None = None,
        user_id: str | None = None,
        profile_overrides: dict[str, Any] | None = None,
    ) -> str:
        return self.ask_with_metadata(
            question,
            memory_target=memory_target,
            context_layer=context_layer,
            user_id=user_id,
            profile_overrides=profile_overrides,
        ).answer

    def ask_with_metadata(
        self,
        question: str,
        memory_target: str = "auto",
        context_layer: str | None = None,
        user_id: str | None = None,
        profile_overrides: dict[str, Any] | None = None,
        task_context: str | None = None,
    ) -> AgentResult:
        question = self._validate_question(question)
        selected_memory_target = normalise_memory_target(memory_target)
        selected_context_layer = (
            normalise_context_layer(context_layer)
            if context_layer is not None else
            "all" if selected_memory_target == "auto" else
            "short_term" if selected_memory_target == "none" else selected_memory_target
        )
        if self._client is None:
            raise AgentError(
                "DEEPSEEK_API_KEY is not configured. Set it before sending a real request."
            )

        previous_state = self.strategy.snapshot()
        previous_memory_state = self.layered_memory.snapshot()
        previous_memory_target = self.last_memory_target
        previous_context_layer = self.last_context_layer
        previous_user_id = self.user_id
        previous_profile = self.last_profile
        previous_effective_profile = self.last_effective_profile
        previous_profile_loaded = self.last_profile_loaded
        previous_profile_overrides = self.last_profile_overrides.copy()
        self._pending_auxiliary = []
        self.last_memory_target = selected_memory_target
        self.last_context_layer = selected_context_layer
        self.user_id = self._select_user_id(user_id)
        explicit_profile_overrides = normalise_overrides(profile_overrides)
        try:
            self.layered_memory.add_user_message(question, selected_memory_target)
            self.strategy.add_user_message(question)
        except Exception:
            self.strategy.restore(previous_state)
            self.layered_memory.restore(previous_memory_state, persist=True)
            self.last_memory_target = previous_memory_target
            self.last_context_layer = previous_context_layer
            self.user_id = previous_user_id
            self.last_profile = previous_profile
            self.last_effective_profile = previous_effective_profile
            self.last_profile_loaded = previous_profile_loaded
            self.last_profile_overrides = previous_profile_overrides
            raise

        request = self._completion_request(
            selected_context_layer,
            profile_overrides=explicit_profile_overrides,
            task_context=task_context,
        )
        estimated_prompt_tokens = self.token_counter.count_messages(request["messages"])
        full_context = self.build_context("all", profile_overrides=explicit_profile_overrides, task_context=task_context)
        full_prompt_tokens = self.token_counter.count_messages(full_context)
        full_context_characters = sum(len(message["content"]) for message in full_context)
        # The full context is only a baseline for token-savings metrics. Restore
        # the actual selection so the UI and request log describe what was sent.
        self.layered_memory.build_context(
            question,
            selected_context_layer,
            exclude_content=question if selected_context_layer == "short_term" else None,
            layer_order=PERSONALIZED_MEMORY_ORDER,
        )
        auxiliary = self._auxiliary_metrics()
        if estimated_prompt_tokens + self.config.max_tokens > self.config.context_limit_tokens:
            self.strategy.restore(previous_state)
            self.layered_memory.restore(previous_memory_state, persist=True)
            self.last_memory_target = previous_memory_target
            self.last_context_layer = previous_context_layer
            self.user_id = previous_user_id
            self.last_profile = previous_profile
            self.last_effective_profile = previous_effective_profile
            self.last_profile_loaded = previous_profile_loaded
            self.last_profile_overrides = previous_profile_overrides
            error = (
                f"Context limit exceeded: {estimated_prompt_tokens} + "
                f"{self.config.max_tokens} > {self.config.context_limit_tokens}"
            )
            metrics = self._metrics(
                request,
                prompt_tokens=estimated_prompt_tokens,
                estimated_prompt_tokens=estimated_prompt_tokens,
                completion_tokens=0,
                full_prompt_tokens=full_prompt_tokens,
                full_context_characters=full_context_characters,
                request_sent=False,
                auxiliary=auxiliary,
                memory_target=selected_memory_target,
                context_layer=selected_context_layer,
            )
            self._log_request(metrics, error=error)
            self.analytics.record(metrics, error=error)
            self.logs.append(
                AgentLog(
                    request,
                    0.0,
                    metrics,
                    error,
                    auxiliary_requests=list(self._pending_auxiliary),
                ),
            )
            raise AgentError(error)

        started = perf_counter()
        try:
            response = self._client.chat.completions.create(**request)
            answer, usage = self._parse_response(response)
        except Exception as error:
            self.strategy.restore(previous_state)
            self.layered_memory.restore(previous_memory_state, persist=True)
            self.last_memory_target = previous_memory_target
            self.last_context_layer = previous_context_layer
            self.user_id = previous_user_id
            self.last_profile = previous_profile
            self.last_effective_profile = previous_effective_profile
            self.last_profile_loaded = previous_profile_loaded
            self.last_profile_overrides = previous_profile_overrides
            elapsed = perf_counter() - started
            metrics = self._metrics(
                request,
                prompt_tokens=estimated_prompt_tokens,
                estimated_prompt_tokens=estimated_prompt_tokens,
                completion_tokens=0,
                full_prompt_tokens=full_prompt_tokens,
                full_context_characters=full_context_characters,
                request_sent=False,
                auxiliary=auxiliary,
                memory_target=selected_memory_target,
                context_layer=selected_context_layer,
            )
            self._log_request(metrics, error=str(error))
            self.analytics.record(metrics, elapsed_seconds=elapsed, error=str(error))
            self.logs.append(
                AgentLog(
                    request,
                    elapsed,
                    metrics,
                    str(error),
                    auxiliary_requests=list(self._pending_auxiliary),
                ),
            )
            raise AgentError(str(error)) from error

        elapsed = perf_counter() - started
        self.strategy.add_assistant_message(answer)
        self.layered_memory.add_assistant_message(answer)
        prompt_tokens = int(usage.get("prompt_tokens", estimated_prompt_tokens))
        completion_tokens = int(
            usage.get("completion_tokens", self.token_counter.count_text(answer)),
        )
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
        metrics = self._metrics(
            request,
            prompt_tokens=prompt_tokens,
            estimated_prompt_tokens=estimated_prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            full_prompt_tokens=full_prompt_tokens,
            full_context_characters=full_context_characters,
            request_sent=True,
            auxiliary=auxiliary,
            memory_target=selected_memory_target,
            context_layer=selected_context_layer,
        )
        self._log_request(metrics)
        self.analytics.record(metrics, elapsed_seconds=elapsed)
        log = AgentLog(
            request,
            elapsed,
            metrics,
            auxiliary_requests=list(self._pending_auxiliary),
        )
        self.logs.append(log)
        return AgentResult(answer=answer, request=request, token_metrics=metrics)

    def _metrics(
        self,
        request: dict[str, Any],
        *,
        prompt_tokens: int,
        estimated_prompt_tokens: int,
        completion_tokens: int,
        full_prompt_tokens: int,
        full_context_characters: int,
        request_sent: bool,
        auxiliary: dict[str, Any],
        total_tokens: int | None = None,
        memory_target: MemoryTarget = "auto",
        context_layer: MemoryContextLayer = "all",
    ) -> dict[str, Any]:
        total_tokens = total_tokens if total_tokens is not None else prompt_tokens + completion_tokens
        saved_tokens = max(0, full_prompt_tokens - estimated_prompt_tokens)
        savings_percent = (saved_tokens / full_prompt_tokens * 100) if full_prompt_tokens else 0.0
        main_cost = cost_for_tokens(self.config.model, prompt_tokens, completion_tokens)
        total_cost = main_cost + float(auxiliary["cost_usd"])
        diagnostics = strategy_diagnostics(self.strategy)
        return {
            "strategy": self.strategy_name,
            "memory_mode": self.strategy_name,
            "recent_limit": self.config.recent_limit,
            "context_characters": sum(len(message["content"]) for message in request["messages"]),
            "full_context_characters": full_context_characters,
            "prompt_tokens": prompt_tokens,
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "full_prompt_tokens": full_prompt_tokens,
            "saved_tokens": saved_tokens,
            "savings_percent": round(savings_percent, 2),
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "auxiliary_calls": auxiliary["calls"],
            "auxiliary_prompt_tokens": auxiliary["prompt_tokens"],
            "auxiliary_completion_tokens": auxiliary["completion_tokens"],
            "auxiliary_total_tokens": auxiliary["total_tokens"],
            "auxiliary_cost_usd": round(float(auxiliary["cost_usd"]), 8),
            "total_tokens_including_auxiliary": total_tokens + auxiliary["total_tokens"],
            "cost_usd": round(total_cost, 8),
            "main_cost_usd": round(main_cost, 8),
            "request_sent": request_sent,
            "tokens_source": "api" if request_sent else self.token_counter.info.method,
            "tokenizer": self.token_counter.info.encoding,
            "memory_storage": self.layered_memory.storage_mode,
            "memory_target": memory_target,
            "memory_context_mode": context_layer,
            "context_messages": len(request["messages"]),
            "memory_selected": len(self.layered_memory.state()["last_selection"]),
            "memory_layers": [item["layer"] for item in self.layered_memory.state()["last_selection"]],
            "user_id": self.user_id,
            "profile_id": self.last_profile.id,
            "profile_loaded": self.last_profile_loaded,
            "profile_settings": self.last_effective_profile.to_dict(),
            "profile_overrides": self.last_profile_overrides.copy(),
            **diagnostics,
        }

    def _auxiliary_metrics(self) -> dict[str, Any]:
        prompt_tokens = sum(int(item.get("prompt_tokens", 0)) for item in self._pending_auxiliary)
        completion_tokens = sum(int(item.get("completion_tokens", 0)) for item in self._pending_auxiliary)
        total_tokens = sum(int(item.get("total_tokens", 0)) for item in self._pending_auxiliary)
        cost = sum(float(item.get("cost_usd", 0.0)) for item in self._pending_auxiliary)
        return {
            "calls": len(self._pending_auxiliary),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost,
        }

    @staticmethod
    def _log_request(metrics: dict[str, Any], error: str | None = None) -> None:
        logger.info(
            "LLM request user_id=%s profile_id=%s profile_loaded=%s "
            "overrides=%s memory_layers=%s context_layer=%s request_sent=%s "
            "prompt_tokens=%s completion_tokens=%s error=%s",
            metrics.get("user_id"),
            metrics.get("profile_id"),
            metrics.get("profile_loaded"),
            metrics.get("profile_overrides", {}),
            metrics.get("memory_layers", []),
            metrics.get("memory_context_mode"),
            metrics.get("request_sent"),
            metrics.get("prompt_tokens", 0),
            metrics.get("completion_tokens", 0),
            error or "",
        )

    def _extract_facts(self, current_facts: dict[str, Any], question: str) -> FactUpdateResult:
        if self._client is None:
            return FactUpdateResult(current_facts, error="client is not configured")
        messages = [
            {
                "role": "system",
                "content": (
                    "You maintain structured memory for a conversation. "
                    "Return only valid JSON with these optional keys: goal, constraints, "
                    "preferences, decisions, agreements, open_questions. "
                    "Keep confirmed facts, update only when the new message supports it, "
                    "and do not invent details."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"current_facts": current_facts, "new_user_message": question},
                    ensure_ascii=False,
                ),
            },
        ]
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": min(256, self.config.max_tokens),
            "temperature": 0,
        }
        if self.config.model in THINKING_TOGGLE_MODELS:
            request["extra_body"] = {"thinking": {"type": "disabled"}}
        started = perf_counter()
        try:
            response = self._client.chat.completions.create(**request)
            content, usage = self._parse_response(response)
            prompt_tokens = int(usage.get("prompt_tokens", self.token_counter.count_messages(messages)))
            completion_tokens = int(usage.get("completion_tokens", self.token_counter.count_text(content)))
            total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
            parsed = _parse_json_object(content)
            result = FactUpdateResult(
                facts=parsed,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        except Exception as error:
            result = FactUpdateResult(current_facts, error=str(error))
        auxiliary_entry = {
            "kind": "facts_extraction",
            "request": request,
            "elapsed_seconds": round(perf_counter() - started, 3),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "cost_usd": cost_for_tokens(
                self.config.model,
                result.prompt_tokens,
                result.completion_tokens,
            ),
            "error": result.error,
        }
        self._pending_auxiliary.append(
            auxiliary_entry,
        )
        return result

    def configure_strategy(
        self,
        *,
        strategy: StrategyName | None = None,
        recent_limit: int | None = None,
        retrieval_top_k: int | None = None,
        retrieval_min_score: float | None = None,
        memory_storage: str | None = None,
    ) -> None:
        selected_strategy = strategy or self.config.strategy
        selected_recent = recent_limit if recent_limit is not None else self.config.recent_limit
        selected_top_k = retrieval_top_k if retrieval_top_k is not None else self.config.retrieval_top_k
        selected_min_score = (
            retrieval_min_score
            if retrieval_min_score is not None
            else self.config.retrieval_min_score
        )
        selected_memory_storage = memory_storage or self.config.memory_storage
        new_config = replace(
            self.config,
            strategy=selected_strategy,
            recent_limit=selected_recent,
            retrieval_top_k=selected_top_k,
            retrieval_min_score=selected_min_score,
            memory_storage=selected_memory_storage,
        )
        self._validate_config(new_config)
        old_state = self.strategy.snapshot()
        old_history = self.strategy.history
        new_strategy = self._new_strategy(new_config)
        if selected_strategy == self.strategy_name:
            new_strategy.restore(old_state)
        else:
            new_strategy.import_history(old_history)
            if isinstance(new_strategy, FactsStrategy) and isinstance(old_state.get("facts"), dict):
                new_strategy.facts = old_state["facts"]
        if hasattr(new_strategy, "recent_limit"):
            new_strategy.recent_limit = selected_recent  # type: ignore[attr-defined]
        if isinstance(new_strategy, RetrievalStrategy):
            new_strategy.top_k = selected_top_k
            new_strategy.min_score = selected_min_score
        self.config = new_config
        self.strategy = new_strategy
        self.layered_memory.set_storage_mode(selected_memory_storage)

    def set_strategy(self, strategy: StrategyName) -> None:
        self.configure_strategy(strategy=strategy)

    def create_checkpoint(self, name: str | None = None) -> dict[str, Any]:
        if not isinstance(self.strategy, BranchingStrategy):
            raise AgentError("checkpoints are available only in branching strategy")
        return self.strategy.create_checkpoint(name=name).__dict__

    def create_branch(self, name: str, checkpoint_id: str) -> dict[str, Any]:
        if not isinstance(self.strategy, BranchingStrategy):
            raise AgentError("branches are available only in branching strategy")
        return self.strategy.create_branch(name, checkpoint_id).__dict__

    def switch_branch(self, branch_id: str) -> None:
        if not isinstance(self.strategy, BranchingStrategy):
            raise AgentError("branches are available only in branching strategy")
        self.strategy.switch_branch(branch_id)

    def clear_chat(self) -> int:
        """Clear the dialogue and short-term memory without touching task facts."""
        self.strategy.reset()
        removed = self.layered_memory.clear_layer("short_term")
        self._pending_auxiliary = []
        self.last_memory_target = "short_term"
        self.last_context_layer = "short_term"
        return removed

    def profile_state(self) -> dict[str, Any]:
        self._load_profile(self.user_id)
        self.last_effective_profile = self.last_profile
        self.last_profile_overrides = {}
        return {
            "user_id": self.user_id,
            "profile": self.last_profile.to_dict(),
            "effective_profile": self.last_effective_profile.to_dict(),
            "profile_loaded": self.last_profile_loaded,
            "overrides": self.last_profile_overrides.copy(),
        }

    def close_session(self) -> int:
        """End the page session by clearing task state but keeping durable memory."""
        return self.layered_memory.clear_layer("working")

    def reset(self) -> None:
        self.strategy.reset()
        self.layered_memory.reset()
        self.logs.clear()
        self.analytics.reset()
        self.last_memory_target = "short_term"
        self.last_context_layer = "short_term"

    def _select_user_id(self, user_id: str | None) -> str:
        selected = user_id if user_id is not None else self.user_id
        if not isinstance(selected, str) or not selected.strip():
            raise ValueError("user_id must be a non-empty string")
        return selected.strip()

    def _load_profile(self, user_id: str) -> UserProfile:
        profile = self.profile_repository.get(user_id)
        self.last_profile_loaded = profile is not None
        self.last_profile = profile or UserProfile.defaults(user_id)
        return self.last_profile

    @staticmethod
    def _parse_response(response: Any) -> tuple[str, dict[str, int]]:
        choices = getattr(response, "choices", None)
        if not choices:
            raise AgentError("The LLM returned no answer choices")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        answer = (content or "").strip()
        if not answer:
            raise AgentError("The LLM returned an empty answer")
        usage_object = getattr(response, "usage", None)
        usage: dict[str, int] = {}
        for field_name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(usage_object, field_name, None)
            if value is not None:
                usage[field_name] = int(value)
        return answer, usage

    @staticmethod
    def _validate_question(question: str) -> str:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        return question.strip()

    @staticmethod
    def _validate_strategy(strategy: str) -> None:
        if strategy not in {"sliding_window", "facts", "branching", "retrieval"}:
            raise ValueError(
                "strategy must be sliding_window, facts, branching, or retrieval"
            )

    @classmethod
    def _validate_config(cls, config: AgentConfig) -> None:
        if not config.model.strip():
            raise ValueError("model must not be empty")
        cls._validate_strategy(config.strategy)
        if config.recent_limit <= 0:
            raise ValueError("recent_limit must be positive")
        if config.retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be positive")
        if config.retrieval_min_score < 0:
            raise ValueError("retrieval_min_score must not be negative")
        if config.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if config.context_limit_tokens <= 0:
            raise ValueError("context_limit_tokens must be positive")
        if not 0 <= config.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not isinstance(config.thinking_enabled, bool):
            raise ValueError("thinking_enabled must be a boolean")
        if config.memory_storage not in {"in_memory", "json_file"}:
            raise ValueError("memory_storage must be in_memory or json_file")
        if min(
            config.memory_short_term_limit,
            config.memory_working_limit,
            config.memory_long_term_limit,
            config.memory_long_term_top_k,
        ) <= 0:
            raise ValueError("memory limits must be positive")
        if config.memory_short_term_ttl_seconds < 0:
            raise ValueError("memory_short_term_ttl_seconds must not be negative")


def _parse_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("facts extractor returned a JSON value instead of an object")
    return parsed
