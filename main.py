"""CLI entry point for Day 13 task state machines and personalized chat."""

from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv

from agent import AgentConfig, AgentError, ChatAgent
from context_strategies import BranchingStrategy, FactsStrategy
from retrieval import RetrievalStrategy
from task_commands import TaskCommands
from task_llm import LLMTaskBackend
from task_models import TaskError
from task_repository import JsonTaskRepository
from task_service import TaskService


STRATEGIES = ("sliding_window", "facts", "branching", "retrieval")


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}


def config_from_environment(
    strategy: str | None = None,
    recent_limit: int | None = None,
) -> AgentConfig:
    load_dotenv()
    selected_strategy = strategy or os.getenv("DEEPSEEK_STRATEGY", "sliding_window")
    selected_recent = (
        recent_limit
        if recent_limit is not None
        else int(os.getenv("DEEPSEEK_RECENT_LIMIT", "10"))
    )
    return AgentConfig(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "512")),
        temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0.7")),
        context_limit_tokens=int(os.getenv("DEEPSEEK_CONTEXT_LIMIT", "32768")),
        strategy=selected_strategy,
        recent_limit=selected_recent,
        retrieval_top_k=int(os.getenv("DEEPSEEK_RETRIEVAL_TOP_K", "5")),
        retrieval_min_score=float(os.getenv("DEEPSEEK_RETRIEVAL_MIN_SCORE", "0.1")),
        thinking_enabled=_as_bool(os.getenv("DEEPSEEK_THINKING", "disabled")),
        memory_storage=os.getenv("MEMORY_STORAGE", "json_file"),
        memory_short_term_limit=int(os.getenv("MEMORY_SHORT_TERM_LIMIT", "6")),
        memory_short_term_ttl_seconds=int(os.getenv("MEMORY_SHORT_TERM_TTL_SECONDS", "86400")),
        memory_working_limit=int(os.getenv("MEMORY_WORKING_LIMIT", "30")),
        memory_long_term_limit=int(os.getenv("MEMORY_LONG_TERM_LIMIT", "100")),
        memory_long_term_top_k=int(os.getenv("MEMORY_LONG_TERM_TOP_K", "8")),
        user_id=os.getenv("ASSISTANT_USER_ID", "user_1"),
    )


def _print_strategy_state(agent: ChatAgent) -> None:
    if isinstance(agent.strategy, FactsStrategy):
        print(json.dumps(agent.strategy.facts, ensure_ascii=False, indent=2))
    elif isinstance(agent.strategy, RetrievalStrategy):
        print(json.dumps(agent.strategy.diagnostics(), ensure_ascii=False, indent=2))
    elif isinstance(agent.strategy, BranchingStrategy):
        print(json.dumps(agent.strategy.snapshot(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(agent.strategy.diagnostics(), ensure_ascii=False, indent=2))


def run_chat(config: AgentConfig) -> int:
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        api_key = None
        print("API-ключ не задан. Управление сохранёнными задачами доступно; для LLM заполните .env.")

    agent = ChatAgent(
        api_key=api_key,
        config=config,
        profile_path=os.getenv("PROFILE_DATA_DIR", "data/profiles"),
    )
    print(
        f"Day 13 task state machine · strategy={config.strategy} · "
        f"recent N={config.recent_limit}"
    )
    print(
        "Команды: /strategy <name>, /checkpoint <name>, "
        "/branch create <name> [checkpoint], /branch switch <id>, "
        "/branches, /facts, /retrieved, /memory, /task complete, /analytics, /reset, /exit"
    )
    backend = LLMTaskBackend(agent)
    if os.getenv("TASK_BACKEND", "llm") == "demo":
        from task_demo import DemoTaskBackend
        backend = DemoTaskBackend(profile_id=agent.user_id)
        print("OFFLINE DEMO: используется учебная задача о вакансии Android-разработчика.")
    commands = TaskCommands(TaskService(
        JsonTaskRepository(os.getenv("TASK_DATA_DIR", "data/tasks")), backend,
    ))
    print("Задачи: /new-task <цель>, /task-status <id>, /pause-task <id>, "
          "/resume-task <id>, /continue-task <id> [ответ], /task-logs <id>, /list-tasks, "
          "/delete-task <id> --confirm, /clear-tasks --confirm")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChat finished.")
            return 0
        lowered = question.lower()
        if lowered in {"/exit", "/quit", "exit", "quit"}:
            print("Chat finished.")
            return 0
        try:
            task_result = commands.handle(question) or commands.answer_if_waiting(question)
            if task_result:
                print(task_result.message)
                continue
        except TaskError as error:
            print(f"Ошибка задачи: {error}")
            continue
        if lowered.startswith("/strategy"):
            selected = question.split(maxsplit=1)[1].strip() if len(question.split(maxsplit=1)) > 1 else ""
            try:
                agent.set_strategy(selected)
                print(f"Стратегия: {selected}")
            except (AgentError, ValueError) as error:
                print(f"Ошибка: {error}")
            continue
        if lowered.startswith("/checkpoint"):
            if not isinstance(agent.strategy, BranchingStrategy):
                print("Checkpoint доступен только в branching стратегии.")
                continue
            name = question.partition(" ")[2].strip() or None
            try:
                print(json.dumps(agent.create_checkpoint(name), ensure_ascii=False, indent=2))
            except (AgentError, ValueError) as error:
                print(f"Ошибка: {error}")
            continue
        if lowered.startswith("/branch create"):
            if not isinstance(agent.strategy, BranchingStrategy):
                print("Ветки доступны только в branching стратегии.")
                continue
            parts = question.split()
            if len(parts) < 3:
                print("Использование: /branch create <name> [checkpoint-id]")
                continue
            name = parts[2]
            checkpoint_id = parts[3] if len(parts) > 3 else next(
                reversed(agent.strategy.checkpoints),
                "",
            )
            try:
                print(json.dumps(agent.create_branch(name, checkpoint_id), ensure_ascii=False, indent=2))
            except (AgentError, ValueError) as error:
                print(f"Ошибка: {error}")
            continue
        if lowered.startswith("/branch switch"):
            branch_id = question.partition(" ")[2].partition(" ")[2].strip()
            try:
                agent.switch_branch(branch_id)
                print(f"Активная ветка: {branch_id}")
            except (AgentError, ValueError) as error:
                print(f"Ошибка: {error}")
            continue
        if lowered == "/memory":
            print(json.dumps(agent.layered_memory.state(), ensure_ascii=False, indent=2))
            continue
        if lowered == "/task complete":
            print(f"Working memory очищена: {agent.layered_memory.complete_task()} записей.")
            continue
        if lowered == "/branches" or lowered == "/facts" or lowered == "/retrieved":
            _print_strategy_state(agent)
            continue
        if lowered == "/analytics":
            print(json.dumps(agent.analytics.snapshot(), ensure_ascii=False, indent=2))
            continue
        if lowered == "/reset":
            agent.reset()
            print("Состояние и аналитика очищены.")
            continue
        if not question:
            continue

        try:
            result = agent.ask_with_metadata(question)
        except (AgentError, ValueError) as error:
            print(f"Agent error: {error}")
            continue
        metrics = result.token_metrics
        print(f"Agent: {result.answer}")
        print(
            "[context] "
            f"strategy={metrics['strategy']}, prompt={metrics['prompt_tokens']} tokens, "
            f"total={metrics['total_tokens_including_auxiliary']} tokens, "
            f"cost=${metrics['cost_usd']:.8f}, "
            f"facts={metrics['facts_count']}, "
            f"retrieved={len(metrics['retrieved_documents'])}, "
            f"memory={metrics['memory_storage']} [{metrics['memory_selected']} selected]"
        )
        print(
            "[personalization] "
            f"user_id={metrics['user_id']}, profile_id={metrics['profile_id']}, "
            f"profile_loaded={metrics['profile_loaded']}, "
            f"settings={json.dumps(metrics['profile_settings'], ensure_ascii=False)}, "
            f"overrides={json.dumps(metrics['profile_overrides'], ensure_ascii=False)}"
        )
        print(
            "[memory layers] "
            f"{', '.join(metrics['memory_layers']) or 'none'}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 13: Task State Machine")
    parser.add_argument("--strategy", choices=STRATEGIES, default=None)
    parser.add_argument("--recent", type=int, default=None, help="number of recent raw messages")
    parser.add_argument("--offline", action="store_true", help="deterministic vacancy task demo backend")
    args = parser.parse_args()
    if args.offline:
        os.environ["TASK_BACKEND"] = "demo"
    return run_chat(config_from_environment(args.strategy, args.recent))


if __name__ == "__main__":
    raise SystemExit(main())
