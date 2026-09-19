"""Flask UI/API for Day 13 task state machines, personalization and memory."""

from __future__ import annotations

import os
import logging
import secrets
from dataclasses import asdict, replace
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session

from agent import AgentError, ChatAgent
from context_strategies import BranchingStrategy, FactsStrategy
from main import config_from_environment
from profile import (
    DEFAULT_USER_ID,
    JsonProfileRepository,
    ProfileStoreError,
    UserProfile,
    demo_profiles,
    seed_demo_profiles,
)
from retrieval import RetrievalStrategy
from task_commands import TaskCommands
from task_llm import LLMTaskBackend
from task_models import TaskError
from task_repository import JsonTaskRepository, TaskNotFoundError
from task_service import TaskService


load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("WEB_SECRET_KEY", "day-13-local-development-secret")
agents: dict[str, ChatAgent] = {}
comparison_agents: dict[tuple[str, str, str], ChatAgent] = {}
STRATEGIES = {"sliding_window", "facts", "branching", "retrieval"}
profile_repository = JsonProfileRepository(os.getenv("PROFILE_DATA_DIR", "data/profiles"))
seed_demo_profiles(profile_repository)


def get_task_service() -> TaskService:
    backend = LLMTaskBackend(get_agent())
    if os.getenv("TASK_BACKEND", "llm") == "demo":
        from task_demo import DemoTaskBackend
        backend = DemoTaskBackend()
    return TaskService(JsonTaskRepository(os.getenv("TASK_DATA_DIR", "data/tasks")), backend)


@app.errorhandler(TaskNotFoundError)
def task_not_found(error):
    return jsonify(error=str(error)), 404


@app.errorhandler(TaskError)
def task_error(error):
    return jsonify(error=str(error)), 400


@app.get("/api/tasks")
def list_tasks():
    service = get_task_service()
    return jsonify(tasks=[task.to_dict() for task in service.repository.list_tasks()],
                   mode=os.getenv("TASK_BACKEND", "llm"))


@app.get("/api/tasks/<task_id>")
def get_task(task_id):
    return jsonify(task=get_task_service().repository.get(task_id).to_dict())


@app.post("/api/tasks")
def create_task():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise TaskError("Ожидается JSON-объект с goal.")
    task = get_task_service().create_task(data.get("goal"))
    return jsonify(task=task.to_dict()), 201


@app.post("/api/tasks/<task_id>/<action>")
def task_action(task_id, action):
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        raise TaskError("Ожидается JSON-объект.")
    if action not in {"pause", "resume", "continue"}:
        raise TaskError("Действие должно быть pause, resume или continue.")
    service = get_task_service()
    if action == "pause":
        reason = data.get("reason", "Команда пользователя")
        if not isinstance(reason, str) or len(reason) > 1000:
            raise TaskError("Причина паузы должна быть строкой до 1000 символов.")
        task = service.pause_task(task_id, reason)
    elif action == "resume":
        task = service.resume_task(task_id)
        if not service.is_busy(task_id):
            task = service.continue_task(task_id)
    else:
        task = service.continue_task(task_id, data.get("answer"))
    return jsonify(task=task.to_dict())


def get_agent() -> ChatAgent:
    conversation_id = session.get("conversation_id")
    if not conversation_id:
        conversation_id = secrets.token_urlsafe(16)
        session["conversation_id"] = conversation_id
    user_id = session.get("user_id", DEFAULT_USER_ID)
    if conversation_id not in agents:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key == "your_api_key_here":
            api_key = None
        agents[conversation_id] = ChatAgent(
            api_key=api_key,
            config=config_from_environment(),
            memory_path=Path(os.getenv("MEMORY_DATA_DIR", "data/memory")) / f"{conversation_id}.json",
            profile_repository=profile_repository,
            user_id=user_id,
        )
    elif agents[conversation_id].user_id != user_id:
        # A browser session can switch demo users. Keep the profile and the
        # conversation state separate by starting a clean agent session.
        agents[conversation_id].reset()
        agents[conversation_id].user_id = user_id
    return agents[conversation_id]


def config_payload(agent: ChatAgent) -> dict[str, object]:
    return {
        "model": agent.config.model,
        "strategy": agent.config.strategy,
        "recent_limit": agent.config.recent_limit,
        "retrieval_top_k": agent.config.retrieval_top_k,
        "retrieval_min_score": agent.config.retrieval_min_score,
        "context_limit_tokens": agent.config.context_limit_tokens,
        "api_configured": agent._client is not None,
        "memory_storage": agent.config.memory_storage,
        "memory_short_term_limit": agent.config.memory_short_term_limit,
        "memory_short_term_ttl_seconds": agent.config.memory_short_term_ttl_seconds,
        "memory_working_limit": agent.config.memory_working_limit,
        "memory_long_term_limit": agent.config.memory_long_term_limit,
        "memory_long_term_top_k": agent.config.memory_long_term_top_k,
        "memory_context_target": agent.last_context_layer,
        "user_id": agent.user_id,
    }


def strategy_payload(agent: ChatAgent) -> dict[str, object]:
    payload: dict[str, object] = {"name": agent.strategy_name}
    if isinstance(agent.strategy, FactsStrategy):
        payload["facts"] = agent.strategy.facts
        payload["fact_updates"] = agent.strategy.fact_updates
    if isinstance(agent.strategy, RetrievalStrategy):
        payload["retrieved"] = [asdict(item) for item in agent.strategy.last_retrieved]
    if isinstance(agent.strategy, BranchingStrategy):
        def branch_view(branch):
            checkpoint = (
                agent.strategy.checkpoints.get(branch.parent_checkpoint_id)
                if branch.parent_checkpoint_id
                else None
            )
            inherited_count = len(checkpoint.messages) if checkpoint else 0
            branch_messages = branch.messages[inherited_count:]
            responses = [
                {
                    "number": index + 1,
                    "content": message["content"],
                }
                for index, message in enumerate(branch_messages)
                if message["role"] == "assistant"
            ]
            user_messages = [
                message["content"]
                for message in branch_messages
                if message["role"] == "user"
            ]
            return {
                "id": branch.id,
                "name": branch.name,
                "parent_checkpoint_id": branch.parent_checkpoint_id,
                "created_at": branch.created_at,
                "message_count": len(branch.messages),
                "answer_count": len(responses),
                "latest_user_message": user_messages[-1] if user_messages else None,
                "latest_answer": responses[-1]["content"] if responses else None,
                "responses": responses,
            }

        payload["branches"] = [
            branch_view(item)
            for item in agent.strategy.branches.values()
        ]
        payload["checkpoints"] = [
            {
                "id": item.id,
                "name": item.name,
                "created_at": item.created_at,
                "message_count": len(item.messages),
            }
            for item in agent.strategy.checkpoints.values()
        ]
        payload["active_branch"] = agent.strategy.active_branch_id
    return payload


def comparison_turns(agent: ChatAgent) -> list[dict[str, str | None]]:
    """Return the comparison agent's private multi-turn dialogue for the UI."""
    turns: list[dict[str, str | None]] = []
    pending_question: str | None = None
    for message in agent.history:
        if message["role"] == "user":
            pending_question = message["content"]
        elif message["role"] == "assistant" and pending_question is not None:
            turns.append({"question": pending_question, "answer": message["content"]})
            pending_question = None
    if pending_question is not None:
        turns.append({"question": pending_question, "answer": None})
    return turns


def comparison_logs_payload(conversation_id: str | None) -> list[dict[str, object]]:
    """Expose comparison requests in the same log panel as the main chat."""
    if not conversation_id:
        return []
    logs: list[dict[str, object]] = []
    for (agent_conversation_id, side, user_id), agent in comparison_agents.items():
        if agent_conversation_id != conversation_id:
            continue
        for index, log in enumerate(agent.logs):
            record = agent.analytics.records[index] if index < len(agent.analytics.records) else None
            metrics = dict(log.token_metrics or {})
            metrics.update(
                {
                    "comparison": True,
                    "comparison_side": side,
                    "user_id": user_id,
                },
            )
            logs.append(
                {
                    "request_number": index + 1,
                    "timestamp": record.timestamp if record else None,
                    "request": log.request,
                    "elapsed_seconds": round(log.elapsed_seconds, 3),
                    "token_metrics": metrics,
                    "auxiliary_requests": log.auxiliary_requests,
                    "error": log.error,
                    "scope": "comparison",
                    "side": side,
                    "user_id": user_id,
                },
            )
    return sorted(
        logs,
        key=lambda item: (
            str(item.get("timestamp") or ""),
            str(item.get("side") or ""),
            int(item.get("request_number") or 0),
        ),
    )


def state_payload(agent: ChatAgent) -> dict[str, object]:
    logs = []
    for index, log in enumerate(agent.logs):
        record = agent.analytics.records[index] if index < len(agent.analytics.records) else None
        logs.append(
            {
                "request_number": index + 1,
                "timestamp": record.timestamp if record else None,
                "request": log.request,
                "elapsed_seconds": round(log.elapsed_seconds, 3),
                "token_metrics": log.token_metrics,
                "auxiliary_requests": log.auxiliary_requests,
                "error": log.error,
            },
        )
    return {
        "config": config_payload(agent),
        "user_profile": agent.profile_state(),
        "available_profiles": [profile.to_dict() for profile in demo_profiles().values()],
        "history": agent.history,
        "context": agent.build_context(),
        "strategy": strategy_payload(agent),
        "analytics": agent.analytics.snapshot(),
        "logs": logs,
        "comparison_logs": comparison_logs_payload(session.get("conversation_id")),
        "memory": agent.layered_memory.state(),
        "memory_analysis": agent.layered_memory.analysis(),
    }


@app.get("/")
def index():
    return render_template("index.html", state=state_payload(get_agent()))


@app.get("/api/state")
def get_state():
    return jsonify(state_payload(get_agent()))


@app.get("/api/analytics")
def get_analytics():
    return jsonify(get_agent().analytics.snapshot())


@app.get("/api/profile")
def get_profile():
    agent = get_agent()
    profile = profile_repository.get_or_default(agent.user_id)
    return jsonify(
        user_id=agent.user_id,
        profile=profile.to_dict(),
        profile_loaded=profile_repository.get(agent.user_id) is not None,
        available_profiles=[profile.to_dict() for profile in demo_profiles().values()],
    )


@app.post("/api/profile/select")
def select_profile():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id", data.get("userId"))
    if not isinstance(user_id, str) or not user_id.strip():
        return jsonify(error="user_id is required"), 400
    session["user_id"] = user_id.strip()
    agent = get_agent()
    return jsonify(state=state_payload(agent), profile=profile_repository.get_or_default(agent.user_id).to_dict())


@app.post("/api/profile")
def create_profile():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id", data.get("userId", session.get("user_id", DEFAULT_USER_ID)))
    profile_data = data.get("profile", data)
    if not isinstance(user_id, str) or not user_id.strip():
        return jsonify(error="user_id is required"), 400
    if not isinstance(profile_data, dict):
        return jsonify(error="profile must be an object"), 400
    try:
        profile = UserProfile.from_dict({**profile_data, "id": user_id.strip()})
        profile_repository.create(profile)
    except (ProfileStoreError, TypeError, ValueError) as error:
        return jsonify(error=str(error)), 400
    session["user_id"] = profile.id
    return jsonify(profile=profile.to_dict()), 201


@app.patch("/api/profile")
def update_profile():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id", data.get("userId", session.get("user_id", DEFAULT_USER_ID)))
    fields = data.get("fields", data.get("profile", data))
    if not isinstance(user_id, str) or not user_id.strip():
        return jsonify(error="user_id is required"), 400
    if not isinstance(fields, dict):
        return jsonify(error="fields must be an object"), 400
    fields = {key: value for key, value in fields.items() if key not in {"id", "user_id", "userId"}}
    try:
        profile = profile_repository.update(user_id.strip(), fields)
    except (ProfileStoreError, TypeError, ValueError) as error:
        return jsonify(error=str(error)), 400
    session["user_id"] = profile.id
    return jsonify(profile=profile.to_dict())


@app.post("/api/compare")
def compare_profiles():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")
    left_user_id = data.get("left_user_id", data.get("leftUserId", "beginner"))
    right_user_id = data.get("right_user_id", data.get("rightUserId", "developer"))
    if not isinstance(question, str) or not question.strip():
        return jsonify(error="question must be a non-empty string"), 400
    if not all(isinstance(value, str) and value.strip() for value in (left_user_id, right_user_id)):
        return jsonify(error="both profile ids are required"), 400

    base_agent = get_agent()
    compare_config = replace(base_agent.config, memory_storage="in_memory")
    conversation_id = session["conversation_id"]
    results: dict[str, object] = {}
    for side, user_id in (("left", left_user_id.strip()), ("right", right_user_id.strip())):
        comparison_key = (conversation_id, side, user_id)
        compare_agent = comparison_agents.get(comparison_key)
        if compare_agent is None:
            compare_agent = ChatAgent(
                client=base_agent._client,
                config=compare_config,
                profile_repository=profile_repository,
                user_id=user_id,
            )
            comparison_agents[comparison_key] = compare_agent
        try:
            result = compare_agent.ask_with_metadata(
                question,
                memory_target="short_term",
                context_layer="short_term",
                user_id=user_id,
            )
        except (AgentError, ValueError) as error:
            return jsonify(error=f"{side} profile: {error}"), 503 if "API key" in str(error) else 400
        results[side] = {
            "user_id": user_id,
            "profile": compare_agent.last_profile.to_dict(),
            "answer": result.answer,
            "metrics": result.token_metrics,
            "turns": comparison_turns(compare_agent),
        }
    return jsonify(
        question=question.strip(),
        same_question=True,
        left=results["left"],
        right=results["right"],
        comparison_logs=comparison_logs_payload(conversation_id),
    )


@app.post("/api/compare/reset")
def reset_comparison():
    conversation_id = session.get("conversation_id")
    if not conversation_id:
        return jsonify(removed_agents=0)
    keys = [key for key in comparison_agents if key[0] == conversation_id]
    for key in keys:
        comparison_agents.pop(key, None)
    return jsonify(removed_agents=len(keys))


@app.post("/api/config")
def update_config():
    data = request.get_json(silent=True) or {}
    agent = get_agent()
    strategy = data.get("strategy", agent.config.strategy)
    recent_limit = data.get("recent_limit", agent.config.recent_limit)
    top_k = data.get("retrieval_top_k", agent.config.retrieval_top_k)
    min_score = data.get("retrieval_min_score", agent.config.retrieval_min_score)
    memory_storage = data.get("memory_storage", agent.config.memory_storage)
    try:
        if not isinstance(strategy, str) or strategy not in STRATEGIES:
            raise ValueError("strategy is invalid")
        if isinstance(recent_limit, bool) or not isinstance(recent_limit, int) or recent_limit <= 0:
            raise ValueError("recent_limit must be a positive integer")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("retrieval_top_k must be a positive integer")
        if isinstance(min_score, bool) or not isinstance(min_score, (int, float)) or min_score < 0:
            raise ValueError("retrieval_min_score must be non-negative")
        if not isinstance(memory_storage, str) or memory_storage not in {"in_memory", "json_file"}:
            raise ValueError("memory_storage must be in_memory or json_file")
        agent.configure_strategy(
            strategy=strategy,
            recent_limit=recent_limit,
            retrieval_top_k=top_k,
            retrieval_min_score=float(min_score),
            memory_storage=memory_storage,
        )
    except (AgentError, ValueError) as error:
        return jsonify(error=str(error)), 400
    return jsonify(state_payload(agent))


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    try:
        commands = TaskCommands(get_task_service())
        task_result = commands.handle(data.get("question", ""))
        if task_result is not None:
            return jsonify(
                answer=task_result.message, task_command=True,
                task=task_result.task.to_dict() if task_result.task else None,
                metrics={}, state=state_payload(get_agent()),
            )
        result = get_agent().ask_with_metadata(
            data.get("question", ""),
            memory_target=data.get("memory_target", "auto"),
            context_layer=data.get("context_layer"),
            user_id=get_agent().user_id,
            profile_overrides=data.get("profile_overrides"),
        )
    except (AgentError, ValueError) as error:
        return jsonify(error=str(error)), 503 if "API key" in str(error) else 400
    return jsonify(
        answer=result.answer,
        metrics=result.token_metrics,
        state=state_payload(get_agent()),
    )


@app.post("/api/reset")
def reset():
    agent = get_agent()
    agent.reset()
    return jsonify(state_payload(agent))


@app.post("/api/chat/clear")
def clear_chat():
    agent = get_agent()
    removed = agent.clear_chat()
    return jsonify(removed_short_term_entries=removed, state=state_payload(agent))


@app.post("/api/session/close")
def close_session():
    agent = get_agent()
    removed = agent.close_session()
    return jsonify(removed_working_entries=removed)


@app.get("/api/memory")
def get_memory():
    agent = get_agent()
    return jsonify(memory=agent.layered_memory.state(), analysis=agent.layered_memory.analysis())


@app.post("/api/memory/complete-task")
def complete_memory_task():
    agent = get_agent()
    removed = agent.layered_memory.complete_task()
    return jsonify(removed_working_entries=removed, state=state_payload(agent))


@app.post("/api/memory/clear-layer")
def clear_memory_layer():
    data = request.get_json(silent=True) or {}
    layer = data.get("layer", "")
    try:
        removed = get_agent().layered_memory.clear_layer(layer)
    except ValueError as error:
        return jsonify(error=str(error)), 400
    return jsonify(layer=layer, removed_entries=removed, state=state_payload(get_agent()))


@app.post("/api/memory/forget")
def forget_memory_entry():
    data = request.get_json(silent=True) or {}
    entry_id = data.get("entry_id", "")
    if not isinstance(entry_id, str) or not entry_id.strip():
        return jsonify(error="entry_id is required"), 400
    if not get_agent().layered_memory.forget(entry_id):
        return jsonify(error="memory entry not found"), 404
    return jsonify(state=state_payload(get_agent()))


@app.post("/api/checkpoints")
def create_checkpoint():
    data = request.get_json(silent=True) or {}
    try:
        checkpoint = get_agent().create_checkpoint(data.get("name"))
    except (AgentError, ValueError) as error:
        return jsonify(error=str(error)), 400
    return jsonify(checkpoint=checkpoint, state=state_payload(get_agent()))


@app.get("/api/checkpoints")
def list_checkpoints():
    agent = get_agent()
    if not isinstance(agent.strategy, BranchingStrategy):
        return jsonify(checkpoints=[])
    return jsonify(checkpoints=[asdict(item) for item in agent.strategy.checkpoints.values()])


@app.post("/api/branches")
def create_branch():
    data = request.get_json(silent=True) or {}
    try:
        branch = get_agent().create_branch(
            str(data.get("name", "branch")),
            str(data.get("checkpoint_id", "")),
        )
    except (AgentError, ValueError) as error:
        return jsonify(error=str(error)), 400
    return jsonify(branch=branch, state=state_payload(get_agent()))


@app.get("/api/branches")
def list_branches():
    agent = get_agent()
    if not isinstance(agent.strategy, BranchingStrategy):
        return jsonify(branches=[], active_branch=None)
    return jsonify(
        branches=[asdict(item) for item in agent.strategy.branches.values()],
        active_branch=agent.strategy.active_branch_id,
    )


@app.post("/api/branches/switch")
def switch_branch():
    data = request.get_json(silent=True) or {}
    try:
        get_agent().switch_branch(str(data.get("branch_id", "")))
    except (AgentError, ValueError) as error:
        return jsonify(error=str(error)), 400
    return jsonify(state=state_payload(get_agent()))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Day 13 web app")
    parser.add_argument("--offline", action="store_true", help="deterministic vacancy task demo backend")
    args = parser.parse_args()
    if args.offline:
        os.environ["TASK_BACKEND"] = "demo"
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    app.run(
        host="127.0.0.1",
        port=int(os.getenv("WEB_PORT", "5013")),
        debug=False,
    )
