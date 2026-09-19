"""Run the same live scenario against all Day 11 memory-aware strategies.

Usage:
    python -X utf8 compare.py --output data/comparison.json

The script intentionally uses real DeepSeek requests.  It does not fabricate
answers or metrics; the resulting JSON contains the provider metrics returned
by the API plus deterministic recall checks for the scenario.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv

from agent import AgentConfig, AgentError, ChatAgent
from scenario import BRANCH_A, BRANCH_B, COMMON_TURNS, EXPECTED_FACTS, MESSAGES, SCENARIO_NAME


STRATEGIES = ("sliding_window", "facts", "branching", "retrieval")


def _text_contains(text: str, alternatives: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(alternative.lower() in lowered for alternative in alternatives)


def quality_checks(text: str) -> dict[str, Any]:
    checks = {
        name: _text_contains(text, alternatives)
        for name, alternatives in EXPECTED_FACTS.items()
    }
    return {
        "checks": checks,
        "score": round(sum(checks.values()) / len(checks), 3),
    }


def _base_config(strategy: str) -> AgentConfig:
    return AgentConfig(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "512")),
        temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0.2")),
        context_limit_tokens=int(os.getenv("DEEPSEEK_CONTEXT_LIMIT", "32768")),
        strategy=strategy,
        recent_limit=int(os.getenv("DEEPSEEK_RECENT_LIMIT", "6")),
        retrieval_top_k=int(os.getenv("DEEPSEEK_RETRIEVAL_TOP_K", "5")),
        retrieval_min_score=float(os.getenv("DEEPSEEK_RETRIEVAL_MIN_SCORE", "0.1")),
        thinking_enabled=False,
    )


def run_linear(agent: ChatAgent) -> dict[str, Any]:
    answers: list[dict[str, Any]] = []
    started = perf_counter()
    for message in MESSAGES:
        result = agent.ask_with_metadata(message)
        answers.append({"question": message, "answer": result.answer, "metrics": result.token_metrics})
    final_text = answers[-1]["answer"]
    return {
        "answers": answers,
        "quality": quality_checks(final_text),
        "wall_time_seconds": round(perf_counter() - started, 3),
        "analytics": agent.analytics.snapshot(),
        "state": agent.strategy.snapshot(),
    }


def run_branching(agent: ChatAgent) -> dict[str, Any]:
    common_answers: list[dict[str, Any]] = []
    for message in MESSAGES[:COMMON_TURNS]:
        result = agent.ask_with_metadata(message)
        common_answers.append({"question": message, "answer": result.answer, "metrics": result.token_metrics})
    checkpoint = agent.create_checkpoint("common-requirements")
    branch_a = agent.create_branch("economy-mvp", checkpoint["id"])
    branch_b = agent.create_branch("expanded-caregiver", checkpoint["id"])

    branches: dict[str, Any] = {}
    for branch, prompts in ((branch_a, BRANCH_A), (branch_b, BRANCH_B)):
        agent.switch_branch(branch["id"])
        answers = []
        for message in prompts:
            result = agent.ask_with_metadata(message)
            answers.append({"question": message, "answer": result.answer, "metrics": result.token_metrics})
        branch_text = " ".join(item["answer"] for item in answers).lower()
        other_branch_marker = (
            "расширенную версию"
            if branch["name"] == "economy-mvp"
            else "максимально дешёвую"
        )
        branches[branch["name"]] = {
            "branch_id": branch["id"],
            "answers": answers,
            "quality": quality_checks(answers[-1]["answer"]),
            "contains_other_branch_text": other_branch_marker in branch_text,
        }
    return {
        "common_answers": common_answers,
        "checkpoint": checkpoint,
        "branches": branches,
        "analytics": agent.analytics.snapshot(),
        "state": agent.strategy.snapshot(),
    }


def run_comparison(api_key: str) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for strategy in STRATEGIES:
        agent = ChatAgent(api_key=api_key, config=_base_config(strategy))
        try:
            results[strategy] = (
                run_branching(agent)
                if strategy == "branching"
                else run_linear(agent)
            )
        except AgentError as error:
            results[strategy] = {"error": str(error), "analytics": agent.analytics.snapshot()}
    return {"scenario": SCENARIO_NAME, "strategies": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Day 11 memory-aware strategies")
    parser.add_argument("--output", default="data/comparison.json")
    args = parser.parse_args()
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise SystemExit("DEEPSEEK_API_KEY is not configured")
    result = run_comparison(api_key)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Comparison saved to {output}")
    print(json.dumps({key: value.get("analytics") for key, value in result["strategies"].items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
