"""Compare one question across the three Day 12 demo profiles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv

from agent import AgentConfig, AgentError, ChatAgent
from profile import (
    InMemoryProfileRepository,
    JsonProfileRepository,
    demo_profiles,
    seed_demo_profiles,
)


QUESTION = "Объясни, как добавить память в AI-агента"
PROFILE_IDS = ("beginner", "developer", "manager")


class OfflineCompletions:
    """A deterministic LLM double for repeatable demos without an API key."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Any:
        self.calls.append(request)
        system = request["messages"][0]["content"]
        if "Expertise: Beginner" in system:
            answer = (
                "Память помогает агенту не терять важные сведения между сообщениями.\n\n"
                "1. Сохраняйте последние сообщения в краткосрочной памяти.\n"
                "2. Данные текущей задачи держите в working memory.\n"
                "3. Устойчивые факты пользователя переносите в long-term memory.\n"
                "4. Перед ответом соберите эти слои и передайте их модели.\n\n"
                "Так агент продолжает диалог и не просит пользователя повторять контекст."
            )
        elif "Expertise: Senior developer" in system:
            answer = (
                "Use three stores and compose them per request:\n\n"
                "```python\n"
                "messages = [system, profile, long_term, working, *short_term, user_query]\n"
                "response = client.chat.completions.create(messages=messages)\n"
                "```\n\n"
                "Persist only durable facts; keep task state and the sliding dialogue window separate."
            )
        else:
            answer = (
                "Память делает ответы последовательными и снижает необходимость повторять вводные.\n\n"
                "- Польза: персональный и непрерывный диалог.\n"
                "- Этапы: определить данные, выбрать сроки хранения, подключить контекст к запросу.\n"
                "- Риски: лишние данные, ошибки изоляции и рост стоимости запросов.\n"
                "- Контроль: ограничения хранения, удаление данных и проверка качества."
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=answer))],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=40, total_tokens=160),
        )


def _client_for(api_key: str | None, offline: bool) -> Any | None:
    if offline:
        return SimpleNamespace(chat=SimpleNamespace(completions=OfflineCompletions()))
    return None if not api_key else None


def run_comparison(
    api_key: str | None = None,
    *,
    offline: bool = False,
    output: str | Path | None = None,
) -> dict[str, Any]:
    if offline:
        repository = InMemoryProfileRepository(demo_profiles())
    else:
        repository = JsonProfileRepository(os.getenv("PROFILE_DATA_DIR", "data/profiles"))
        seed_demo_profiles(repository)

    results: dict[str, Any] = {}
    for profile_id in PROFILE_IDS:
        client = _client_for(api_key, offline)
        agent = ChatAgent(
            api_key=None if offline else api_key,
            client=client,
            config=AgentConfig(
                max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "512")),
                temperature=0.2,
                memory_storage="in_memory",
            ),
            profile_repository=repository,
            user_id=profile_id,
        )
        try:
            result = agent.ask_with_metadata(QUESTION, user_id=profile_id)
        except AgentError as error:
            results[profile_id] = {"error": str(error)}
            continue
        results[profile_id] = {
            "profile": repository.get_or_default(profile_id).to_dict(),
            "answer": result.answer,
            "metrics": result.token_metrics,
            "system_prompt": result.request["messages"][0]["content"],
        }

    report = {
        "question": QUESTION,
        "profiles": results,
        "offline": offline,
        "note": "Offline results use a deterministic LLM double; live mode uses DeepSeek.",
    }
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Day 12 personalization profiles")
    parser.add_argument("--offline", action="store_true", help="use the deterministic local LLM double")
    parser.add_argument("--output", default="data/personalization_comparison.json")
    args = parser.parse_args()
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not args.offline and (not api_key or api_key == "your_api_key_here"):
        raise SystemExit("DEEPSEEK_API_KEY is not configured; use --offline for a deterministic demo")
    report = run_comparison(api_key, offline=args.offline, output=args.output)
    print(f"Comparison saved to {args.output}")
    for profile_id, result in report["profiles"].items():
        print(f"\n[{profile_id}]\n{result.get('answer', result.get('error'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
