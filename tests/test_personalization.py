import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent import AgentConfig, ChatAgent
from profile import InMemoryProfileRepository, JsonProfileRepository, UserProfile


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **request):
        self.calls.append(request)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="personalized answer"))],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=5, total_tokens=25),
        )


def fake_client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


class PersonalizationTest(unittest.TestCase):
    def test_profile_is_loaded_once_in_system_prompt_before_memory_and_question_is_last(self):
        completions = FakeCompletions()
        repository = InMemoryProfileRepository(
            {
                "user_1": UserProfile(
                    id="user_1",
                    name="Maria",
                    language="ru",
                    preferred_format="step_by_step",
                    response_length="medium",
                ),
            },
        )
        agent = ChatAgent(
            client=fake_client(completions),
            config=AgentConfig(strategy="sliding_window"),
            profile_repository=repository,
            user_id="user_1",
        )
        agent.layered_memory.add_user_message("Постоянный стек: Kotlin", memory_target="long_term")
        agent.layered_memory.add_user_message("Шаг задачи: собрать prompt", memory_target="working")

        result = agent.ask_with_metadata("Объясни, как добавить память в AI-агента")
        messages = completions.calls[-1]["messages"]
        system = messages[0]["content"]

        self.assertEqual(sum("[USER PROFILE]" in message["content"] for message in messages), 1)
        self.assertIn("Maria", system)
        self.assertIn("Preferred language: Russian (ru)", system)
        self.assertIn("Required response language: Russian (ru)", system)
        self.assertLess(system.index("[LONG_TERM MEMORY]"), system.index("[WORKING MEMORY]"))
        self.assertLess(system.index("[WORKING MEMORY]"), system.index("[SHORT_TERM MEMORY]"))
        self.assertEqual(messages[-1], {"role": "user", "content": "Объясни, как добавить память в AI-агента"})
        self.assertEqual(result.token_metrics["user_id"], "user_1")
        self.assertEqual(result.token_metrics["profile_id"], "user_1")

    def test_switching_user_id_does_not_leak_profile_settings(self):
        completions = FakeCompletions()
        repository = InMemoryProfileRepository(
            {
                "alice": UserProfile(id="alice", name="Alice", language="en"),
                "bob": UserProfile(id="bob", name="Bob", language="ru"),
            },
        )
        agent = ChatAgent(
            client=fake_client(completions),
            profile_repository=repository,
            user_id="alice",
        )

        agent.ask("First", user_id="alice")
        agent.ask("Second", user_id="bob")
        second_system = completions.calls[-1]["messages"][0]["content"]

        self.assertIn("Bob", second_system)
        self.assertIn("Required response language: Russian (ru)", second_system)
        self.assertNotIn("Alice", second_system)
        self.assertEqual(agent.logs[-1].token_metrics["user_id"], "bob")

    def test_current_request_overrides_profile_and_is_logged(self):
        completions = FakeCompletions()
        repository = InMemoryProfileRepository(
            {"user_1": UserProfile(id="user_1", language="en", response_length="short")},
        )
        agent = ChatAgent(client=fake_client(completions), profile_repository=repository)

        agent.ask(
            "Объясни подробно и пошагово на русском",
            profile_overrides={"responseLength": "detailed", "language": "ru"},
        )
        system = completions.calls[-1]["messages"][0]["content"]
        metrics = agent.logs[-1].token_metrics

        self.assertIn("Preferred language: Russian (ru)", system)
        self.assertIn("Response length: Detailed", system)
        self.assertIn("[CURRENT REQUEST OVERRIDES]", system)
        self.assertEqual(metrics["profile_overrides"]["response_length"], "detailed")
        self.assertEqual(metrics["profile_settings"]["responseLength"], "detailed")

    def test_json_profile_is_available_to_a_new_agent_instance(self):
        completions = FakeCompletions()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles"
            JsonProfileRepository(path).create(
                UserProfile(id="user_1", language="ru", preferred_format="step_by_step"),
            )
            agent = ChatAgent(
                client=fake_client(completions),
                profile_repository=JsonProfileRepository(path),
                user_id="user_1",
            )

            agent.ask("Обычный запрос")
            system = completions.calls[-1]["messages"][0]["content"]

            self.assertIn("Preferred language: Russian (ru)", system)
            self.assertIn("Preferred format: Step by step", system)
            self.assertTrue(agent.logs[-1].token_metrics["profile_loaded"])


if __name__ == "__main__":
    unittest.main()
