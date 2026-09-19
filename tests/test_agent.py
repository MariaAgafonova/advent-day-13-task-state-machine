import unittest
from types import SimpleNamespace

from agent import AgentConfig, ChatAgent
from context_strategies import BranchingStrategy, FactsStrategy


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **request):
        self.calls.append(request)
        is_fact_call = "structured memory" in request["messages"][0]["content"]
        content = '{"goal": "build a test app"}' if is_fact_call else "A concise answer"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=5, total_tokens=25),
        )


def fake_client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


class AgentTest(unittest.TestCase):
    def test_sliding_window_sends_only_recent_messages(self):
        completions = FakeCompletions()
        agent = ChatAgent(
            client=fake_client(completions),
            config=AgentConfig(strategy="sliding_window", recent_limit=2),
        )

        agent.ask("First question")
        agent.ask("Second question")
        agent.ask("Third question")

        request = completions.calls[-1]
        contents = [message["content"] for message in request["messages"]]
        self.assertNotIn("First question", contents)
        self.assertIn("Third question", contents)
        self.assertIn("A concise answer", contents)
        self.assertEqual(agent.logs[-1].token_metrics["window_messages"], 2)
        self.assertEqual(agent.logs[-1].token_metrics["strategy"], "sliding_window")

    def test_facts_strategy_updates_facts_and_counts_auxiliary_call(self):
        completions = FakeCompletions()
        agent = ChatAgent(
            client=fake_client(completions),
            config=AgentConfig(strategy="facts", recent_limit=2),
        )

        agent.ask("The goal is to build a test app")

        self.assertIsInstance(agent.strategy, FactsStrategy)
        self.assertEqual(agent.strategy.facts["goal"], "build a test app")
        request = completions.calls[-1]
        self.assertTrue(any("STICKY FACTS" in message["content"] for message in request["messages"]))
        metrics = agent.logs[-1].token_metrics
        self.assertEqual(metrics["auxiliary_calls"], 1)
        self.assertGreater(metrics["total_tokens_including_auxiliary"], metrics["total_tokens"])
        self.assertEqual(len(agent.logs[-1].auxiliary_requests), 1)
        self.assertEqual(agent.logs[-1].auxiliary_requests[0]["kind"], "facts_extraction")

    def test_branching_keeps_two_histories_independent(self):
        completions = FakeCompletions()
        agent = ChatAgent(
            client=fake_client(completions),
            config=AgentConfig(strategy="branching"),
        )
        agent.ask("Shared requirement")
        checkpoint = agent.create_checkpoint("shared")
        left = agent.create_branch("left", checkpoint["id"])
        right = agent.create_branch("right", checkpoint["id"])

        agent.switch_branch(left["id"])
        agent.ask("Left-only decision")
        agent.switch_branch(right["id"])

        self.assertNotIn("Left-only decision", [item["content"] for item in agent.history])
        self.assertEqual(agent.strategy.active_branch_id, right["id"])

        agent.ask("Right-only decision")
        request_contents = [
            item["content"]
            for item in completions.calls[-1]["messages"]
            if item["role"] != "system"
        ]
        self.assertIn("Shared requirement", request_contents)
        self.assertIn("Right-only decision", request_contents)
        self.assertNotIn("Left-only decision", request_contents)

    def test_switching_strategy_preserves_history(self):
        agent = ChatAgent(
            client=fake_client(FakeCompletions()),
            config=AgentConfig(strategy="sliding_window", recent_limit=2),
        )
        agent.strategy.add_user_message("Keep this message")
        agent.set_strategy("retrieval")

        self.assertEqual(agent.history[-1]["content"], "Keep this message")
        self.assertEqual(agent.strategy_name, "retrieval")

    def test_agent_records_dialogue_in_short_term_memory(self):
        agent = ChatAgent(
            client=fake_client(FakeCompletions()),
            config=AgentConfig(strategy="sliding_window"),
        )

        agent.ask("Обычный вопрос")

        entries = agent.layered_memory.state()["layers"]["short_term"]
        self.assertCountEqual(
            [item["content"] for item in entries],
            ["Обычный вопрос", "A concise answer"],
        )

    def test_agent_records_task_and_preference_in_their_durable_layers(self):
        agent = ChatAgent(
            client=fake_client(FakeCompletions()),
            config=AgentConfig(strategy="sliding_window"),
        )

        agent.ask("Шаг задачи: выбран параметр limit=3")
        agent.ask("Запомни: я предпочитаю краткие ответы")

        state = agent.layered_memory.state()
        self.assertEqual(state["counts"]["working"], 1)
        self.assertEqual(state["counts"]["long_term"], 1)

    def test_agent_uses_explicit_memory_target_and_reports_it(self):
        agent = ChatAgent(
            client=fake_client(FakeCompletions()),
            config=AgentConfig(strategy="sliding_window"),
        )

        agent.ask("Сохрани это как шаг текущей задачи", memory_target="working")

        state = agent.layered_memory.state()
        self.assertEqual(state["counts"]["working"], 1)
        self.assertEqual(agent.logs[-1].token_metrics["memory_target"], "working")

    def test_explicit_memory_target_limits_prompt_to_that_layer(self):
        completions = FakeCompletions()
        agent = ChatAgent(
            client=fake_client(completions),
            config=AgentConfig(strategy="sliding_window"),
        )

        agent.ask("Сообщение только для short-term", memory_target="short_term")
        agent.ask("Шаг только для working", memory_target="working")

        request_text = "\n".join(item["content"] for item in completions.calls[-1]["messages"])
        self.assertIn("[WORKING MEMORY]", request_text)
        self.assertIn("Шаг только для working", request_text)
        self.assertNotIn("Сообщение только для short-term", request_text)
        self.assertNotIn("A concise answer", request_text)
        self.assertEqual(
            agent.logs[-1].token_metrics["memory_context_mode"],
            "working",
        )

    def test_context_layer_can_be_selected_independently_from_save_target(self):
        completions = FakeCompletions()
        agent = ChatAgent(
            client=fake_client(completions),
            config=AgentConfig(strategy="sliding_window"),
        )

        agent.ask(
            "Меня зовут Мария, я из Сербии",
            memory_target="long_term",
            context_layer="long_term",
        )
        agent.ask(
            "Какие данные обо мне сохранены?",
            memory_target="short_term",
            context_layer="long_term",
        )

        request_text = "\n".join(
            item["content"] for item in completions.calls[-1]["messages"]
        )
        self.assertIn("[LONG_TERM MEMORY]", request_text)
        self.assertIn("Меня зовут Мария, я из Сербии", request_text)
        self.assertNotIn("A concise answer", request_text)
        self.assertEqual(
            agent.logs[-1].token_metrics["memory_target"],
            "short_term",
        )
        self.assertEqual(
            agent.logs[-1].token_metrics["memory_context_mode"],
            "long_term",
        )

    def test_short_term_context_includes_working_and_long_term_support(self):
        completions = FakeCompletions()
        agent = ChatAgent(
            client=fake_client(completions),
            config=AgentConfig(strategy="sliding_window"),
        )
        agent.layered_memory.add_user_message(
            "Стек для разработки: Python",
            memory_target="long_term",
        )
        agent.layered_memory.add_user_message(
            "Архитектура проекта: Flask API",
            memory_target="working",
        )

        agent.ask(
            "Текущая задача: добавить endpoint",
            memory_target="short_term",
            context_layer="short_term",
        )

        request_text = "\n".join(
            item["content"] for item in completions.calls[-1]["messages"]
        )
        self.assertIn("[WORKING MEMORY]", request_text)
        self.assertIn("Архитектура проекта: Flask API", request_text)
        self.assertIn("[LONG_TERM MEMORY]", request_text)
        self.assertIn("Стек для разработки: Python", request_text)
        self.assertIn("[SHORT_TERM MEMORY]", request_text)
        self.assertEqual(
            set(agent.logs[-1].token_metrics["memory_layers"]),
            {"working", "short_term", "long_term"},
        )

    def test_next_short_term_chat_uses_working_and_long_term_after_clear(self):
        completions = FakeCompletions()
        agent = ChatAgent(
            client=fake_client(completions),
            config=AgentConfig(strategy="sliding_window"),
        )
        agent.layered_memory.add_user_message(
            "Стек для разработки: Python",
            memory_target="long_term",
        )
        agent.layered_memory.add_user_message(
            "Архитектура проекта: Flask API",
            memory_target="working",
        )
        agent.ask(
            "Старая задача",
            memory_target="short_term",
            context_layer="short_term",
        )

        agent.clear_chat()
        agent.ask(
            "Новая задача",
            memory_target="short_term",
            context_layer="short_term",
        )

        request_text = "\n".join(
            item["content"] for item in completions.calls[-1]["messages"]
        )
        self.assertIn("Архитектура проекта: Flask API", request_text)
        self.assertIn("Стек для разработки: Python", request_text)
        self.assertNotIn("Старая задача", request_text)
        self.assertNotIn("A concise answer", request_text)

    def test_clear_chat_preserves_working_and_long_term_memory(self):
        agent = ChatAgent(
            client=fake_client(FakeCompletions()),
            config=AgentConfig(strategy="sliding_window"),
        )

        agent.layered_memory.add_user_message(
            "Меня зовут Мария",
            memory_target="long_term",
        )
        agent.layered_memory.add_user_message(
            "Шаг задачи: выбрать формат",
            memory_target="working",
        )
        agent.layered_memory.add_user_message(
            "Текущий вопрос",
            memory_target="short_term",
        )
        agent.strategy.add_user_message("Текущий вопрос")

        removed = agent.clear_chat()
        counts = agent.layered_memory.state()["counts"]

        self.assertEqual(removed, 3)
        self.assertEqual(counts, {"short_term": 0, "working": 1, "long_term": 1})
        self.assertEqual(agent.history, [])


if __name__ == "__main__":
    unittest.main()
