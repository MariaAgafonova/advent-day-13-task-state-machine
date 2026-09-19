import unittest

from context_strategies import FactsStrategy, SlidingWindowStrategy
from retrieval import LocalTfidfRetriever, RetrievalStrategy


class StrategyTest(unittest.TestCase):
    def test_sliding_window_drops_old_messages_from_context(self):
        strategy = SlidingWindowStrategy(recent_limit=2)
        for index in range(4):
            strategy.add_user_message(f"Message {index}")

        context = strategy.build_context()

        self.assertEqual([item["content"] for item in context], ["Message 2", "Message 3"])
        self.assertEqual(strategy.diagnostics()["discarded_messages"], 2)

    def test_facts_strategy_keeps_structured_memory(self):
        strategy = FactsStrategy(recent_limit=1)
        strategy.add_user_message("Цель проекта — запустить MVP")
        strategy.add_assistant_message("Принято")
        strategy.add_user_message("Ограничение: бюджет 50 000")

        self.assertEqual(strategy.facts["goal"], "Цель проекта — запустить MVP")
        self.assertIn("STICKY FACTS", strategy.build_context()[0]["content"])
        self.assertEqual(len(strategy.build_context()), 2)

    def test_retrieval_returns_relevant_old_document(self):
        strategy = RetrievalStrategy(
            recent_limit=2,
            top_k=2,
            min_score=0.01,
            retriever=LocalTfidfRetriever(),
        )
        strategy.add_user_message("Срок MVP — три месяца")
        strategy.add_assistant_message("Зафиксировал срок три месяца")
        strategy.add_user_message("Обсудим цвет кнопки")
        strategy.add_assistant_message("Предлагаю синий цвет")
        strategy.add_user_message("Напомни срок MVP")

        context = strategy.build_context()

        self.assertTrue(strategy.last_retrieved)
        self.assertTrue(any("три месяца" in item["content"] for item in context))
        self.assertTrue(any(item["role"] == "system" for item in context))
        self.assertEqual(
            [item["content"] for item in context if item["role"] == "user"],
            ["Напомни срок MVP"],
        )


if __name__ == "__main__":
    unittest.main()
