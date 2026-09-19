import tempfile
import unittest
from pathlib import Path

from memory_layers import LayeredMemory, classify_message
from storage import JsonMemoryStore


class MemoryLayersTest(unittest.TestCase):
    def test_dialogue_is_saved_in_short_term_window(self):
        memory = LayeredMemory(short_term_limit=2)

        memory.add_user_message("Привет, как дела?")
        memory.add_assistant_message("Привет!")
        memory.add_user_message("Расскажи подробнее")

        state = memory.state()
        self.assertEqual(state["counts"], {"short_term": 2, "working": 0, "long_term": 0})
        self.assertCountEqual(
            [item["content"] for item in state["layers"]["short_term"]],
            ["Расскажи подробнее", "Привет!"],
        )

    def test_task_step_and_selected_parameter_are_working_memory(self):
        classification = classify_message("Шаг задачи: выбран параметр timeout=30")
        self.assertEqual(classification.layers, ("short_term", "working"))

        memory = LayeredMemory()
        memory.add_user_message("Шаг задачи: выбран параметр timeout=30")
        state = memory.state()
        self.assertEqual(state["counts"]["working"], 1)
        self.assertEqual(state["layers"]["working"][0]["kind"], "task")

    def test_explicit_preference_is_long_term_and_is_upserted(self):
        memory = LayeredMemory()
        memory.add_user_message("Запомни: я предпочитаю краткие ответы")
        memory.add_user_message("Запомни: я предпочитаю краткие ответы")

        state = memory.state()
        self.assertEqual(state["counts"]["long_term"], 1)
        self.assertEqual(state["layers"]["long_term"][0]["kind"], "preference")

    def test_explicit_target_overrides_automatic_classification(self):
        memory = LayeredMemory()

        memory.add_user_message("Обычная заметка для текущей задачи", memory_target="working")
        working = memory.state()["layers"]["working"][0]

        self.assertEqual(working["kind"], "task")
        self.assertEqual(working["source"], "user:explicit")
        self.assertEqual(working["metadata"]["retention_mode"], "working")
        self.assertEqual(memory.state()["last_event"]["type"], "retention_overridden")

        memory.add_user_message("Обычная заметка на будущее", memory_target="long_term")
        durable = memory.state()["layers"]["long_term"][0]
        self.assertEqual(durable["kind"], "knowledge")
        self.assertEqual(durable["metadata"]["retention_mode"], "long_term")

    def test_none_target_keeps_dialogue_without_semantic_memory(self):
        memory = LayeredMemory()

        memory.add_user_message("Не сохраняй это как факт", memory_target="none")

        self.assertEqual(memory.state()["counts"], {
            "short_term": 1, "working": 0, "long_term": 0,
        })
        self.assertEqual(memory.state()["last_event"]["retention_mode"], "none")

    def test_clearing_short_term_does_not_touch_long_term(self):
        memory = LayeredMemory()
        memory.add_user_message("Запомни: предпочитаю русский язык")

        removed = memory.clear_layer("short_term")
        state = memory.state()

        self.assertEqual(removed, 1)
        self.assertEqual(state["counts"]["short_term"], 0)
        self.assertEqual(state["counts"]["long_term"], 1)
        self.assertEqual(
            state["layers"]["long_term"][0]["content"],
            "Запомни: предпочитаю русский язык",
        )

    def test_extraction_uses_working_then_short_term_then_long_term(self):
        memory = LayeredMemory()
        memory.add_user_message("Запомни: я предпочитаю тёмную тему")
        memory.add_user_message("Шаг задачи: выбран параметр тема")

        context = memory.build_context("какую тему выбрать")
        content = context[0]["content"]
        self.assertLess(content.index("[WORKING MEMORY]"), content.index("[SHORT_TERM MEMORY]"))
        self.assertLess(content.index("[SHORT_TERM MEMORY]"), content.index("[LONG_TERM MEMORY]"))

    def test_context_scope_includes_only_its_allowed_supporting_layers(self):
        memory = LayeredMemory()
        memory.add_user_message("Запомни: предпочитаю русский язык")
        memory.add_user_message("Шаг задачи: выбран параметр язык")
        memory.add_assistant_message("Ответ только для short-term")

        working_context = memory.build_context("язык", "working")[0]["content"]
        long_term_context = memory.build_context("язык", "long_term")[0]["content"]
        short_term_context = memory.build_context("язык", "short_term")[0]["content"]

        self.assertIn("[WORKING MEMORY]", working_context)
        self.assertIn("[LONG_TERM MEMORY]", working_context)
        self.assertNotIn("[SHORT_TERM MEMORY]", working_context)
        self.assertIn("[LONG_TERM MEMORY]", long_term_context)
        self.assertNotIn("[WORKING MEMORY]", long_term_context)
        self.assertNotIn("[SHORT_TERM MEMORY]", long_term_context)
        self.assertIn("[SHORT_TERM MEMORY]", short_term_context)
        self.assertIn("[WORKING MEMORY]", short_term_context)
        self.assertIn("[LONG_TERM MEMORY]", short_term_context)

    def test_completing_task_only_clears_working_memory(self):
        memory = LayeredMemory()
        memory.add_user_message("Шаг задачи: выбрать параметры")
        memory.add_user_message("Запомни: предпочитаю JSON")

        removed = memory.complete_task()

        self.assertEqual(removed, 1)
        self.assertEqual(memory.state()["counts"]["working"], 0)
        self.assertEqual(memory.state()["counts"]["long_term"], 1)

    def test_short_term_ttl_is_applied_on_read(self):
        now = [100.0]
        memory = LayeredMemory(short_term_ttl_seconds=10, clock=lambda: now[0])
        memory.add_user_message("Временный диалог")
        self.assertEqual(memory.state()["counts"]["short_term"], 1)

        now[0] = 111.0
        self.assertEqual(memory.state()["counts"]["short_term"], 0)

    def test_json_storage_survives_new_memory_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonMemoryStore(Path(directory) / "memory.json")
            first = LayeredMemory(storage_mode="json_file", persistence=store)
            first.add_user_message("Сообщение только для текущей сессии")
            first.add_user_message("Запомни: предпочитаю русский язык")

            restored = LayeredMemory(storage_mode="json_file", persistence=store)

            self.assertEqual(restored.state()["counts"]["short_term"], 0)
            self.assertEqual(restored.state()["counts"]["long_term"], 1)
            self.assertEqual(
                restored.state()["layers"]["long_term"][0]["content"],
                "Запомни: предпочитаю русский язык",
            )


if __name__ == "__main__":
    unittest.main()
