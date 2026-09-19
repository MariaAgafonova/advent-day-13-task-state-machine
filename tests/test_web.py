import unittest
from types import SimpleNamespace

import web
from profile import InMemoryProfileRepository, UserProfile


class CompareCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **request):
        self.calls.append(request)
        system = request["messages"][0]["content"]
        answer = (
            "beginner profile answer"
            if "Expertise: Beginner" in system
            else "developer profile answer"
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=answer))],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=5, total_tokens=25),
        )


def fake_client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


class WebTest(unittest.TestCase):
    def setUp(self):
        web.agents.clear()
        web.comparison_agents.clear()
        web.app.config.update(TESTING=True)
        self.client = web.app.test_client()

    def tearDown(self):
        web.agents.clear()
        web.comparison_agents.clear()

    def test_profile_api_creates_loads_and_updates_current_user(self):
        previous_repository = web.profile_repository
        web.profile_repository = InMemoryProfileRepository()
        try:
            created = self.client.post(
                "/api/profile",
                json={"userId": "web-user", "language": "ru", "preferredFormat": "step_by_step"},
            )
            self.assertEqual(created.status_code, 201)
            self.assertEqual(created.get_json()["profile"]["id"], "web-user")

            updated = self.client.patch(
                "/api/profile",
                json={"fields": {"responseLength": "detailed"}},
            )
            self.assertEqual(updated.status_code, 200)
            self.assertEqual(updated.get_json()["profile"]["responseLength"], "detailed")

            loaded = self.client.get("/api/profile")
            self.assertEqual(loaded.status_code, 200)
            self.assertEqual(loaded.get_json()["profile"]["preferredFormat"], "step_by_step")
        finally:
            web.profile_repository = previous_repository
            web.agents.clear()

    def test_profile_selection_is_visible_in_state(self):
        previous_repository = web.profile_repository
        web.profile_repository = InMemoryProfileRepository(
            {"developer": UserProfile(id="developer", language="en")},
        )
        try:
            response = self.client.post("/api/profile/select", json={"user_id": "developer"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["state"]["user_profile"]["user_id"], "developer")
            self.assertEqual(response.get_json()["state"]["user_profile"]["profile"]["language"], "en")
        finally:
            web.profile_repository = previous_repository
            web.agents.clear()

    def test_compare_endpoint_sends_one_question_to_two_isolated_profiles(self):
        previous_repository = web.profile_repository
        web.profile_repository = InMemoryProfileRepository(
            {
                "beginner": UserProfile(
                    id="beginner",
                    language="ru",
                    expertise_level="beginner",
                    response_length="detailed",
                ),
                "developer": UserProfile(
                    id="developer",
                    language="en",
                    expertise_level="senior_developer",
                    response_length="short",
                ),
            },
        )
        completions = CompareCompletions()
        try:
            with self.client:
                self.client.get("/")
                base_agent = web.get_agent()
                base_agent._client = fake_client(completions)
                response = self.client.post(
                    "/api/compare",
                    json={
                        "question": "Объясни, как добавить память в AI-агента",
                        "left_user_id": "beginner",
                        "right_user_id": "developer",
                    },
                )

            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertTrue(payload["same_question"])
            self.assertEqual(payload["question"], "Объясни, как добавить память в AI-агента")
            self.assertEqual(payload["left"]["user_id"], "beginner")
            self.assertEqual(payload["right"]["user_id"], "developer")
            self.assertNotEqual(payload["left"]["answer"], payload["right"]["answer"])
            self.assertEqual(len(completions.calls), 2)
            self.assertEqual(len(payload["comparison_logs"]), 2)
            self.assertTrue(all(item["scope"] == "comparison" for item in payload["comparison_logs"]))

            for request in completions.calls:
                self.assertEqual(request["messages"][-1], {
                    "role": "user",
                    "content": "Объясни, как добавить память в AI-агента",
                })
            self.assertIn("Expertise: Beginner", completions.calls[0]["messages"][0]["content"])
            self.assertIn("Expertise: Senior developer", completions.calls[1]["messages"][0]["content"])
            self.assertEqual(payload["left"]["metrics"]["profile_id"], "beginner")
            self.assertEqual(payload["right"]["metrics"]["profile_id"], "developer")
        finally:
            web.profile_repository = previous_repository
            web.agents.clear()

    def test_compare_endpoint_reuses_private_history_for_follow_up(self):
        previous_repository = web.profile_repository
        web.profile_repository = InMemoryProfileRepository(
            {
                "beginner": UserProfile(id="beginner", expertise_level="beginner"),
                "developer": UserProfile(id="developer", expertise_level="senior_developer"),
            },
        )
        completions = CompareCompletions()
        try:
            with self.client:
                self.client.get("/")
                web.get_agent()._client = fake_client(completions)
                first = self.client.post(
                    "/api/compare",
                    json={
                        "question": "first comparison turn",
                        "left_user_id": "beginner",
                        "right_user_id": "developer",
                    },
                )
                second = self.client.post(
                    "/api/compare",
                    json={
                        "question": "follow up comparison turn",
                        "left_user_id": "beginner",
                        "right_user_id": "developer",
                    },
                )

            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            payload = second.get_json()
            self.assertEqual(len(completions.calls), 4)
            self.assertEqual(len(payload["left"]["turns"]), 2)
            self.assertEqual(len(payload["right"]["turns"]), 2)
            self.assertEqual(payload["left"]["turns"][0]["question"], "first comparison turn")
            self.assertEqual(payload["left"]["turns"][1]["question"], "follow up comparison turn")
            self.assertEqual(payload["right"]["turns"][1]["question"], "follow up comparison turn")
            self.assertIn("first comparison turn", completions.calls[2]["messages"][-2]["content"])
            self.assertEqual(completions.calls[2]["messages"][-1]["content"], "follow up comparison turn")
            state = self.client.get("/api/state")
            self.assertEqual(state.status_code, 200)
            self.assertEqual(len(state.get_json()["comparison_logs"]), 4)

            reset = self.client.post("/api/compare/reset", json={})
            self.assertEqual(reset.status_code, 200)
            self.assertEqual(reset.get_json()["removed_agents"], 2)
        finally:
            web.profile_repository = previous_repository
            web.agents.clear()
            web.comparison_agents.clear()

    def test_page_and_analytics_are_available_without_api_key(self):
        page = self.client.get("/")
        analytics = self.client.get("/api/analytics")

        self.assertEqual(page.status_code, 200)
        page_html = page.get_data(as_text=True)
        self.assertIn("День 13 — состояние задачи", page_html)
        self.assertIn('id="mic-button"', page_html)
        self.assertIn('id="clear-chat"', page_html)
        self.assertIn('id="memory-context"', page_html)
        self.assertIn('id="profile-select"', page_html)
        self.assertIn('id="compare-form"', page_html)
        self.assertIn('id="clear-compare"', page_html)
        self.assertIn('id="compare-left-profile"', page_html)
        self.assertIn('id="compare-right-profile"', page_html)
        self.assertIn('id="compare-results"', page_html)
        self.assertIn("SpeechRecognition", page_html)
        self.assertIn('value="working">Working memory', page_html)
        self.assertIn('value="long_term">Long-term memory', page_html)
        self.assertIn('value="short_term" selected>Только short-term', page_html)
        self.assertNotIn('value="auto"', page_html)
        self.assertNotIn('value="none"', page_html)
        self.assertEqual(analytics.status_code, 200)
        self.assertEqual(analytics.get_json()["total_requests"], 0)

    def test_chat_rejects_unknown_explicit_memory_target(self):
        response = self.client.post(
            "/api/chat",
            json={"question": "Проверка", "memory_target": "unknown"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("memory_target", response.get_json()["error"])

    def test_chat_rejects_unknown_context_layer(self):
        response = self.client.post(
            "/api/chat",
            json={"question": "Проверка", "context_layer": "unknown"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("context_layer", response.get_json()["error"])

    def test_config_switches_strategy(self):
        response = self.client.post(
            "/api/config",
            json={"strategy": "retrieval", "recent_limit": 5, "retrieval_top_k": 3},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["config"]["strategy"], "retrieval")
        self.assertEqual(payload["config"]["recent_limit"], 5)
        self.assertEqual(payload["config"]["retrieval_top_k"], 3)

    def test_branch_endpoints_reject_non_branching_strategy(self):
        response = self.client.post("/api/checkpoints", json={"name": "x"})
        self.assertEqual(response.status_code, 400)

    def test_memory_storage_can_be_switched_from_ui_api(self):
        response = self.client.post(
            "/api/config", json={"memory_storage": "in_memory"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["config"]["memory_storage"], "in_memory")

    def test_memory_endpoint_exposes_layers_and_completes_task(self):
        self.client.post("/api/config", json={"memory_storage": "in_memory"})
        with self.client.session_transaction() as stored_session:
            conversation_id = stored_session["conversation_id"]
        agent = web.agents[conversation_id]
        agent.layered_memory.add_user_message("Шаг задачи: выбран параметр limit=3")
        agent.layered_memory.add_user_message("Запомни: предпочитаю короткие ответы")

        memory = self.client.get("/api/memory")
        self.assertEqual(memory.status_code, 200)
        self.assertEqual(memory.get_json()["memory"]["counts"]["working"], 1)
        self.assertEqual(memory.get_json()["memory"]["counts"]["long_term"], 1)

        completed = self.client.post("/api/memory/complete-task")
        self.assertEqual(completed.status_code, 200)
        self.assertEqual(
            completed.get_json()["state"]["memory"]["counts"]["working"], 0
        )

    def test_clear_layer_is_isolated(self):
        self.client.post("/api/config", json={"memory_storage": "in_memory"})
        with self.client.session_transaction() as stored_session:
            conversation_id = stored_session["conversation_id"]
        agent = web.agents[conversation_id]
        agent.layered_memory.add_user_message("Запомни: люблю русский язык")

        response = self.client.post(
            "/api/memory/clear-layer", json={"layer": "short_term"}
        )

        self.assertEqual(response.status_code, 200)
        counts = response.get_json()["state"]["memory"]["counts"]
        self.assertEqual(counts["short_term"], 0)
        self.assertEqual(counts["long_term"], 1)

    def test_clear_chat_preserves_working_and_long_term_memory(self):
        self.client.post("/api/config", json={"memory_storage": "in_memory"})
        with self.client.session_transaction() as stored_session:
            conversation_id = stored_session["conversation_id"]
        agent = web.agents[conversation_id]
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

        response = self.client.post("/api/chat/clear")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["removed_short_term_entries"], 3)
        self.assertEqual(
            payload["state"]["memory"]["counts"],
            {"short_term": 0, "working": 1, "long_term": 1},
        )
        self.assertEqual(payload["state"]["history"], [])

    def test_page_close_clears_working_but_keeps_short_and_long_term(self):
        self.client.post("/api/config", json={"memory_storage": "in_memory"})
        with self.client.session_transaction() as stored_session:
            conversation_id = stored_session["conversation_id"]
        agent = web.agents[conversation_id]
        agent.layered_memory.add_user_message(
            "Стек для разработки: Python",
            memory_target="long_term",
        )
        agent.layered_memory.add_user_message(
            "Архитектура проекта: Flask API",
            memory_target="working",
        )
        agent.layered_memory.add_user_message(
            "Текущая задача: добавить endpoint",
            memory_target="short_term",
        )

        response = self.client.post("/api/session/close")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["removed_working_entries"], 1)
        counts = agent.layered_memory.state()["counts"]
        self.assertEqual(counts["short_term"], 3)
        self.assertEqual(counts["working"], 0)
        self.assertEqual(counts["long_term"], 1)

    def test_checkpoint_branch_and_saved_branch_list(self):
        self.assertEqual(
            self.client.post("/api/config", json={"strategy": "branching"}).status_code,
            200,
        )
        checkpoint = self.client.post("/api/checkpoints", json={"name": "base"})
        self.assertEqual(checkpoint.status_code, 200)
        checkpoint_id = checkpoint.get_json()["checkpoint"]["id"]

        branch = self.client.post(
            "/api/branches",
            json={"name": "MVP", "checkpoint_id": checkpoint_id},
        )
        self.assertEqual(branch.status_code, 200)
        state_branches = branch.get_json()["state"]["strategy"]["branches"]
        self.assertNotIn("messages", state_branches[0])
        self.assertIn("message_count", state_branches[0])
        branches = self.client.get("/api/branches").get_json()["branches"]
        self.assertEqual([item["name"] for item in branches], ["main", "MVP"])


if __name__ == "__main__":
    unittest.main()
