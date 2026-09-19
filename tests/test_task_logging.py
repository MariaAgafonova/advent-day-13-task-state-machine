import json
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent import AgentConfig, ChatAgent
from profile import InMemoryProfileRepository, UserProfile
from task_commands import TaskCommands
from task_demo import DemoTaskBackend, GOAL
from task_llm import LLMTaskBackend
from task_models import TaskStage, TaskState
from task_repository import JsonTaskRepository
from task_service import TaskService
from test_task_integration import QueueCompletions
import test_task_integration as integration


class TaskLoggingTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repository = JsonTaskRepository(self.temp.name)
        self.profiles = InMemoryProfileRepository({
            "developer": UserProfile(
                id="developer", language="en", response_style="technical",
                preferred_format="code_first", response_length="short",
            ),
            "manager": UserProfile(id="manager", language="ru", response_style="business"),
        })

    def service(self, answers, user_id="developer"):
        completions = QueueCompletions(answers)
        agent = ChatAgent(
            client=SimpleNamespace(api_key="secret-never-in-logs", chat=SimpleNamespace(completions=completions)),
            config=AgentConfig(), profile_repository=self.profiles, user_id=user_id,
        )
        return TaskService(self.repository, LLMTaskBackend(agent)), completions

    def test_profile_is_pinned_after_restart_and_chat_profile_change(self):
        service, calls = self.service(['{"question":{"key":"level","text":"Which level?"}}'])
        task = service.create_task("Write an Android vacancy")
        service.continue_task(task.task_id)
        service.pause_task(task.task_id)
        restarted, new_calls = self.service([
            '{"steps":[{"title":"Draft","description":"Write vacancy"}],"acceptance_criteria":["Complete"]}'
        ], "manager")
        restarted.resume_task(task.task_id)
        task = restarted.answer_question(task.task_id, "Senior")
        self.assertEqual(task.profile_id, "developer")
        record = task.request_logs[-1]
        self.assertEqual(record.profile_settings["language"], "en")
        self.assertEqual(record.profile_settings["preferredFormat"], "code_first")
        self.assertEqual(record.profile_overrides, {})
        system = new_calls.calls[0]["messages"][0]["content"]
        self.assertIn("Required response language: English", system)
        self.assertNotIn("Use Russian for human-facing content", new_calls.calls[0]["messages"][-1]["content"])

    def test_goal_overrides_profile_but_internal_protocol_does_not(self):
        service, calls = self.service(['{"question":{"key":"level","text":"Какой уровень?"}}'])
        task = service.create_task("Напиши вакансию на русском подробно")
        task = service.continue_task(task.task_id)
        record = task.request_logs[-1]
        self.assertEqual(record.profile_settings["language"], "ru")
        self.assertEqual(record.profile_settings["responseLength"], "detailed")
        self.assertEqual(record.profile_settings["preferredFormat"], "code_first")
        self.assertEqual(record.profile_overrides, {"language": "ru", "response_length": "detailed"})
        self.assertIn("Required response language: Russian", calls.calls[0]["messages"][0]["content"])

    def test_success_log_survives_new_repository_and_contains_usage(self):
        service, calls = self.service(['{"question":{"key":"level","text":"Which level?"}}'])
        task = service.create_task("Create vacancy")
        task = service.continue_task(task.task_id)
        restored = JsonTaskRepository(self.temp.name).get(task.task_id)
        record = restored.request_logs[0]
        self.assertEqual(record.status, "success")
        self.assertEqual(record.operation, "plan")
        self.assertEqual(record.request, calls.calls[0])
        self.assertEqual(record.metrics["total_tokens"], 40)
        self.assertTrue(record.profile_loaded)
        self.assertTrue(record.profile_used)
        self.assertIn("Which level?", record.response)
        self.assertNotIn("secret-never-in-logs", json.dumps(restored.to_dict()))
        self.assertIsNotNone(record.finished_at)
        # Request logs must never be recursively added to the next prompt.
        self.assertNotIn('"request_logs"', record.request["messages"][0]["content"])

    def test_malformed_response_and_validation_error_have_error_logs(self):
        for answer in ('not json', '{"steps":[]}'):
            service, _ = self.service([answer])
            task = service.create_task("Create vacancy")
            task = service.continue_task(task.task_id)
            record = task.request_logs[-1]
            self.assertEqual(record.status, "error")
            self.assertEqual(record.response, answer)
            self.assertTrue(record.error)
            self.assertEqual(record.metrics["completion_tokens"], 10)
            self.assertEqual(task.stage, TaskStage.PLANNING)

    def test_transport_error_is_logged_with_prompt_and_profile(self):
        service, calls = self.service([])
        task = service.create_task("Create vacancy")
        with patch.object(calls, "create", side_effect=RuntimeError("Network unavailable")):
            task = service.continue_task(task.task_id)
        record = task.request_logs[-1]
        self.assertEqual(record.status, "error")
        self.assertIn("Network unavailable", record.error)
        self.assertTrue(record.profile_used)
        self.assertEqual(record.profile_settings["language"], "en")
        self.assertIsNone(record.response)

    def test_missing_api_key_is_a_logged_error_with_selected_profile(self):
        agent = ChatAgent(profile_repository=self.profiles, user_id="developer")
        service = TaskService(self.repository, LLMTaskBackend(agent))
        task = service.continue_task(service.create_task("Create vacancy").task_id)
        record = task.request_logs[-1]
        self.assertEqual(record.status, "error")
        self.assertEqual(record.profile_id, "developer")
        self.assertFalse(record.profile_used)
        self.assertIsNone(record.request)

    def test_running_log_is_durable_and_late_response_survives_pause(self):
        backend = DemoTaskBackend(profile_id="developer")
        service = TaskService(self.repository, backend)
        task = service.create_task(GOAL)
        original = backend.plan
        def plan(snapshot):
            running = self.repository.get(task.task_id).request_logs[-1]
            self.assertEqual(running.status, "running")
            self.assertTrue(service.is_busy(task.task_id))
            service.pause_task(task.task_id)
            return original(snapshot)
        with patch.object(backend, "plan", side_effect=plan):
            task = service.continue_task(task.task_id)
        self.assertEqual(task.stage, TaskStage.PAUSED)
        self.assertEqual(len(task.request_logs), 1)
        self.assertEqual(task.request_logs[-1].status, "success")
        self.assertFalse(task.request_logs[-1].profile_used)
        self.assertEqual(task.request_logs[-1].metrics, {})

    def test_unfinished_operation_is_marked_interrupted_after_restart(self):
        service = TaskService(self.repository, DemoTaskBackend())
        task = service.create_task(GOAL)
        with self.repository.locked(task.task_id):
            service._claim(task)
        service._release(task.task_id)  # Simulate loss of the old process's in-memory guard.
        restarted = TaskService(JsonTaskRepository(self.temp.name), DemoTaskBackend())
        task = restarted.continue_task(task.task_id)
        self.assertEqual([log.status for log in task.request_logs], ["interrupted", "success"])

    def test_old_task_schema_loads_and_first_request_binds_profile(self):
        service, _ = self.service(['{"question":{"key":"level","text":"Which level?"}}'])
        task = service.create_task("Create vacancy")
        old = task.to_dict()
        old.pop("request_logs")
        old.pop("profile_id")
        self.repository.save(TaskState.from_dict(old))
        restored = service.continue_task(task.task_id)
        self.assertEqual(restored.profile_id, "developer")
        self.assertEqual(len(restored.request_logs), 1)

    def test_cli_logs_command_returns_saved_request_details(self):
        service, _ = self.service(['{"question":{"key":"level","text":"Which level?"}}'])
        task = service.continue_task(service.create_task("Create vacancy").task_id)
        result = TaskCommands(service).handle(f"/task-logs {task.task_id}")
        logs = json.loads(result.message)
        self.assertEqual(logs[0]["profile_id"], "developer")
        self.assertEqual(logs[0]["status"], "success")

    def test_new_profile_settings_are_used_but_old_snapshot_does_not_change(self):
        service, _ = self.service([
            '{"question":{"key":"level","text":"Which level?"}}',
            '{"steps":[{"title":"Draft","description":"Write vacancy"}],"acceptance_criteria":["Complete"]}',
        ])
        task = service.continue_task(service.create_task("Create vacancy").task_id)
        self.profiles.update("developer", {"language": "de"})
        task = service.answer_question(task.task_id, "Senior")
        self.assertEqual(task.request_logs[0].profile_settings["language"], "en")
        self.assertEqual(task.request_logs[1].profile_settings["language"], "de")


class TaskLogWebTest(unittest.TestCase):
    setUp = integration.TaskWebTest.setUp
    create = integration.TaskWebTest.create
    act = integration.TaskWebTest.act
    def test_profile_selection_and_persistent_logs_endpoint(self):
        response = self.client.post("/api/tasks", json={"goal": GOAL, "profile_id": " developer "})
        task = response.get_json()["task"]
        self.assertEqual(task["profile_id"], "developer")
        self.act(task["task_id"], "continue")
        payload = self.client.get(f"/api/tasks/{task['task_id']}/logs").get_json()
        self.assertEqual(payload["profile_id"], "developer")
        self.assertEqual(payload["logs"][0]["mode"], "demo")
        self.assertFalse(payload["logs"][0]["profile_used"])
        self.client.post("/api/reset")
        payload2 = self.client.get(f"/api/tasks/{task['task_id']}/logs").get_json()
        self.assertEqual(payload2, payload)
        listing = self.client.get("/api/tasks").get_json()
        self.assertIn("current_profile", listing)
        self.assertIn("available_profiles", listing)
        self.assertEqual(listing["busy_task_ids"], [])

    def test_invalid_profile_id_is_rejected_and_ui_has_profile_and_logs(self):
        for invalid in ("", [], 42):
            response = self.client.post("/api/tasks", json={"goal": GOAL, "profile_id": invalid})
            self.assertEqual(response.status_code, 400)
        page = self.client.get("/").get_data(as_text=True)
        self.assertIn('id="task-create-profile"', page)
        self.assertIn('id="task-logs"', page)
        self.assertIn('id="task-profile-summary"', page)


if __name__ == "__main__":
    unittest.main()
