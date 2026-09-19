import json
import os
from dataclasses import replace
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent import AgentConfig, ChatAgent
from profile import InMemoryProfileRepository
from task_demo import GOAL
from task_llm import LLMTaskBackend
from task_models import TaskStage
from task_repository import JsonTaskRepository
from task_service import TaskService
import web


class QueueCompletions:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def create(self, **request):
        self.calls.append(request)
        answer = next(self.answers)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=answer))],
            usage=SimpleNamespace(prompt_tokens=30, completion_tokens=10, total_tokens=40),
        )


class TaskLLMTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def service(self, answers):
        self.completions = QueueCompletions(answers)
        self.agent = ChatAgent(
            client=SimpleNamespace(chat=SimpleNamespace(completions=self.completions)),
            config=AgentConfig(), profile_repository=InMemoryProfileRepository(),
        )
        return TaskService(JsonTaskRepository(self.temp.name), LLMTaskBackend(self.agent))

    def test_real_adapter_contract_and_task_context_on_every_call(self):
        service = self.service([
            '{"question":{"key":"level","text":"Какой уровень?"}}',
            '{"steps":[{"title":"Текст","description":"Составить вакансию"}],"acceptance_criteria":["Уровень Senior"]}',
            '{"result":"Ищем Senior Android разработчика"}',
            '{"issues":[],"final_result":"Ищем Senior Android разработчика"}',
        ])
        task = service.create_task(GOAL)
        service.continue_task(task.task_id)
        service.answer_question(task.task_id, "Senior")
        service.continue_task(task.task_id)
        result = service.continue_task(task.task_id)
        self.assertEqual(result.stage, TaskStage.DONE)
        self.assertEqual(len(self.completions.calls), 4)
        for call in self.completions.calls:
            system = call["messages"][0]["content"]
            self.assertIn("TASK STATE", system)
            self.assertIn(task.task_id, system)
            self.assertIn("[USER PROFILE]", system)
        self.assertIn("Senior", self.completions.calls[-1]["messages"][0]["content"])
        self.assertEqual(self.agent.history, [])

    def test_malformed_plan_does_not_advance_stage(self):
        for output in ('not json', '[]', '{"steps":[]}', '{"question":"text"}'):
            service = self.service([output])
            task = service.create_task(GOAL)
            task = service.continue_task(task.task_id)
            self.assertEqual(task.stage, TaskStage.PLANNING)
            self.assertTrue(task.last_error)
            self.assertEqual(task.steps, [])

    def test_fact_strategy_does_not_add_unscoped_auxiliary_task_calls(self):
        service = self.service(['{"question":{"key":"level","text":"Какой уровень?"}}'])
        self.agent.config = replace(self.agent.config, strategy="facts")
        task = service.create_task(GOAL)
        service.continue_task(task.task_id)
        self.assertEqual(len(self.completions.calls), 1)
        self.assertIn("TASK STATE", self.completions.calls[0]["messages"][0]["content"])

    def test_validator_cannot_reopen_unknown_step(self):
        service = self.service([
            '{"steps":[{"title":"Текст","description":"Вакансия"}],"acceptance_criteria":["Кратко"]}',
            '{"result":"Текст"}',
            '{"issues":[{"step_id":42,"description":"Ошибка"}]}',
        ])
        task = service.create_task(GOAL)
        for _ in range(3):
            task = service.continue_task(task.task_id)
        self.assertEqual(task.stage, TaskStage.VALIDATION)
        self.assertIn("неизвестный", task.last_error)


class TaskWebTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.environment = patch.dict(os.environ, {
            "TASK_DATA_DIR": self.temp.name, "TASK_BACKEND": "demo",
            "MEMORY_DATA_DIR": self.temp.name + "/memory",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        web.agents.clear()
        web.app.config.update(TESTING=True)
        self.client = web.app.test_client()
        self.addCleanup(web.agents.clear)

    def create(self):
        response = self.client.post("/api/tasks", json={"goal": GOAL})
        self.assertEqual(response.status_code, 201)
        return response.get_json()["task"]["task_id"]

    def act(self, tid, action, body=None):
        response = self.client.post(f"/api/tasks/{tid}/{action}", json=body or {})
        self.assertEqual(response.status_code, 200, response.get_json())
        return response.get_json()["task"]

    def test_web_full_route_pause_restart_fix_done(self):
        tid = self.create()
        task = self.act(tid, "continue")
        self.assertEqual(task["expected_action"]["actor"], "user")
        task = self.act(tid, "continue", {"answer": "Senior"})
        self.assertEqual(task["stage"], "execution")
        self.act(tid, "continue")
        self.act(tid, "pause")
        web.agents.clear()
        self.client = web.app.test_client()  # New browser session, same persisted task.
        task = self.act(tid, "resume")
        self.assertEqual(task["current_step_id"], 3)
        self.act(tid, "continue")
        task = self.act(tid, "continue")
        self.assertEqual(task["stage"], "execution")
        self.assertEqual(task["current_step_id"], 2)
        self.act(tid, "continue")
        task = self.act(tid, "continue")
        self.assertEqual(task["stage"], "done")
        self.assertIn("Опыт:", task["final_result"])

    def test_chat_commands_and_error_status(self):
        created = self.client.post("/api/chat", json={"question": "/new-task " + GOAL})
        self.assertEqual(created.status_code, 200)
        data = created.get_json()
        self.assertTrue(data["task_command"])
        tid = data["task"]["task_id"]
        listed = self.client.post("/api/chat", json={"question": "/list-tasks"})
        self.assertIn(tid, listed.get_json()["answer"])
        self.assertEqual(self.client.get("/api/tasks/task-missing").status_code, 404)
        self.assertEqual(self.client.post("/api/tasks", json={"goal": ""}).status_code, 400)
        self.assertEqual(self.client.post("/api/tasks", json=["invalid"]).status_code, 400)

    def test_chat_clear_reset_and_session_close_preserve_task(self):
        tid = self.create()
        for endpoint in ("/api/chat/clear", "/api/reset", "/api/session/close"):
            self.assertEqual(self.client.post(endpoint).status_code, 200)
            task = self.client.get(f"/api/tasks/{tid}").get_json()["task"]
            self.assertEqual(task["goal"], GOAL)

    def test_panel_and_script_available(self):
        page = self.client.get("/").get_data(as_text=True)
        self.assertIn('id="task-create-form"', page)
        self.assertIn('id="task-transitions"', page)
        script = self.client.get("/static/tasks.js")
        self.assertEqual(script.status_code, 200)
        script.close()


if __name__ == "__main__":
    unittest.main()
