from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

from task_commands import TaskCommands
from task_demo import DemoTaskBackend, GOAL
from task_models import TaskError, TaskStage
from task_repository import JsonTaskRepository, TaskBusyError, TaskNotFoundError
from task_service import TaskService
import test_task_integration as integration


class TaskDeletionTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repository = JsonTaskRepository(self.temp.name)
        self.backend = DemoTaskBackend()
        self.service = TaskService(self.repository, self.backend)

    def create(self):
        return self.service.create_task(GOAL).task_id

    def test_delete_removes_task_and_logs_after_restart_only_for_selected_id(self):
        tid, kept = self.create(), self.create()
        task = self.service.continue_task(tid)
        self.assertTrue(task.request_logs)
        self.service.delete_task(tid)
        restarted = JsonTaskRepository(self.temp.name)
        with self.assertRaises(TaskNotFoundError):
            restarted.get(tid)
        self.assertEqual([t.task_id for t in restarted.list_tasks()], [kept])
        self.assertFalse(restarted.path_for(tid).exists())

    def test_clear_removes_corrupt_task_but_preserves_unrelated_files(self):
        tids = {self.create(), self.create(), "task-corrupt"}
        self.repository.path_for("task-corrupt").write_text("broken json", encoding="utf-8")
        root = Path(self.temp.name)
        kept = [root / "profile.json", root / ".task-incomplete.tmp", root / "task-invalid_name.json"]
        nested = root / "memory" / "task-nested.json"
        nested.parent.mkdir()
        kept.append(nested)
        for path in kept:
            path.write_text("keep", encoding="utf-8")
        self.assertEqual(set(self.service.clear_tasks()), tids)
        self.assertEqual(self.service.clear_tasks(), [])
        self.assertEqual(self.repository.list_tasks(), [])
        for path in kept:
            self.assertEqual(path.read_text(encoding="utf-8"), "keep")

    def test_unknown_and_invalid_ids_do_not_delete_other_files(self):
        tid = self.create()
        with self.assertRaises(TaskNotFoundError):
            self.service.delete_task("task-missing")
        for invalid in ("../task-other", "task-../other", "task-other.json", "", None):
            with self.subTest(task_id=invalid), self.assertRaises(TaskError):
                self.service.delete_task(invalid)
        self.assertEqual(self.repository.get(tid).goal, GOAL)

    def test_inflight_request_blocks_delete_and_entire_clear_even_when_paused(self):
        tid, other = self.create(), self.create()
        started, release = Event(), Event()
        original = self.backend.plan

        def slow_plan(task):
            started.set()
            if not release.wait(5):
                raise RuntimeError("Test did not release request")
            return original(task)

        with patch.object(self.backend, "plan", side_effect=slow_plan), ThreadPoolExecutor(1) as pool:
            pending = pool.submit(self.service.continue_task, tid)
            try:
                self.assertTrue(started.wait(5))
                self.service.pause_task(tid)
                # Another HTTP session uses a separate service/repository instance.
                concurrent = TaskService(JsonTaskRepository(self.temp.name), DemoTaskBackend())
                with self.assertRaises(TaskBusyError):
                    concurrent.delete_task(tid)
                with self.assertRaises(TaskBusyError):
                    concurrent.clear_tasks()
                self.assertEqual(set(self.repository.list_task_ids()), {tid, other})
            finally:
                release.set()
            self.assertEqual(pending.result(timeout=5).stage, TaskStage.PAUSED)
        self.assertFalse(self.service.is_busy(tid))
        self.service.delete_task(tid)
        self.assertEqual(self.repository.list_task_ids(), [other])

    def test_stale_operation_from_stopped_process_can_be_deleted(self):
        tid = self.create()
        with self.repository.locked(tid):
            self.service._claim(self.repository.get(tid))
        self.service._release(tid)  # Simulate losing the old process's active guard.
        TaskService(JsonTaskRepository(self.temp.name), DemoTaskBackend()).delete_task(tid)
        self.assertEqual(self.repository.list_task_ids(), [])

    def test_filesystem_failure_is_reported_without_claiming_success(self):
        tid = self.create()
        with patch.object(Path, "unlink", side_effect=PermissionError("File is locked")):
            with self.assertRaisesRegex(TaskError, "Удалено 0 из 1"):
                self.service.clear_tasks()
        self.assertEqual(self.repository.get(tid).goal, GOAL)

    def test_commands_require_confirmation_and_reset_active_task(self):
        commands = TaskCommands(self.service)
        tid = commands.handle("/new-task " + GOAL).task.task_id
        other = self.create()
        for command in ("/clear-tasks", "/clear-tasks yes", f"/delete-task {tid}"):
            with self.subTest(command=command), self.assertRaises(TaskError):
                commands.handle(command)
            self.assertEqual(len(self.repository.list_tasks()), 2)
        commands.handle(f"/delete-task {other} --confirm")
        self.assertEqual(commands.active_task_id, tid)
        commands.handle(f"/delete-task {tid} --confirm")
        self.assertIsNone(commands.active_task_id)
        commands.handle("/new-task " + GOAL)
        result = commands.handle("/clear-tasks --confirm")
        self.assertIn("1", result.message)
        self.assertIsNone(commands.active_task_id)
        self.assertIsNone(commands.answer_if_waiting("ordinary chat"))

    def test_other_sessions_deleted_task_does_not_break_chat(self):
        commands = TaskCommands(self.service)
        tid = commands.handle("/new-task " + GOAL).task.task_id
        self.service.continue_task(tid)
        self.service.delete_task(tid)
        self.assertIsNone(commands.answer_if_waiting("Senior"))
        self.assertIsNone(commands.active_task_id)


class TaskDeletionWebTest(unittest.TestCase):
    setUp = integration.TaskWebTest.setUp
    create = integration.TaskWebTest.create
    act = integration.TaskWebTest.act

    def test_delete_routes_require_boolean_confirmation(self):
        tid = self.create()
        for route in ("/api/tasks", f"/api/tasks/{tid}"):
            for body in ({}, {"confirm": False}, {"confirm": "true"}, {"confirm": 1}, []):
                with self.subTest(route=route, body=body):
                    self.assertEqual(self.client.delete(route, json=body).status_code, 400)
            self.assertEqual(self.client.delete(route).status_code, 400)
        self.assertEqual(self.client.get(f"/api/tasks/{tid}").status_code, 200)

    def test_delete_selected_removes_logs_but_preserves_other_task(self):
        tid, kept = self.create(), self.create()
        self.act(tid, "continue")
        response = self.client.delete(f"/api/tasks/{tid}", json={"confirm": True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"deleted_count": 1, "deleted_task_ids": [tid]})
        self.assertEqual(self.client.get(f"/api/tasks/{tid}/logs").status_code, 404)
        self.assertEqual(self.client.get(f"/api/tasks/{kept}").status_code, 200)
        self.assertEqual(self.client.delete(f"/api/tasks/{tid}", json={"confirm": True}).status_code, 404)

    def test_clear_preserves_profiles_and_can_be_repeated(self):
        tids = {self.create(), self.create()}
        before = self.client.get("/api/tasks").get_json()
        response = self.client.delete("/api/tasks", json={"confirm": True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.get_json()["deleted_task_ids"]), tids)
        after = self.client.get("/api/tasks").get_json()
        self.assertEqual(after["tasks"], [])
        self.assertEqual(after["current_profile"], before["current_profile"])
        self.assertEqual(after["available_profiles"], before["available_profiles"])
        self.assertEqual(self.client.delete("/api/tasks", json={"confirm": True}).get_json()["deleted_count"], 0)

    def test_busy_task_returns_conflict_without_partial_clear(self):
        tid, other = self.create(), self.create()
        repository = JsonTaskRepository(self.temp.name)
        service = TaskService(repository, DemoTaskBackend())
        with repository.locked(tid):
            service._claim(repository.get(tid))
        try:
            for route in ("/api/tasks", f"/api/tasks/{tid}"):
                self.assertEqual(self.client.delete(route, json={"confirm": True}).status_code, 409)
            self.assertEqual(set(repository.list_task_ids()), {tid, other})
        finally:
            service._release(tid)

    def test_chat_clear_command_uses_same_confirmation(self):
        self.create()
        response = self.client.post("/api/chat", json={"question": "/clear-tasks"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self.client.get("/api/tasks").get_json()["tasks"]), 1)
        response = self.client.post("/api/chat", json={"question": "/clear-tasks --confirm"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["task_command"])
        self.assertEqual(self.client.get("/api/tasks").get_json()["tasks"], [])

    def test_page_contains_deletion_controls(self):
        page = self.client.get("/").get_data(as_text=True)
        self.assertIn('id="task-delete"', page)
        self.assertIn('id="task-clear"', page)


if __name__ == "__main__":
    unittest.main()
