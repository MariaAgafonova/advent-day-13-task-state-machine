import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Thread
import unittest
from unittest.mock import patch

from task_commands import TaskCommands
from task_context import TaskContextBuilder
from task_demo import DemoTaskBackend, GOAL
from task_llm import PlanResult, ValidationResult
from task_models import ExpectedAction, StepStatus, TaskError, TaskQuestion, TaskStage
from task_repository import JsonTaskRepository, TaskNotFoundError
from task_service import TaskService
from task_state_machine import ALLOWED_TRANSITIONS, InvalidTaskTransitionError


class TasksTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = JsonTaskRepository(self.temp.name)
        self.backend = DemoTaskBackend()
        self.service = TaskService(self.repo, self.backend)
        self.task = self.service.create_task(GOAL)
        self.tid = self.task.task_id

    def planned(self):
        self.service.plan_task(self.tid)
        return self.service.answer_question(self.tid, "Senior")

    def validation(self):
        self.planned()
        for _ in range(3):
            task = self.service.execute_current_step(self.tid)
        self.assertEqual(task.stage, TaskStage.VALIDATION)
        return task

    def test_successful_route(self):
        self.validation()
        with patch.object(self.backend, "validate", return_value=ValidationResult([], [], "Готовая вакансия")):
            task = self.service.validate_task(self.tid)
        self.assertEqual(task.stage, TaskStage.DONE)
        self.assertEqual([t.to_stage for t in task.transition_history],
                         [TaskStage.EXECUTION, TaskStage.VALIDATION, TaskStage.DONE])
        self.assertEqual(self.repo.get(self.tid).final_result, "Готовая вакансия")

    def test_forbidden_transitions(self):
        for target in (TaskStage.DONE, TaskStage.VALIDATION):
            with self.assertRaises(InvalidTaskTransitionError):
                self.service.machine.transition(self.task, target)
        task = self.planned()
        with self.assertRaises(InvalidTaskTransitionError):
            self.service.machine.transition(task, TaskStage.DONE)
        self.assertEqual(self.repo.get(self.tid).stage, TaskStage.EXECUTION)
        self.assertEqual(ALLOWED_TRANSITIONS[TaskStage.DONE], set())

    def test_pause_planning_restores_waiting_question_after_restart(self):
        asked = self.service.plan_task(self.tid)
        self.service.pause_task(self.tid)
        new_service = TaskService(JsonTaskRepository(self.temp.name), DemoTaskBackend())
        restored = new_service.resume_task(self.tid)
        self.assertEqual(restored.goal, GOAL)
        self.assertEqual(restored.expected_action, asked.expected_action)
        self.assertEqual(restored.stage, TaskStage.PLANNING)
        new_service.continue_task(self.tid)
        self.assertEqual(new_service.backend.plan_calls, 0)
        answered = new_service.answer_question(self.tid, "Senior")
        self.assertEqual(answered.questions[0].answer, "Senior")

    def test_answer_persists_even_when_next_model_call_fails(self):
        self.service.plan_task(self.tid)
        with patch.object(self.backend, "plan", side_effect=RuntimeError("offline")):
            task = self.service.answer_question(self.tid, "Senior")
        self.assertEqual(task.questions[0].answer, "Senior")
        self.service.pause_task(self.tid)
        self.service.resume_task(self.tid)
        task = self.service.continue_task(self.tid)
        self.assertEqual(task.stage, TaskStage.EXECUTION)
        self.assertEqual(len(task.questions), 1)

    def test_pause_execution_keeps_step_and_does_not_repeat_completed(self):
        self.planned()
        task = self.service.execute_current_step(self.tid)
        first = task.steps[0].result
        self.service.start_current_step(self.tid)
        paused = self.service.pause_task(self.tid)
        self.assertEqual(paused.current_step_id, 2)
        self.assertEqual(paused.steps[1].status, StepStatus.IN_PROGRESS)
        backend = DemoTaskBackend()
        restarted = TaskService(JsonTaskRepository(self.temp.name), backend)
        restarted.resume_task(self.tid)
        task = restarted.continue_task(self.tid)
        self.assertEqual(backend.executed, [2])
        self.assertEqual(task.steps[0].result, first)
        self.assertEqual(task.current_step_id, 3)

    def test_pause_validation(self):
        self.validation()
        self.service.pause_task(self.tid)
        restored = TaskService(JsonTaskRepository(self.temp.name), DemoTaskBackend())
        self.assertEqual(restored.resume_task(self.tid).stage, TaskStage.VALIDATION)
        restored.continue_task(self.tid)
        self.assertEqual(restored.backend.executed, [])
        self.assertEqual(restored.backend.validation_calls, 1)

    def test_validation_fixes_only_problem_step(self):
        self.validation()
        task = self.service.validate_task(self.tid)
        self.assertEqual(task.stage, TaskStage.EXECUTION)
        self.assertEqual(task.current_step_id, 2)
        self.assertEqual(task.expected_action.action_type, "fix_validation_issue")
        self.assertEqual([s.status for s in task.steps],
                         [StepStatus.COMPLETED, StepStatus.PENDING, StepStatus.COMPLETED])
        self.service.execute_current_step(self.tid)
        task = self.service.validate_task(self.tid)
        self.assertEqual(task.stage, TaskStage.DONE)
        self.assertEqual(self.backend.executed, [1, 2, 3, 2])
        self.assertEqual(task.validation_issues, [])

    def test_done_is_terminal_and_no_model_calls(self):
        self.validation()
        self.service.validate_task(self.tid)
        self.service.execute_current_step(self.tid)
        done = self.service.validate_task(self.tid)
        with self.assertRaises(InvalidTaskTransitionError):
            self.service.machine.transition(done, TaskStage.EXECUTION)
        for method in (self.service.continue_task, self.service.resume_task):
            self.assertEqual(method(self.tid), done)
        command = TaskCommands(self.service).handle(f"/continue-task {self.tid}")
        self.assertIn("Задача уже завершена", command.message)
        self.assertEqual(self.backend.executed, [1, 2, 3, 2])

    def test_repeated_pause_is_idempotent(self):
        self.planned()
        once = self.service.pause_task(self.tid, "Первая причина")
        twice = self.service.pause_task(self.tid, "Другая причина")
        self.assertEqual(once, twice)
        self.assertEqual(twice.pause_info.reason, "Первая причина")

    def test_resume_cannot_choose_another_stage(self):
        task = self.service.pause_task(self.tid)
        with self.assertRaises(InvalidTaskTransitionError):
            self.service.machine.transition(task, TaskStage.EXECUTION)

    def test_stale_completed_pointer_is_skipped(self):
        self.planned()
        task = self.service.execute_current_step(self.tid)
        task.current_step_id = 1
        self.repo.save(task)
        self.service.execute_current_step(self.tid)
        self.assertEqual(self.backend.executed, [1, 2])

    def test_step_error_and_retry(self):
        self.planned()
        with patch.object(self.backend, "execute", side_effect=RuntimeError("model unavailable")):
            task = self.service.execute_current_step(self.tid)
        self.assertEqual(task.stage, TaskStage.FAILED)
        self.assertEqual(task.steps[0].status, StepStatus.FAILED)
        self.assertEqual(task.expected_action.action_type, "retry_step")
        self.service.pause_task(self.tid)
        self.assertEqual(self.service.resume_task(self.tid).stage, TaskStage.FAILED)
        self.assertEqual(self.service.continue_task(self.tid).current_step_id, 2)

    def test_atomic_write_failure_preserves_original(self):
        path = self.repo.path_for(self.tid)
        before = path.read_bytes()
        self.task.goal = "Новая цель"
        with patch("task_repository.os.replace", side_effect=OSError("disk unavailable")):
            with self.assertRaises(TaskError):
                self.repo.save(self.task)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(Path(self.temp.name).glob("*.tmp")), [])

    def test_unknown_invalid_corrupt_ids(self):
        with self.assertRaises(TaskNotFoundError):
            self.repo.get("task-missing")
        with self.assertRaises(TaskError):
            self.repo.get("../escape")
        self.repo.path_for(self.tid).write_text("{broken", encoding="utf-8")
        with self.assertRaises(TaskError):
            self.repo.get(self.tid)

    def test_duplicate_question_is_rejected_without_losing_answer(self):
        self.service.plan_task(self.tid)
        duplicate = PlanResult([], [], TaskQuestion("developer_level", "Какой уровень?"))
        with patch.object(self.backend, "plan", return_value=duplicate):
            task = self.service.answer_question(self.tid, "Senior")
        self.assertEqual(len(task.questions), 1)
        self.assertEqual(task.questions[0].answer, "Senior")
        self.assertIn("повторила", task.last_error)

    def test_context_contains_progress_and_collected_information(self):
        self.planned()
        task = self.service.execute_current_step(self.tid)
        context = TaskContextBuilder().build(task)
        for expected in ("TASK STATE", "Senior", "Current step: 2", "completed", "Expected action"):
            self.assertIn(expected, context)

    def test_expected_action_actor_is_validated(self):
        with self.assertRaises(TaskError):
            ExpectedAction("unknown", "action", "description")

    def test_pause_while_model_is_running_and_duplicate_continue(self):
        self.planned()
        started, release = Event(), Event()
        result, errors = [], []
        def execute(task, step):
            started.set()
            self.assertTrue(release.wait(5))
            return "Сохранённый результат"
        def worker():
            try:
                result.append(self.service.execute_current_step(self.tid))
            except Exception as error:
                errors.append(error)
        with patch.object(self.backend, "execute", side_effect=execute):
            thread = Thread(target=worker)
            thread.start()
            try:
                self.assertTrue(started.wait(5))
                with self.assertRaises(TaskError):
                    self.service.continue_task(self.tid)
                self.service.pause_task(self.tid)
            finally:
                release.set()
                thread.join(5)
        self.assertEqual(errors, [])
        self.assertEqual(result[0].stage, TaskStage.PAUSED)
        self.assertEqual(result[0].steps[0].result, "Сохранённый результат")
        self.assertEqual(self.service.resume_task(self.tid).current_step_id, 2)

    def test_resume_after_late_last_step(self):
        self.planned()
        self.service.execute_current_step(self.tid)
        self.service.execute_current_step(self.tid)
        original = self.backend.execute
        def pause_inside(task, step):
            self.service.pause_task(self.tid)
            return original(task, step)
        with patch.object(self.backend, "execute", side_effect=pause_inside):
            task = self.service.execute_current_step(self.tid)
        self.assertEqual(task.stage, TaskStage.PAUSED)
        self.assertIsNone(task.current_step_id)
        self.service.resume_task(self.tid)
        self.assertEqual(self.service.continue_task(self.tid).stage, TaskStage.VALIDATION)

    def test_resume_after_validation_finished_during_pause(self):
        self.validation()
        original = self.backend.validate
        def pause_inside(task):
            self.service.pause_task(self.tid)
            return original(task)
        with patch.object(self.backend, "validate", side_effect=pause_inside):
            task = self.service.validate_task(self.tid)
        self.assertEqual(task.stage, TaskStage.PAUSED)
        self.service.resume_task(self.tid)
        task = self.service.continue_task(self.tid)
        self.assertEqual(task.stage, TaskStage.EXECUTION)
        self.assertEqual(task.current_step_id, 2)
        self.assertEqual(self.backend.validation_calls, 1)

    def test_plan_completed_during_pause_is_not_generated_again(self):
        self.service.plan_task(self.tid)
        original = self.backend.plan
        def pause_inside(task):
            self.service.pause_task(self.tid)
            return original(task)
        with patch.object(self.backend, "plan", side_effect=pause_inside):
            task = self.service.answer_question(self.tid, "Senior")
        self.assertEqual(task.stage, TaskStage.PAUSED)
        calls = self.backend.plan_calls
        self.service.resume_task(self.tid)
        task = self.service.continue_task(self.tid)
        self.assertEqual(task.stage, TaskStage.EXECUTION)
        self.assertEqual(self.backend.plan_calls, calls)

    def test_successful_validation_completed_during_pause(self):
        self.validation()
        def pause_inside(task):
            self.service.pause_task(self.tid)
            return ValidationResult([], [], "Проверенный итог")
        with patch.object(self.backend, "validate", side_effect=pause_inside) as validator:
            task = self.service.validate_task(self.tid)
            self.assertEqual(task.stage, TaskStage.PAUSED)
            self.service.resume_task(self.tid)
            task = self.service.continue_task(self.tid)
            self.assertEqual(validator.call_count, 1)
        self.assertEqual(task.stage, TaskStage.DONE)

    def test_resume_command_in_new_cli_process(self):
        self.planned()
        self.service.execute_current_step(self.tid)
        self.service.start_current_step(self.tid)
        self.service.pause_task(self.tid)
        directory = Path(__file__).resolve().parents[1]
        environment = dict(os.environ, TASK_DATA_DIR=self.temp.name,
                           MEMORY_DATA_DIR=self.temp.name + "/memory", MEMORY_STORAGE="in_memory")
        run = subprocess.run(
            [sys.executable, "-B", "-X", "utf8", str(directory / "main.py"), "--offline"],
            input=f"/resume-task {self.tid}\n/exit\n", capture_output=True,
            text=True, encoding="utf-8", timeout=30, cwd=directory, env=environment,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        task = self.repo.get(self.tid)
        self.assertEqual(task.current_step_id, 3)
        self.assertEqual(task.steps[0].status, StepStatus.COMPLETED)
        self.assertEqual(task.steps[1].status, StepStatus.COMPLETED)
        self.assertEqual(len(task.questions), 1)

    def test_demo_restarts_in_another_process(self):
        script = Path(__file__).resolve().parents[1] / "task_demo.py"
        run = subprocess.run(
            [sys.executable, "-B", "-X", "utf8", str(script), "--offline", "--directory", self.temp.name],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads(next((Path(self.temp.name) / "reports").glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(report["executed_after_restart"], [2, 3, 2])
        self.assertEqual(report["plan_calls_after_restart"], 0)
        self.assertEqual(report["state"]["stage"], "done")


if __name__ == "__main__":
    unittest.main()
