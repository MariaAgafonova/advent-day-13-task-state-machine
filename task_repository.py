"""Atomic JSON persistence, separate from all conversation/memory stores.

One application process owns a directory. RLocks serialize its HTTP/CLI threads.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import tempfile
import time
from threading import RLock

from task_models import TaskError, TaskState


class TaskNotFoundError(TaskError):
    pass


class TaskBusyError(TaskError):
    pass


class TaskRepository(ABC):
    @abstractmethod
    def create(self, task: TaskState) -> None: ...

    @abstractmethod
    def get(self, task_id: str) -> TaskState: ...

    @abstractmethod
    def save(self, task: TaskState) -> None: ...

    @abstractmethod
    def list_tasks(self) -> list[TaskState]: ...

    @abstractmethod
    def list_task_ids(self) -> list[str]: ...

    @abstractmethod
    def delete(self, task_id: str) -> None: ...

    @abstractmethod
    def collection_locked(self): ...

    @abstractmethod
    def locked(self, task_id: str): ...


class JsonTaskRepository(TaskRepository):
    _locks: dict[str, RLock] = {}
    _guard = RLock()

    def __init__(self, directory: str | Path = "data/tasks") -> None:
        self.directory = Path(directory).resolve()

    def path_for(self, task_id: str) -> Path:
        if not isinstance(task_id, str) or not re.fullmatch(r"task-[a-zA-Z0-9-]{1,80}", task_id):
            raise TaskError("Некорректный task_id.")
        return self.directory / f"{task_id}.json"

    @contextmanager
    def collection_locked(self):
        key = str(self.directory) + ":collection"
        with self._guard:
            lock = self._locks.setdefault(key, RLock())
        with lock:
            yield

    @contextmanager
    def locked(self, task_id: str):
        key = str(self.path_for(task_id))
        # Consistent lock order for create, read, claim and deletion.
        # The collection lock is released before every network/model call.
        with self.collection_locked():
            with self._guard:
                lock = self._locks.setdefault(key, RLock())
            with lock:
                yield

    def create(self, task: TaskState) -> None:
        with self.locked(task.task_id):
            if self.path_for(task.task_id).exists():
                raise TaskError(f"Задача {task.task_id} уже существует.")
            self.save(task)

    def get(self, task_id: str) -> TaskState:
        path = self.path_for(task_id)
        with self.locked(task_id):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict) or raw.get("task_id") != task_id:
                    raise TaskError("ID или структура файла не соответствует задаче.")
                return TaskState.from_dict(raw)
            except FileNotFoundError as error:
                raise TaskNotFoundError(f"Задача {task_id} не найдена.") from error
            except (OSError, json.JSONDecodeError) as error:
                raise TaskError(f"Не удалось прочитать задачу {task_id}: {error}") from error

    def save(self, task: TaskState) -> None:
        task.check()
        path = self.path_for(task.task_id)
        temporary = None
        with self.locked(task.task_id):
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=self.directory,
                    prefix=f".{task.task_id}-", suffix=".tmp", delete=False,
                ) as output:
                    temporary = Path(output.name)
                    json.dump(task.to_dict(), output, ensure_ascii=False, indent=2)
                    output.write("\n")
                    output.flush()
                    os.fsync(output.fileno())
                for attempt in range(6):
                    try:
                        os.replace(temporary, path)
                        break
                    except PermissionError:
                        # Windows indexers/antivirus can briefly hold the target
                        # without FILE_SHARE_DELETE. The original stays intact.
                        if os.name != "nt" or attempt == 5:
                            raise
                        time.sleep(0.02 * (2 ** attempt))
            except OSError as error:
                raise TaskError(f"Не удалось сохранить задачу {task.task_id}: {error}") from error
            finally:
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass

    def list_tasks(self) -> list[TaskState]:
        with self.collection_locked():
            return sorted(
                (self.get(task_id) for task_id in self.list_task_ids()),
                key=lambda task: task.created_at, reverse=True,
            )

    def list_task_ids(self) -> list[str]:
        with self.collection_locked():
            return sorted(
                path.stem for path in self.directory.glob("task-*.json")
                if path.is_file() and re.fullmatch(r"task-[a-zA-Z0-9-]{1,80}", path.stem)
            )

    def delete(self, task_id: str) -> None:
        # Delete just the validated task file, including its embedded logs.
        # Do not recursively remove directories or follow paths from the JSON.
        path = self.path_for(task_id)
        with self.locked(task_id):
            try:
                path.unlink()
            except FileNotFoundError as error:
                raise TaskNotFoundError(f"Задача {task_id} не найдена.") from error
            except OSError as error:
                raise TaskError(f"Не удалось удалить задачу {task_id}: {error}") from error
