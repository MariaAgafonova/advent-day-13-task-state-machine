"""Persistent JSON storage for conversational histories."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from typing import Any


DEFAULT_HISTORY_FILE = "data/history.json"
VALID_ROLES = {"system", "user", "assistant"}


class HistoryStoreError(RuntimeError):
    """Raised when a conversation history cannot be read or written."""


class JsonHistoryStore:
    """Store multiple conversation histories in one JSON file."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._lock = RLock()

    def load(self, conversation_id: str) -> list[dict[str, str]]:
        """Load one conversation, returning an empty list when it is new."""

        state = self.load_state(conversation_id)
        messages = state["messages"]
        if messages and messages[0].get("role") == "system":
            return [messages[0], *state["context"], *messages[1:]]
        return state["context"] + messages

    def load_state(self, conversation_id: str) -> dict[str, list[dict[str, str]]]:
        """Load saved context and the active dialog for one conversation."""

        with self._lock:
            data = self._read()
            conversations = data.get("conversations", {})
            if not isinstance(conversations, dict):
                raise HistoryStoreError("history file has invalid conversations data")

            stored_conversation = conversations.get(conversation_id, [])
            if isinstance(stored_conversation, list):
                # Read histories written by the first Day 7 implementation.
                return {"context": [], "messages": self._validate_messages(stored_conversation)}
            if not isinstance(stored_conversation, dict):
                raise HistoryStoreError("conversation state must be an object")
            return {
                "context": self._validate_messages(stored_conversation.get("context", [])),
                "messages": self._validate_messages(stored_conversation.get("messages", [])),
            }

    def save(self, conversation_id: str, messages: list[dict[str, str]]) -> None:
        """Save a complete history using the legacy-compatible API."""

        self.save_state(conversation_id, context=[], messages=messages)

    def save_state(
        self,
        conversation_id: str,
        *,
        context: list[dict[str, str]],
        messages: list[dict[str, str]],
    ) -> None:
        """Atomically save context and active dialog for one session."""

        validated_context = self._validate_messages(context)
        validated_messages = self._validate_messages(messages)
        with self._lock:
            data = self._read()
            conversations = data.setdefault("conversations", {})
            if not isinstance(conversations, dict):
                raise HistoryStoreError("history file has invalid conversations data")
            conversations[conversation_id] = {
                "context": validated_context,
                "messages": validated_messages,
            }

            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.path.with_name(f".{self.path.name}.tmp")
            try:
                temporary_path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary_path, self.path)
            except OSError as error:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise HistoryStoreError(f"could not save conversation history: {error}") from error

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"conversations": {}}

        try:
            raw_data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise HistoryStoreError(f"could not read conversation history: {error}") from error

        if not isinstance(raw_data, dict):
            raise HistoryStoreError("history file must contain a JSON object")
        return raw_data

    @staticmethod
    def _validate_messages(messages: Any) -> list[dict[str, str]]:
        if not isinstance(messages, list):
            raise HistoryStoreError("conversation history must be a JSON array")

        validated: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise HistoryStoreError("conversation history contains an invalid message")
            role = message.get("role")
            content = message.get("content")
            if role not in VALID_ROLES or not isinstance(content, str):
                raise HistoryStoreError("conversation history contains an invalid message")
            validated.append({"role": role, "content": content})
        return validated


class JsonMemoryStore:
    """Atomic JSON persistence for one conversation's layered memory."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._lock = RLock()

    def load(self) -> dict[str, Any]:
        with self._lock:
            if not self.path.exists():
                return {"version": 1, "entries": [], "events": []}
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise HistoryStoreError(f"could not read layered memory: {error}") from error
            if not isinstance(data, dict):
                raise HistoryStoreError("layered memory must be a JSON object")
            if not isinstance(data.get("entries", []), list):
                raise HistoryStoreError("layered memory entries must be a JSON array")
            return data

    def save(self, state: dict[str, Any]) -> None:
        with self._lock:
            data = {"version": 1, **state}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.path.with_name(f".{self.path.name}.tmp")
            try:
                temporary_path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary_path, self.path)
            except OSError as error:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise HistoryStoreError(f"could not save layered memory: {error}") from error

    def clear(self) -> None:
        with self._lock:
            try:
                self.path.unlink(missing_ok=True)
            except OSError as error:
                raise HistoryStoreError(f"could not clear layered memory: {error}") from error
