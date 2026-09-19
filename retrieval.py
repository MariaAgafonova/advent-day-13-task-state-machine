"""Local retrieval and retrieval-backed conversation memory.

The default retriever is intentionally local and deterministic.  It uses a
small TF-IDF cosine implementation so the experiment does not require an
additional embeddings key or network call.  The ``Retriever`` protocol makes
it possible to replace it with an embedding index later.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any, Protocol

from context_strategies import Message, StrategyName, _copy_messages, _message


TOKEN_RE = re.compile(r"[\w\u0400-\u04ff]+", re.UNICODE)


@dataclass(frozen=True)
class RetrievedDocument:
    id: str
    text: str
    role: str
    turn: int
    score: float


class Retriever(Protocol):
    def add(self, document_id: str, text: str, role: str, turn: int) -> None: ...
    def search(self, query: str, top_k: int, min_score: float) -> list[RetrievedDocument]: ...
    def reset(self) -> None: ...
    def snapshot(self) -> dict[str, Any]: ...
    def restore(self, state: dict[str, Any]) -> None: ...


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


@dataclass
class LocalTfidfRetriever:
    documents: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, document_id: str, text: str, role: str, turn: int) -> None:
        if not text.strip():
            return
        self.documents[document_id] = {
            "id": document_id,
            "text": text,
            "role": role,
            "turn": turn,
            "tokens": _tokens(text),
        }

    def search(self, query: str, top_k: int = 5, min_score: float = 0.1) -> list[RetrievedDocument]:
        if top_k <= 0:
            return []
        query_tokens = Counter(_tokens(query))
        if not query_tokens or not self.documents:
            return []

        document_frequency = Counter(
            token
            for document in self.documents.values()
            for token in set(document["tokens"])
        )
        total_documents = len(self.documents)

        def weight(token: str, frequency: int) -> float:
            inverse_document_frequency = math.log((1 + total_documents) / (1 + document_frequency[token])) + 1
            return frequency * inverse_document_frequency

        query_vector = {token: weight(token, frequency) for token, frequency in query_tokens.items()}
        query_norm = math.sqrt(sum(value * value for value in query_vector.values())) or 1.0
        results: list[RetrievedDocument] = []
        for document in self.documents.values():
            vector_counts = Counter(document["tokens"])
            vector = {
                token: weight(token, frequency)
                for token, frequency in vector_counts.items()
                if token in query_vector
            }
            if not vector:
                continue
            dot = sum(query_vector[token] * value for token, value in vector.items())
            norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
            score = dot / (query_norm * norm)
            if score >= min_score:
                results.append(
                    RetrievedDocument(
                        id=document["id"],
                        text=document["text"],
                        role=document["role"],
                        turn=int(document["turn"]),
                        score=round(score, 6),
                    ),
                )
        results.sort(key=lambda item: (-item.score, item.turn))
        return results[:top_k]

    def reset(self) -> None:
        self.documents.clear()

    def snapshot(self) -> dict[str, Any]:
        return {"documents": {key: value.copy() for key, value in self.documents.items()}}

    def restore(self, state: dict[str, Any]) -> None:
        self.documents = {
            key: value.copy()
            for key, value in state.get("documents", {}).items()
        }


@dataclass
class RetrievalStrategy:
    recent_limit: int = 10
    top_k: int = 5
    min_score: float = 0.1
    retriever: Retriever = field(default_factory=LocalTfidfRetriever)
    name: StrategyName = "retrieval"
    _history: list[Message] = field(default_factory=list)
    last_retrieved: list[RetrievedDocument] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.recent_limit, bool) or self.recent_limit <= 0:
            raise ValueError("recent_limit must be a positive integer")
        if isinstance(self.top_k, bool) or self.top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if self.min_score < 0:
            raise ValueError("min_score must not be negative")

    @property
    def history(self) -> list[Message]:
        return _copy_messages(self._history)

    def add_user_message(self, content: str) -> None:
        self._history.append(_message("user", content))

    def add_assistant_message(self, content: str) -> None:
        self._history.append(_message("assistant", content))
        turn = len(self._history)
        self.retriever.add(f"message-{turn}", content, "assistant", turn)
        previous_user = next(
            (message for message in reversed(self._history[:-1]) if message["role"] == "user"),
            None,
        )
        if previous_user:
            user_turn = turn - 1
            self.retriever.add(
                f"message-{user_turn}",
                previous_user["content"],
                "user",
                user_turn,
            )

    def build_context(self) -> list[Message]:
        query = next(
            (message["content"] for message in reversed(self._history) if message["role"] == "user"),
            "",
        )
        self.last_retrieved = [
            document
            for document in self.retriever.search(query, self.top_k, self.min_score)
        ]
        retrieved_messages = [
            {
                "role": "system",
                "content": (
                    f"[RETRIEVED MEMORY id={document.id} score={document.score}]\n"
                    f"{document.text}"
                ),
            }
            for document in self.last_retrieved
        ]
        current_user_message = next(
            (message for message in reversed(self._history) if message["role"] == "user"),
            None,
        )
        return [
            *retrieved_messages,
            *([current_user_message.copy()] if current_user_message else []),
        ]

    def snapshot(self) -> dict[str, Any]:
        return {
            "strategy": self.name,
            "recent_limit": self.recent_limit,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "history": self.history,
            "retriever": self.retriever.snapshot(),
            "last_retrieved": [asdict(document) for document in self.last_retrieved],
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.recent_limit = int(state.get("recent_limit", self.recent_limit))
        self.top_k = int(state.get("top_k", self.top_k))
        self.min_score = float(state.get("min_score", self.min_score))
        self._history = _copy_messages(state.get("history", []))
        self.retriever.restore(state.get("retriever", {}))
        self.last_retrieved = [
            RetrievedDocument(**document)
            for document in state.get("last_retrieved", [])
        ]

    def import_history(self, messages: list[Message]) -> None:
        self.reset()
        for message in messages:
            if message["role"] == "user":
                self.add_user_message(message["content"])
            else:
                self.add_assistant_message(message["content"])

    def reset(self) -> None:
        self._history.clear()
        self.retriever.reset()
        self.last_retrieved.clear()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "window_messages": 1 if self._history else 0,
            "discarded_messages": max(0, len(self._history) - 1),
            "facts_count": 0,
            "retrieved_documents": [asdict(document) for document in self.last_retrieved],
            "retrieval_documents_indexed": len(self.retriever.documents),
            "active_branch": None,
        }
