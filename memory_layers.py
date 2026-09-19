"""Explicit three-layer memory and its retention policy.

Short-term memory contains recent dialogue, working memory contains the active
task state, and long-term memory contains explicit durable user facts.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import re
import time
from typing import Any, Callable, Literal, Protocol
from uuid import uuid4


LayerName = Literal["short_term", "working", "long_term"]
MemoryKind = Literal[
    "dialogue", "task", "decision", "profile", "preference", "pattern", "knowledge"
]
MemoryStorageMode = Literal["in_memory", "json_file"]
MemoryTarget = Literal["auto", "short_term", "working", "long_term", "none"]
MemoryContextLayer = Literal["all", "short_term", "working", "long_term"]
LAYER_PRIORITY: tuple[LayerName, ...] = ("working", "short_term", "long_term")
VALID_LAYERS = set(LAYER_PRIORITY)
VALID_CONTEXT_LAYERS = {"all", "short_term", "working", "long_term"}
CONTEXT_LAYER_SOURCES: dict[str, tuple[LayerName, ...]] = {
    "all": LAYER_PRIORITY,
    "short_term": LAYER_PRIORITY,
    "working": ("working", "long_term"),
    "long_term": ("long_term",),
}
VALID_MEMORY_TARGETS = {"auto", "short_term", "working", "long_term", "none"}
MEMORY_TARGET_ALIASES = {
    "automatic": "auto", "auto": "auto",
    "short": "short_term", "short_term": "short_term",
    "working": "working", "task": "working",
    "long": "long_term", "long_term": "long_term",
    "none": "none", "off": "none",
}
VALID_KINDS = {
    "dialogue", "task", "decision", "profile", "preference", "pattern", "knowledge"
}
STORAGE_ALIASES = {
    "memory": "in_memory", "in_memory": "in_memory",
    "json": "json_file", "json_file": "json_file",
}
TOKEN_RE = re.compile(r"[\w\u0400-\u04ff]+", re.UNICODE)


class MemoryPersistence(Protocol):
    def load(self) -> dict[str, Any]: ...
    def save(self, state: dict[str, Any]) -> None: ...
    def clear(self) -> None: ...


def _now() -> float:
    return time.time()


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds")


def _normalise(text: str) -> str:
    return " ".join(text.casefold().split())


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(text)}


@dataclass
class MemoryEntry:
    id: str
    layer: LayerName
    kind: MemoryKind
    content: str
    source: str
    importance: float
    created_at: float
    updated_at: float
    expires_at: float | None = None
    key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Any) -> "MemoryEntry":
        if not isinstance(raw, dict):
            raise ValueError("memory entry must be an object")
        layer, kind, content = raw.get("layer"), raw.get("kind"), raw.get("content")
        if layer not in VALID_LAYERS or kind not in VALID_KINDS:
            raise ValueError("memory entry has invalid layer or kind")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("memory entry content must be non-empty")
        created_at = float(raw.get("created_at", 0))
        return cls(
            id=str(raw.get("id") or uuid4().hex), layer=layer, kind=kind,
            content=content.strip(), source=str(raw.get("source", "user")),
            importance=max(0.0, min(1.0, float(raw.get("importance", 0.5)))),
            created_at=created_at, updated_at=float(raw.get("updated_at", created_at)),
            expires_at=float(raw["expires_at"]) if raw.get("expires_at") is not None else None,
            key=str(raw["key"]) if raw.get("key") is not None else None,
            metadata=deepcopy(raw.get("metadata", {}))
            if isinstance(raw.get("metadata", {}), dict) else {},
        )


@dataclass(frozen=True)
class Classification:
    layers: tuple[LayerName, ...]
    kind: MemoryKind
    reason: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {"layers": list(self.layers), "kind": self.kind,
                "reason": self.reason, "confidence": self.confidence}


def classify_message(content: str) -> Classification:
    """Apply explicit, deterministic retention rules to a user message."""
    text = _normalise(content)
    if re.search(
        r"\b(запомни|предпочитаю|предпочтение|мне нравится|мне не нравится|"
        r"люблю|не люблю|по умолчанию|всегда|обычно|prefer|preference|"
        r"i like|i prefer|remember this|my preferred)\b", text,
    ):
        kind: MemoryKind = "pattern" if re.search(
            r"\b(всегда|обычно|часто|как правило)\b", text
        ) else "preference"
        return Classification(("short_term", "long_term"), kind,
                              "repeated or explicit user preference", 0.92)
    if re.search(
        r"\b(мой|моя|мои|меня зовут|я живу|мой часовой пояс|мой профиль|"
        r"my name|my profile|i live|my timezone)\b", text,
    ):
        return Classification(("short_term", "long_term"), "profile",
                              "explicit user profile data", 0.9)
    if re.search(
        r"(из прошлого|раньше работало|проверенное решение|запомни решение|"
        r"опыт|знание|known solution|from before|remember the solution)", text,
    ):
        return Classification(("short_term", "long_term"), "knowledge",
                              "explicit reusable knowledge or past solution", 0.88)
    if re.search(
        r"(задач|цель|план|шаг|следующ|решаем|решение|реши|нужно|надо|"
        r"требован|ограничен|выбран|параметр|вычисл|расч|промежуточ|"
        r"todo|task|goal|plan|step|decision|constraint|selected|parameter)", text,
    ):
        kind: MemoryKind = "decision" if re.search(r"решени|decision", text) else "task"
        return Classification(("short_term", "working"), kind,
                              "active task step, calculation, parameter, or decision", 0.84)
    return Classification(("short_term",), "dialogue", "ordinary dialogue message", 0.7)


class LayeredMemory:
    """Retention, retrieval, lifecycle, and audit data for all memory layers."""

    def __init__(
        self, *, storage_mode: str = "in_memory",
        persistence: MemoryPersistence | None = None,
        short_term_limit: int = 6, short_term_ttl_seconds: int = 86_400,
        working_limit: int = 30, long_term_limit: int = 100,
        long_term_top_k: int = 8, clock: Callable[[], float] = _now,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.storage_mode = normalise_storage_mode(storage_mode)
        if min(short_term_limit, working_limit, long_term_limit, long_term_top_k) <= 0:
            raise ValueError("memory limits must be positive")
        if short_term_ttl_seconds < 0:
            raise ValueError("short_term_ttl_seconds must not be negative")
        if self.storage_mode == "json_file" and persistence is None:
            raise ValueError("json_file memory requires persistence")
        self.persistence = persistence
        self.short_term_limit, self.short_term_ttl_seconds = short_term_limit, short_term_ttl_seconds
        self.working_limit, self.long_term_limit = working_limit, long_term_limit
        self.long_term_top_k = long_term_top_k
        self._clock, self._id_factory = clock, id_factory or (lambda: uuid4().hex)
        self._entries: list[MemoryEntry] = []
        self._events: list[dict[str, Any]] = []
        self._last_selection: list[dict[str, Any]] = []
        self._last_event: dict[str, Any] | None = None
        if self.storage_mode == "json_file":
            self.restore(self.persistence.load())  # type: ignore[union-attr]
        self._prune(save=False)

    def add_user_message(self, content: str, memory_target: str = "auto") -> Classification:
        content = self._validate_content(content)
        selected_target = normalise_memory_target(memory_target)
        automatic = classify_message(content)
        classification = self._classification_for_target(automatic, selected_target)
        timestamp = self._clock()
        self._append(layer="short_term", kind="dialogue", content=content, source="user",
                      importance=0.55, timestamp=timestamp, expires_at=self._expires_at(timestamp),
                      metadata={"classification": classification.kind,
                                "retention_mode": selected_target})
        semantic_source = "user:explicit" if selected_target != "auto" else "user"
        for layer in classification.layers:
            if layer == "short_term" or selected_target == "none":
                continue
            self._upsert(layer=layer, kind=classification.kind, content=content, source=semantic_source,
                         importance=classification.confidence, timestamp=timestamp,
                         key=f"{classification.kind}:{_normalise(content)}",
                         metadata={"reason": classification.reason,
                                   "retention_mode": selected_target,
                                   "explicit": selected_target != "auto"})
        self._last_event = {
            "timestamp": _iso(timestamp),
            "type": "classified" if selected_target == "auto" else "retention_overridden",
            "content": content,
            "retention_mode": selected_target,
            "explicit": selected_target != "auto",
            "stored_layers": list(classification.layers),
            **classification.to_dict(),
        }
        self._events.append(deepcopy(self._last_event))
        self._events = self._events[-30:]
        self._commit()
        return classification

    def add_assistant_message(self, content: str) -> None:
        content, timestamp = self._validate_content(content), self._clock()
        self._append(layer="short_term", kind="dialogue", content=content, source="assistant",
                      importance=0.45, timestamp=timestamp, expires_at=self._expires_at(timestamp))
        self._commit()

    def build_context(
        self,
        query: str | None = None,
        memory_layer: str = "all",
        exclude_content: str | None = None,
        layer_order: tuple[LayerName, ...] | None = None,
    ) -> list[dict[str, str]]:
        """Return the selected memory view with its allowed supporting layers.

        The short-term view is the normal chat context and includes all three
        layers in priority order. Working adds long-term support, while the
        long-term view is isolated to durable records.
        """
        if memory_layer not in VALID_CONTEXT_LAYERS:
            raise ValueError("memory_layer must be all, short_term, working, or long_term")
        self._prune(save=True)
        excluded = _normalise(exclude_content) if exclude_content else None
        sources = CONTEXT_LAYER_SOURCES[memory_layer]
        working = self._entries_for("working") if "working" in sources else []
        short_term = (
            sorted(self._entries_for("short_term"), key=lambda item: item.updated_at)[-self.short_term_limit:]
            if "short_term" in sources else []
        )
        long_term = self._select_long_term(query) if "long_term" in sources else []
        if excluded:
            short_term = [item for item in short_term if _normalise(item.content) != excluded]
        entries_by_layer = {
            "working": working,
            "short_term": short_term,
            "long_term": long_term,
        }
        selected_order = tuple(
            layer for layer in (layer_order or LAYER_PRIORITY) if layer in sources
        )
        if set(selected_order) != set(sources) or len(selected_order) != len(sources):
            raise ValueError("layer_order must contain each selected memory layer once")
        selected = [
            item
            for layer in selected_order
            for item in entries_by_layer[layer]
        ]
        self._last_selection = [
            {"id": item.id, "layer": item.layer, "kind": item.kind,
             "score": round(score, 4), "reason": reason}
            for item, score, reason in [
                *[
                    (item, 1.0, "active task state")
                    for item in entries_by_layer["working"]
                ],
                *[
                    (item, 1.0, "recent dialogue")
                    for item in entries_by_layer["short_term"]
                ],
                *[
                    (item, self._relevance(item, query), "relevant durable memory")
                    for item in entries_by_layer["long_term"]
                ],
            ]
        ]
        if not selected:
            return []
        sections = ["[MEMORY POLICY]"]
        if memory_layer == "all":
            sections.extend([
                "Use memory in priority order: working > short_term > long_term.",
                "Working memory is current-task state; short-term is recent dialogue; "
                "long-term is durable user information and is used when relevant.",
            ])
        elif memory_layer == "short_term":
            sections.extend([
                "Short-term chat context includes supporting working and long-term memory.",
                "Use memory in priority order: working > short_term > long_term.",
            ])
        elif memory_layer == "working":
            sections.extend([
                "Working context includes the active task and supporting long-term memory.",
                "Do not use short-term dialogue records.",
            ])
        else:
            sections.extend([
                "Use only long-term memory for this answer.",
                "Do not use records from other memory layers.",
            ])
        for layer in selected_order:
            entries = [item for item in selected if item.layer == layer]
            if not entries:
                continue
            sections.append(f"\n[{layer.upper()} MEMORY]")
            sections.extend(
                f"- id={item.id}; kind={item.kind}; importance={item.importance:.2f}; {item.content}"
                for item in entries
            )
        return [{"role": "system", "content": "\n".join(sections)}]

    def complete_task(self) -> int:
        """Clear working memory and keep short-term and long-term memory."""
        removed = sum(item.layer == "working" for item in self._entries)
        self._entries = [item for item in self._entries if item.layer != "working"]
        self._last_event = {"timestamp": _iso(self._clock()), "type": "task_completed",
                            "removed_working_entries": removed}
        self._commit()
        return removed

    def clear_layer(self, layer: str) -> int:
        """Clear one layer without touching records in the other layers."""
        if layer not in VALID_LAYERS:
            raise ValueError("layer must be short_term, working, or long_term")
        removed = sum(item.layer == layer for item in self._entries)
        if removed:
            self._entries = [item for item in self._entries if item.layer != layer]
        self._last_selection.clear()
        self._last_event = {
            "timestamp": _iso(self._clock()),
            "type": "layer_cleared",
            "layer": layer,
            "removed_entries": removed,
        }
        self._commit()
        return removed

    def forget(self, entry_id: str) -> bool:
        before = len(self._entries)
        self._entries = [item for item in self._entries if item.id != entry_id]
        if len(self._entries) == before:
            return False
        self._last_event = {"timestamp": _iso(self._clock()), "type": "forgotten", "entry_id": entry_id}
        self._commit()
        return True

    def set_storage_mode(self, storage_mode: str) -> None:
        selected = normalise_storage_mode(storage_mode)
        if selected == "json_file" and self.persistence is None:
            raise ValueError("json_file memory requires persistence")
        self.storage_mode = selected
        self._commit()

    def reset(self) -> None:
        self._entries.clear()
        self._events.clear()
        self._last_selection.clear()
        self._last_event = {"timestamp": _iso(self._clock()), "type": "reset"}
        if self.persistence is not None:
            self.persistence.clear()

    def snapshot(self) -> dict[str, Any]:
        return {"entries": [item.to_dict() for item in self._entries],
                "events": deepcopy(self._events),
                "last_selection": deepcopy(self._last_selection),
                "last_event": deepcopy(self._last_event)}

    def persistent_snapshot(self) -> dict[str, Any]:
        """Persist only durable facts; session and task context never cross sessions."""
        return {
            "entries": [item.to_dict() for item in self._entries if item.layer == "long_term"],
            "events": [],
            "last_selection": [],
            "last_event": None,
        }

    def restore(self, state: dict[str, Any], *, persist: bool = False) -> None:
        if not isinstance(state, dict) or not isinstance(state.get("entries", []), list):
            raise ValueError("memory state must contain an entries array")
        self._entries = [MemoryEntry.from_dict(item) for item in state.get("entries", [])]
        self._events = deepcopy(state.get("events", []))[-30:]
        self._last_selection = deepcopy(state.get("last_selection", []))
        self._last_event = deepcopy(state.get("last_event"))
        self._prune(save=False)
        if persist:
            self._commit()

    def state(self) -> dict[str, Any]:
        self._prune(save=True)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in self._entries:
            view = item.to_dict()
            view["created_at_iso"], view["updated_at_iso"] = _iso(item.created_at), _iso(item.updated_at)
            view["expires_at_iso"] = _iso(item.expires_at) if item.expires_at else None
            grouped[item.layer].append(view)
        for layer in LAYER_PRIORITY:
            grouped[layer] = list(
                reversed(sorted(grouped[layer], key=lambda item: item["updated_at"]))
            )
        return {
            "storage_mode": self.storage_mode, "priority": list(LAYER_PRIORITY),
            "limits": {"short_term": self.short_term_limit,
                        "short_term_ttl_seconds": self.short_term_ttl_seconds,
                        "working": self.working_limit, "long_term": self.long_term_limit,
                        "long_term_top_k": self.long_term_top_k},
            "counts": {layer: len(grouped[layer]) for layer in LAYER_PRIORITY},
            "layers": {layer: grouped[layer] for layer in LAYER_PRIORITY},
            "last_selection": deepcopy(self._last_selection),
            "events": deepcopy(self._events[-20:]), "last_event": deepcopy(self._last_event),
        }

    def analysis(self) -> dict[str, Any]:
        current = self.state()
        return {
            "saved_by_layer": current["counts"],
            "classification_events": len(self._events),
            "last_classification": self._events[-1] if self._events else None,
            "classification_errors": [],
            "quality_effect": {
                "working": "preserves active task state after older dialogue leaves the window",
                "short_term": "preserves recent conversational continuity",
                "long_term": "reuses explicit durable facts without sending all history",
            },
            "quality_note": "Memory coverage is observable here; answer quality needs an A/B evaluation.",
        }

    def _append(self, *, layer: LayerName, kind: MemoryKind, content: str, source: str,
                importance: float, timestamp: float, expires_at: float | None = None,
                metadata: dict[str, Any] | None = None) -> MemoryEntry:
        entry = MemoryEntry(id=self._id_factory(), layer=layer, kind=kind, content=content,
                            source=source, importance=max(0.0, min(1.0, importance)),
                            created_at=timestamp, updated_at=timestamp, expires_at=expires_at,
                            metadata=metadata or {})
        self._entries.append(entry)
        return entry

    def _upsert(self, *, layer: LayerName, kind: MemoryKind, content: str, source: str,
                importance: float, timestamp: float, key: str,
                metadata: dict[str, Any] | None = None) -> MemoryEntry:
        existing = next((item for item in self._entries if item.layer == layer and item.key == key), None)
        if existing is None:
            existing = self._append(layer=layer, kind=kind, content=content, source=source,
                                    importance=importance, timestamp=timestamp, metadata=metadata)
            existing.key = key
        else:
            existing.content, existing.updated_at, existing.source = content, timestamp, source
            existing.importance = max(existing.importance, importance)
            existing.metadata = {**existing.metadata, **(metadata or {})}
        return existing

    @staticmethod
    def _classification_for_target(
        automatic: Classification,
        target: MemoryTarget,
    ) -> Classification:
        if target == "auto":
            return automatic
        if target in {"short_term", "none"}:
            reason = (
                "explicit short-term-only target"
                if target == "short_term"
                else "explicitly disabled semantic retention"
            )
            return Classification(("short_term",), "dialogue", reason, 1.0)
        if target == "working":
            kind: MemoryKind = automatic.kind if automatic.kind in {"task", "decision"} else "task"
            return Classification(("short_term", "working"), kind,
                                  "explicit working-memory target", 1.0)
        kind = automatic.kind if automatic.kind in {
            "profile", "preference", "pattern", "knowledge"
        } else "knowledge"
        return Classification(("short_term", "long_term"), kind,
                              "explicit long-term target", 1.0)

    def _select_long_term(self, query: str | None) -> list[MemoryEntry]:
        entries = self._entries_for("long_term")
        ranked = sorted(entries, key=lambda item: (self._relevance(item, query), item.importance, item.updated_at), reverse=True)
        if query and any(self._overlap(item.content, query) for item in entries):
            ranked = [item for item in ranked if self._overlap(item.content, query)]
        return ranked[: self.long_term_top_k]

    @staticmethod
    def _overlap(content: str, query: str) -> bool:
        return bool(_tokens(content) & _tokens(query))

    def _relevance(self, item: MemoryEntry, query: str | None) -> float:
        if not query:
            return item.importance
        return len(_tokens(item.content) & _tokens(query)) / max(1, len(_tokens(query))) + item.importance * 0.1

    def _entries_for(self, layer: LayerName) -> list[MemoryEntry]:
        return [item for item in self._entries if item.layer == layer]

    def _prune(self, *, save: bool) -> None:
        timestamp, before = self._clock(), len(self._entries)
        self._entries = [item for item in self._entries if item.expires_at is None or item.expires_at > timestamp]
        for layer, limit in (("short_term", self.short_term_limit), ("working", self.working_limit)):
            items = sorted(self._entries_for(layer), key=lambda item: item.updated_at)
            keep_ids = {item.id for item in items[-limit:]}
            self._entries = [item for item in self._entries if item.layer != layer or item.id in keep_ids]
        long_term = sorted(self._entries_for("long_term"), key=lambda item: (item.importance, item.updated_at), reverse=True)
        keep_ids = {item.id for item in long_term[: self.long_term_limit]}
        self._entries = [item for item in self._entries if item.layer != "long_term" or item.id in keep_ids]
        if save and len(self._entries) != before:
            self._commit()

    def _commit(self) -> None:
        self._prune(save=False)
        if self.storage_mode == "json_file" and self.persistence is not None:
            self.persistence.save(self.persistent_snapshot())

    def _expires_at(self, timestamp: float) -> float | None:
        return timestamp + self.short_term_ttl_seconds if self.short_term_ttl_seconds > 0 else None

    @staticmethod
    def _validate_content(content: str) -> str:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("memory content must be a non-empty string")
        return content.strip()


def normalise_storage_mode(storage_mode: str) -> MemoryStorageMode:
    selected = STORAGE_ALIASES.get(storage_mode.strip().casefold()) if isinstance(storage_mode, str) else None
    if selected not in {"in_memory", "json_file"}:
        raise ValueError("memory_storage must be in_memory or json_file")
    return selected  # type: ignore[return-value]


def normalise_memory_target(memory_target: str) -> MemoryTarget:
    selected = (
        MEMORY_TARGET_ALIASES.get(memory_target.strip().casefold())
        if isinstance(memory_target, str) else None
    )
    if selected not in VALID_MEMORY_TARGETS:
        raise ValueError(
            "memory_target must be auto, short_term, working, long_term, or none"
        )
    return selected  # type: ignore[return-value]


def normalise_context_layer(memory_layer: str) -> MemoryContextLayer:
    selected = (
        memory_layer.strip().casefold()
        if isinstance(memory_layer, str) else None
    )
    if selected not in VALID_CONTEXT_LAYERS:
        raise ValueError("context_layer must be all, short_term, working, or long_term")
    return selected  # type: ignore[return-value]
