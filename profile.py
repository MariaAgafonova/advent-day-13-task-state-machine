"""User profiles and their isolated JSON persistence for Day 12."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
from threading import RLock
from typing import Any, Mapping, Protocol


PROFILE_FIELDS = (
    "name",
    "language",
    "expertise_level",
    "response_style",
    "preferred_format",
    "response_length",
    "interests",
    "restrictions",
    "custom_instructions",
)
PROFILE_JSON_FIELDS = {
    "id": "id",
    "name": "name",
    "language": "language",
    "expertiseLevel": "expertise_level",
    "expertise_level": "expertise_level",
    "responseStyle": "response_style",
    "response_style": "response_style",
    "preferredFormat": "preferred_format",
    "preferred_format": "preferred_format",
    "responseLength": "response_length",
    "response_length": "response_length",
    "interests": "interests",
    "restrictions": "restrictions",
    "customInstructions": "custom_instructions",
    "custom_instructions": "custom_instructions",
}
SENSITIVE_MARKERS = (
    "password",
    "passcode",
    "secret",
    "api_key",
    "apikey",
    "token",
    "authorization",
    "card_number",
    "cvv",
    "medical",
    "diagnosis",
)
DEFAULT_USER_ID = "user_1"


class ProfileRepository(Protocol):
    def get(self, user_id: str) -> "UserProfile | None": ...

    def get_or_default(self, user_id: str) -> "UserProfile": ...

    def create(self, profile: "UserProfile") -> "UserProfile": ...

    def update(self, user_id: str, fields: Mapping[str, Any]) -> "UserProfile": ...


class ProfileStoreError(RuntimeError):
    """Raised when a profile cannot be read or safely persisted."""


@dataclass(frozen=True)
class UserProfile:
    """Durable, non-sensitive preferences used to personalize responses."""

    id: str
    name: str = ""
    language: str = "ru"
    expertise_level: str = "general"
    response_style: str = "clear_and_practical"
    preferred_format: str = "plain_text"
    response_length: str = "medium"
    interests: tuple[str, ...] = ()
    restrictions: tuple[str, ...] = ()
    custom_instructions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("profile id must be a non-empty string")
        object.__setattr__(self, "id", self.id.strip())
        for field_name in PROFILE_FIELDS:
            value = getattr(self, field_name)
            if field_name in {"interests", "restrictions", "custom_instructions"}:
                object.__setattr__(self, field_name, _normalise_list(value))
            else:
                object.__setattr__(self, field_name, _normalise_text(value))

    @classmethod
    def defaults(cls, user_id: str = DEFAULT_USER_ID) -> "UserProfile":
        return cls(id=user_id)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], user_id: str | None = None) -> "UserProfile":
        if not isinstance(raw, Mapping):
            raise ValueError("profile must be a JSON object")
        values: dict[str, Any] = {"id": user_id or raw.get("id")}
        for key, field_name in PROFILE_JSON_FIELDS.items():
            if key in raw and field_name != "id":
                values[field_name] = raw[key]
        if not values["id"]:
            raise ValueError("profile id is required")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return the public JSON schema used by the profile files and API."""
        return {
            "id": self.id,
            "name": self.name,
            "language": self.language,
            "expertiseLevel": self.expertise_level,
            "responseStyle": self.response_style,
            "preferredFormat": self.preferred_format,
            "responseLength": self.response_length,
            "interests": list(self.interests),
            "restrictions": list(self.restrictions),
            "customInstructions": list(self.custom_instructions),
        }

    def with_updates(self, fields: Mapping[str, Any]) -> "UserProfile":
        normalised: dict[str, Any] = {}
        for key, value in fields.items():
            field_name = PROFILE_JSON_FIELDS.get(key)
            if field_name is None or field_name == "id":
                raise ValueError(f"unknown or immutable profile field: {key}")
            normalised[field_name] = value
        return replace(self, **normalised)

    def prompt_block(self) -> str:
        """Render only safe preference fields; the full profile is sent once."""
        lines = [
            "User profile:",
            f"- Preferred language: {_language_label(self.language)}",
            f"- Required response language: {_language_label(self.language)}",
            "- Write the complete final answer in the required response language. "
            "An explicit language request in the current message overrides this preference.",
            f"- Expertise: {_label(self.expertise_level)}",
            f"- Style: {_label(self.response_style)}",
            f"- Preferred format: {_label(self.preferred_format)}",
            f"- Response length: {_label(self.response_length)}",
        ]
        if self.name:
            lines.insert(1, f"- Name: {self.name}")
        if self.interests:
            lines.append(f"- Interests: {', '.join(self.interests)}")
        for restriction in self.restrictions:
            lines.append(f"- Restriction: {restriction}")
        for instruction in self.custom_instructions:
            lines.append(f"- Custom instruction: {instruction}")
        return "\n".join(lines)

    # Camel-case aliases make the model convenient for integrations using the
    # JSON field names from the assignment while keeping Python naming idiomatic.
    @property
    def expertiseLevel(self) -> str:  # noqa: N802
        return self.expertise_level

    @property
    def responseStyle(self) -> str:  # noqa: N802
        return self.response_style

    @property
    def preferredFormat(self) -> str:  # noqa: N802
        return self.preferred_format

    @property
    def responseLength(self) -> str:  # noqa: N802
        return self.response_length

    @property
    def customInstructions(self) -> tuple[str, ...]:  # noqa: N802
        return self.custom_instructions


class JsonProfileRepository:
    """Store profiles separately from conversation and memory JSON files."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)
        self._lock = RLock()

    def path_for_user(self, user_id: str) -> Path:
        user_id = _validate_user_id(user_id)
        filename = user_id if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", user_id) else hashlib.sha256(
            user_id.encode("utf-8"),
        ).hexdigest()
        return self.directory / f"{filename}.json"

    def get(self, user_id: str) -> UserProfile | None:
        path = self.path_for_user(user_id)
        with self._lock:
            if not path.exists():
                return None
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                return UserProfile.from_dict(raw, user_id=user_id)
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
                raise ProfileStoreError(f"could not read profile: {error}") from error

    def get_or_default(self, user_id: str) -> UserProfile:
        return self.get(user_id) or UserProfile.defaults(user_id)

    def create(self, profile: UserProfile) -> UserProfile:
        path = self.path_for_user(profile.id)
        with self._lock:
            if path.exists():
                raise ProfileStoreError(f"profile already exists: {profile.id}")
            self._save_locked(profile, path)
        return profile

    def save(self, profile: UserProfile) -> UserProfile:
        path = self.path_for_user(profile.id)
        with self._lock:
            self._save_locked(profile, path)
        return profile

    def update(self, user_id: str, fields: Mapping[str, Any]) -> UserProfile:
        current = self.get_or_default(user_id)
        updated = current.with_updates(fields)
        return self.save(updated)

    def _save_locked(self, profile: UserProfile, path: Path) -> None:
        # UserProfile is a whitelist, so unknown/sensitive payload keys never
        # reach disk. This is a preference store, not a secret store.
        payload = {"version": 1, **profile.to_dict()}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp")
        try:
            temporary_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_path, path)
        except OSError as error:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise ProfileStoreError(f"could not save profile: {error}") from error


class InMemoryProfileRepository:
    """Deterministic fake used by unit tests and local comparisons."""

    def __init__(self, profiles: Mapping[str, UserProfile] | None = None) -> None:
        self.profiles = {key: deepcopy(value) for key, value in (profiles or {}).items()}

    def get(self, user_id: str) -> UserProfile | None:
        return self.profiles.get(user_id)

    def get_or_default(self, user_id: str) -> UserProfile:
        return self.get(user_id) or UserProfile.defaults(user_id)

    def create(self, profile: UserProfile) -> UserProfile:
        if profile.id in self.profiles:
            raise ProfileStoreError(f"profile already exists: {profile.id}")
        self.profiles[profile.id] = profile
        return profile

    def update(self, user_id: str, fields: Mapping[str, Any]) -> UserProfile:
        updated = self.get_or_default(user_id).with_updates(fields)
        self.profiles[user_id] = updated
        return updated


def demo_profiles() -> dict[str, UserProfile]:
    return {
        "beginner": UserProfile(
            id="beginner",
            language="ru",
            expertise_level="beginner",
            response_style="educational",
            preferred_format="step_by_step",
            response_length="detailed",
            restrictions=("avoid_complex_terms",),
        ),
        "developer": UserProfile(
            id="developer",
            language="en",
            expertise_level="senior_developer",
            response_style="technical",
            preferred_format="code_first",
            response_length="short",
            restrictions=("no_basic_explanations",),
        ),
        "manager": UserProfile(
            id="manager",
            language="ru",
            expertise_level="non_technical",
            response_style="business",
            preferred_format="summary_and_bullets",
            response_length="medium",
            restrictions=("avoid_implementation_details",),
        ),
        "user_1": UserProfile(
            id="user_1",
            name="Maria",
            language="ru",
            expertise_level="middle_android_developer",
            response_style="clear_and_practical",
            preferred_format="steps_and_code_examples",
            response_length="medium",
            interests=("Android", "Kotlin", "AI"),
            restrictions=("avoid_unnecessary_theory",),
            custom_instructions=("Use Kotlin for programming examples", "Explain unfamiliar AI terms"),
        ),
    }


def seed_demo_profiles(repository: ProfileRepository) -> None:
    for profile in demo_profiles().values():
        if repository.get(profile.id) is None:
            try:
                repository.create(profile)
            except ProfileStoreError:
                # A concurrent web request may have created the same profile.
                pass


def parse_request_overrides(question: str) -> dict[str, str]:
    """Extract obvious per-request preference overrides for transparent logs."""
    text = question.casefold()
    overrides: dict[str, str] = {}
    if re.search(r"\b(in english|на английском|по-английски|english)\b", text):
        overrides["language"] = "en"
    elif re.search(r"\b(in russian|на русском|по-русски|русский|русском)\b", text):
        overrides["language"] = "ru"
    if re.search(r"\b(detailed|подробн|развернут|детальн)\w*\b", text):
        overrides["response_length"] = "detailed"
    elif re.search(r"\b(short|brief|кратк|коротк)\w*\b", text):
        overrides["response_length"] = "short"
    if re.search(r"\b(step[-_ ]by[-_ ]step|пошаг|по шагам|steps)\w*\b", text):
        overrides["preferred_format"] = "step_by_step"
    elif re.search(r"\b(code[-_ ]first|сначала код|только код)\b", text):
        overrides["preferred_format"] = "code_first"
    if re.search(r"\b(technical|техническ)\w*\b", text):
        overrides["response_style"] = "technical"
    elif re.search(r"\b(business|делов|бизнес)\w*\b", text):
        overrides["response_style"] = "business"
    return overrides


def normalise_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping):
        raise ValueError("profile_overrides must be an object")
    normalised: dict[str, Any] = {}
    for key, value in overrides.items():
        field_name = PROFILE_JSON_FIELDS.get(key)
        if field_name is None or field_name == "id":
            raise ValueError(f"unknown or immutable profile override: {key}")
        if isinstance(value, str) and any(marker in value.casefold() for marker in SENSITIVE_MARKERS):
            raise ValueError("profile overrides must not contain sensitive data")
        normalised[field_name] = value
    return normalised


def _normalise_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("profile text fields must be strings")
    if any(marker in value.casefold() for marker in SENSITIVE_MARKERS):
        raise ValueError("profiles must not contain sensitive data")
    return value.strip()


def _normalise_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        raise ValueError("profile list fields must be arrays")
    result: list[str] = []
    for item in value:
        text = _normalise_text(item)
        if text:
            result.append(text)
    return tuple(result)


def _validate_user_id(user_id: str) -> str:
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")
    return user_id.strip()


def _label(value: str) -> str:
    return value.replace("_", " ").strip().capitalize()


def _language_label(value: str) -> str:
    labels = {
        "ru": "Russian",
        "en": "English",
        "uk": "Ukrainian",
        "sr": "Serbian",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "it": "Italian",
        "pt": "Portuguese",
    }
    return f"{labels.get(value.casefold(), value)} ({value})"
