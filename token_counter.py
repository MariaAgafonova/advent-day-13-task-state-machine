"""Local token counting helpers used by the Day 8 token laboratory.

The provider's ``usage`` object is the source of truth after a request.  The
local counter is still useful before the request is sent: it can show how
large the prompt is, estimate the remaining context window, and reject an
obviously oversized request before paying for a failed API call.

``tiktoken`` is optional on purpose.  The application remains runnable in a fresh
checkout, while installations that include it get a model-aware BPE count.
The fallback is labelled as an estimate in the UI and in every metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TokenCounterInfo:
    """Describe the tokenizer used for a count."""

    method: str
    encoding: str
    is_exact: bool


class TokenCounter:
    """Count text and chat messages with an optional ``tiktoken`` backend."""

    def __init__(self, model: str = "") -> None:
        self.model = model
        self._encoding = None
        self._info = TokenCounterInfo(
            method="heuristic",
            encoding="characters-per-token",
            is_exact=False,
        )

        try:
            import tiktoken  # type: ignore[import-not-found]

            try:
                self._encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                # DeepSeek and other OpenAI-compatible providers may expose a
                # model name that tiktoken does not know yet.
                self._encoding = tiktoken.get_encoding("cl100k_base")
            self._info = TokenCounterInfo(
                method="tiktoken",
                encoding=self._encoding.name,
                is_exact=False,
            )
        except Exception:
            # Counting is an observability aid, not a reason to make the
            # whole agent unusable when the optional package is unavailable.
            self._encoding = None

    @property
    def info(self) -> TokenCounterInfo:
        return self._info

    def count_text(self, text: str) -> int:
        """Return the number of tokens in a text value."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if self._encoding is not None:
            return len(self._encoding.encode(text, disallowed_special=()))
        if not text:
            return 0
        # A deliberately conservative approximation. English prose is often
        # close to four characters per token; punctuation and CJK text make
        # the result less precise, hence the explicit "estimate" label.
        return max(1, (len(text) + 3) // 4)

    def count_message(self, message: dict[str, str]) -> int:
        """Count one chat message, including a small chat-format overhead."""

        role = message.get("role", "")
        content = message.get("content", "")
        if not isinstance(role, str) or not isinstance(content, str):
            raise TypeError("message role and content must be strings")
        return 4 + self.count_text(role) + self.count_text(content)

    def count_messages(self, messages: Sequence[dict[str, str]]) -> int:
        """Estimate tokens for a list of messages sent to a chat API."""

        # Two tokens approximate the assistant priming added by common chat
        # APIs. The provider's returned prompt_tokens supersede this estimate.
        return 2 + sum(self.count_message(message) for message in messages)
