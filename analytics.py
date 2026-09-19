"""Request-level analytics for comparing Day 11 memory-aware strategies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class AnalyticsRecord:
    request_number: int
    timestamp: str
    strategy: str
    request_sent: bool
    prompt_tokens: int
    full_prompt_tokens: int
    saved_tokens: int
    savings_percent: float
    completion_tokens: int
    total_tokens: int
    auxiliary_calls: int
    auxiliary_prompt_tokens: int
    auxiliary_completion_tokens: int
    auxiliary_total_tokens: int
    total_tokens_including_auxiliary: int
    main_cost_usd: float
    auxiliary_cost_usd: float
    cost_usd: float
    elapsed_seconds: float
    context_characters: int
    full_context_characters: int
    window_messages: int
    discarded_messages: int
    facts_count: int
    retrieved_documents: list[dict[str, Any]]
    active_branch: str | None
    branch_count: int = 0
    checkpoint_count: int = 0
    error: str | None = None


class Analytics:
    """Collect request metrics and strategy-level aggregate snapshots."""

    def __init__(self) -> None:
        self.records: list[AnalyticsRecord] = []

    def record(
        self,
        metrics: dict[str, Any],
        *,
        elapsed_seconds: float = 0.0,
        error: str | None = None,
    ) -> AnalyticsRecord:
        record = AnalyticsRecord(
            request_number=len(self.records) + 1,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            strategy=str(metrics.get("strategy", metrics.get("memory_mode", "unknown"))),
            request_sent=bool(metrics.get("request_sent", False)),
            prompt_tokens=int(metrics.get("prompt_tokens", 0)),
            full_prompt_tokens=int(metrics.get("full_prompt_tokens", 0)),
            saved_tokens=int(metrics.get("saved_tokens", 0)),
            savings_percent=float(metrics.get("savings_percent", 0.0)),
            completion_tokens=int(metrics.get("completion_tokens", 0)),
            total_tokens=int(metrics.get("total_tokens", 0)),
            auxiliary_calls=int(metrics.get("auxiliary_calls", 0)),
            auxiliary_prompt_tokens=int(metrics.get("auxiliary_prompt_tokens", 0)),
            auxiliary_completion_tokens=int(metrics.get("auxiliary_completion_tokens", 0)),
            auxiliary_total_tokens=int(metrics.get("auxiliary_total_tokens", 0)),
            total_tokens_including_auxiliary=int(
                metrics.get("total_tokens_including_auxiliary", metrics.get("total_tokens", 0)),
            ),
            main_cost_usd=float(metrics.get("main_cost_usd", 0.0)),
            auxiliary_cost_usd=float(metrics.get("auxiliary_cost_usd", 0.0)),
            cost_usd=float(metrics.get("cost_usd", 0.0)),
            elapsed_seconds=round(elapsed_seconds, 3),
            context_characters=int(metrics.get("context_characters", 0)),
            full_context_characters=int(metrics.get("full_context_characters", 0)),
            window_messages=int(metrics.get("window_messages", 0)),
            discarded_messages=int(metrics.get("discarded_messages", 0)),
            facts_count=int(metrics.get("facts_count", 0)),
            retrieved_documents=list(metrics.get("retrieved_documents", [])),
            active_branch=metrics.get("active_branch"),
            branch_count=int(metrics.get("branch_count", 0)),
            checkpoint_count=int(metrics.get("checkpoint_count", 0)),
            error=error,
        )
        self.records.append(record)
        return record

    def snapshot(self) -> dict[str, Any]:
        successful = [record for record in self.records if record.request_sent and not record.error]
        failed = [record for record in self.records if record.error]
        by_strategy: dict[str, list[AnalyticsRecord]] = {}
        for record in successful:
            by_strategy.setdefault(record.strategy, []).append(record)

        def aggregate(records: list[AnalyticsRecord]) -> dict[str, Any]:
            if not records:
                return {
                    "requests": 0,
                    "prompt_tokens": 0,
                    "total_tokens_including_auxiliary": 0,
                    "cost_usd": 0.0,
                    "average_latency_seconds": 0.0,
                    "average_savings_percent": 0.0,
                    "average_facts_count": 0.0,
                    "average_retrieved_documents": 0.0,
                }
            return {
                "requests": len(records),
                "prompt_tokens": sum(item.prompt_tokens for item in records),
                "total_tokens_including_auxiliary": sum(
                    item.total_tokens_including_auxiliary for item in records
                ),
                "cost_usd": round(sum(item.cost_usd for item in records), 8),
                "average_latency_seconds": round(mean(item.elapsed_seconds for item in records), 3),
                "average_savings_percent": round(mean(item.savings_percent for item in records), 2),
                "average_facts_count": round(mean(item.facts_count for item in records), 2),
                "average_retrieved_documents": round(
                    mean(len(item.retrieved_documents) for item in records),
                    2,
                ),
            }

        return {
            "total_requests": len(self.records),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "total_prompt_tokens": sum(record.prompt_tokens for record in successful),
            "total_full_prompt_tokens": sum(record.full_prompt_tokens for record in successful),
            "total_saved_tokens": sum(record.saved_tokens for record in successful),
            "average_savings_percent": round(
                mean(record.savings_percent for record in successful), 2,
            ) if successful else 0.0,
            "total_completion_tokens": sum(record.completion_tokens for record in successful),
            "total_tokens": sum(record.total_tokens for record in successful),
            "total_auxiliary_tokens": sum(record.auxiliary_total_tokens for record in successful),
            "total_tokens_including_auxiliary": sum(
                record.total_tokens_including_auxiliary for record in successful
            ),
            "total_cost_usd": round(sum(record.cost_usd for record in successful), 8),
            "average_latency_seconds": round(
                mean(record.elapsed_seconds for record in successful), 3,
            ) if successful else 0.0,
            "by_strategy": {key: aggregate(value) for key, value in by_strategy.items()},
            "latest": asdict(self.records[-1]) if self.records else None,
            "series": [asdict(record) for record in self.records],
        }

    def reset(self) -> None:
        self.records.clear()
