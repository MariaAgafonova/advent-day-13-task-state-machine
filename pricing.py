"""DeepSeek model pricing used by the token and cost charts.

Rates are kept in one place so the application can compare models without
making an API request. They are the current official off-peak/peak prices per million
tokens. The application uses off-peak cache-miss rates for a conservative
default estimate; the UI displays all four input-rate variants.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelPricing:
    canonical_model: str
    context_limit_tokens: int
    input_cache_hit_off_peak: float
    input_cache_hit_peak: float
    input_cache_miss_off_peak: float
    input_cache_miss_peak: float
    output_off_peak: float
    output_peak: float

    def input_rate(self, *, peak: bool = False, cache_hit: bool = False) -> float:
        if peak and cache_hit:
            return self.input_cache_hit_peak
        if cache_hit:
            return self.input_cache_hit_off_peak
        if peak:
            return self.input_cache_miss_peak
        return self.input_cache_miss_off_peak

    def output_rate(self, *, peak: bool = False) -> float:
        return self.output_peak if peak else self.output_off_peak


# Official DeepSeek API prices, USD per 1,000,000 tokens.
MODEL_PRICING: dict[str, ModelPricing] = {
    "deepseek-v4-flash": ModelPricing(
        canonical_model="deepseek-v4-flash",
        context_limit_tokens=1_000_000,
        input_cache_hit_off_peak=0.007,
        input_cache_hit_peak=0.014,
        input_cache_miss_off_peak=0.22,
        input_cache_miss_peak=0.44,
        output_off_peak=0.66,
        output_peak=1.32,
    ),
    "deepseek-v4-pro": ModelPricing(
        canonical_model="deepseek-v4-pro",
        context_limit_tokens=1_000_000,
        input_cache_hit_off_peak=0.022,
        input_cache_hit_peak=0.044,
        input_cache_miss_off_peak=0.66,
        input_cache_miss_peak=1.32,
        output_off_peak=1.98,
        output_peak=3.96,
    ),
    "deepseek-v4-flash-vision-exp": ModelPricing(
        canonical_model="deepseek-v4-flash-vision-exp",
        context_limit_tokens=1_000_000,
        input_cache_hit_off_peak=0.007,
        input_cache_hit_peak=0.014,
        input_cache_miss_off_peak=0.22,
        input_cache_miss_peak=0.44,
        output_off_peak=0.66,
        output_peak=1.32,
    ),
}

MODEL_ALIASES = {
    "deepseek-flash": "deepseek-v4-flash",
    "deepseek-v4-flash": "deepseek-v4-flash",
    "deepseek-chat": "deepseek-v4-flash",
    "deepseek-reasoner": "deepseek-v4-flash",
    "deepseek-v4-pro": "deepseek-v4-pro",
    "deepseek-v4-flash-vision-exp": "deepseek-v4-flash-vision-exp",
}


def pricing_for_model(model: str) -> ModelPricing:
    """Return catalog pricing, falling back to the Flash rate for custom names."""

    canonical = MODEL_ALIASES.get(model, "deepseek-v4-flash")
    return MODEL_PRICING[canonical]


def cost_for_tokens(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    peak: bool = False,
    cache_hit: bool = False,
) -> float:
    """Estimate one request cost in USD."""

    pricing = pricing_for_model(model)
    return round(
        input_tokens * pricing.input_rate(peak=peak, cache_hit=cache_hit) / 1_000_000
        + output_tokens * pricing.output_rate(peak=peak) / 1_000_000,
        8,
    )


def catalog_payload() -> list[dict[str, Any]]:
    """Return JSON-friendly pricing rows for the web chart."""

    return [asdict(pricing) for pricing in MODEL_PRICING.values()]
