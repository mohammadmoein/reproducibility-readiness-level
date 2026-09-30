"""OpenAI list pricing, USD per million tokens, for cost attribution on stored runs.

Recorded here rather than estimated ad hoc so that a cost figure in the paper can be
traced to the table it was computed from. Spend is never estimated from list prices
*instead of* measured usage; this module multiplies prices by the token counts actually
returned by the API.

Long-context rates apply once a request's input exceeds LONG_CONTEXT_THRESHOLD tokens.
ARA runs realtime (its six queries are sequentially dependent), so it pays STANDARD, not
BATCH. Stage 0 and Stage 1 run through the Batch API and pay BATCH.

Captured 2026-09-18.
"""

from __future__ import annotations

LONG_CONTEXT_THRESHOLD = 128_000

PRICING_CAPTURED = "2026-09-18"

# model -> tier -> (short_in, short_out, long_in, long_out), USD per 1M tokens
PRICES: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "gpt-6-astra": {
        "standard": (10.00, 50.00, 20.00, 75.00),
        "batch": (5.00, 25.00, 10.00, 37.50),
    },
    "gpt-5.6-sol": {
        "standard": (4.00, 20.00, 8.00, 30.00),
        "batch": (2.00, 10.00, 4.00, 15.00),
    },
    "gpt-5.6-terra": {
        "standard": (2.00, 12.00, 4.00, 18.00),
        "batch": (1.00, 6.00, 2.00, 9.00),
    },
    "gpt-5.6-luna": {
        "standard": (0.20, 1.20, 0.40, 1.80),
        "batch": (0.10, 0.60, 0.20, 0.90),
    },
}

# Cached-input and cache-write rates, USD per 1M tokens: (short_cached, short_write,
# long_cached, long_write). These dominate ARA: it sends the same PDF to six sequential
# queries, so queries 2-6 and every later repetition read from the prompt cache. Measured
# on the 3-repetition smoke: 63% of input tokens cached, 37% cache-write. Note a cache
# write costs MORE than an uncached read ($2.50 vs $2.00 per 1M on terra), so a single
# repetition pays the write and never amortises it; repetitions are cheaper per unit.
CACHE_PRICES: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "gpt-6-astra": {
        "standard": (1.00, 12.50, 2.00, 25.00),
        "batch": (0.50, 6.25, 1.00, 12.50),
    },
    "gpt-5.6-sol": {
        "standard": (0.40, 5.00, 0.80, 10.00),
        "batch": (0.20, 2.50, 0.40, 5.00),
    },
    "gpt-5.6-terra": {
        "standard": (0.20, 2.50, 0.40, 5.00),
        "batch": (0.10, 1.25, 0.20, 2.50),
    },
    "gpt-5.6-luna": {
        "standard": (0.02, 0.25, 0.04, 0.50),
        "batch": (0.01, 0.125, 0.02, 0.25),
    },
}


class UnknownModelError(KeyError):
    """Raised for a model with no recorded pricing, rather than guessing a rate."""


def run_cost_usd(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    *,
    tier: str = "standard",
) -> float | None:
    """Cost of one request. Returns None when usage was not recorded.

    The long-context rate is chosen on input tokens, which is how the published table is
    indexed. A request is billed wholly at one tier; there is no per-token crossover.
    """
    if input_tokens is None and output_tokens is None:
        return None
    try:
        short_in, short_out, long_in, long_out = PRICES[model][tier]
    except KeyError as exc:
        raise UnknownModelError(
            f"No recorded pricing for model={model!r} tier={tier!r}. Add it to "
            f"pricing.py rather than assuming a rate."
        ) from exc
    tokens_in = int(input_tokens or 0)
    tokens_out = int(output_tokens or 0)
    long_context = tokens_in > LONG_CONTEXT_THRESHOLD
    rate_in = long_in if long_context else short_in
    rate_out = long_out if long_context else short_out
    return (tokens_in * rate_in + tokens_out * rate_out) / 1_000_000.0


def is_long_context(input_tokens: int | None) -> bool:
    return int(input_tokens or 0) > LONG_CONTEXT_THRESHOLD


def request_cost_usd(
    model: str,
    usage: dict,
    *,
    tier: str = "standard",
) -> float:
    """Cost of one request from its raw API usage block, honouring the prompt cache.

    Reads `input_tokens_details.cached_tokens` and `.cache_write_tokens` and bills each
    class at its own rate. Uncached input is the remainder. `output_tokens` already
    includes `output_tokens_details.reasoning_tokens`, which are billed as output, so
    reasoning must not be added again.
    """
    try:
        short_in, short_out, long_in, long_out = PRICES[model][tier]
        short_cached, short_write, long_cached, long_write = CACHE_PRICES[model][tier]
    except KeyError as exc:
        raise UnknownModelError(
            f"No recorded pricing for model={model!r} tier={tier!r}."
        ) from exc

    total_in = int(usage.get("input_tokens") or 0)
    details = usage.get("input_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    written = int(details.get("cache_write_tokens") or 0)
    fresh = max(total_in - cached - written, 0)
    out = int(usage.get("output_tokens") or 0)

    long_context = total_in > LONG_CONTEXT_THRESHOLD
    rate_in = long_in if long_context else short_in
    rate_out = long_out if long_context else short_out
    rate_cached = long_cached if long_context else short_cached
    rate_write = long_write if long_context else short_write
    return (
        fresh * rate_in + cached * rate_cached + written * rate_write + out * rate_out
    ) / 1_000_000.0


def assessment_cost_usd(
    model: str,
    per_query: list[dict] | None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    *,
    responses: list[dict] | None = None,
    tier: str = "standard",
) -> float | None:
    """Cost of a multi-request assessment, billed per request.

    An ARA assessment is six sequential Responses calls of roughly 25k input tokens each.
    The long-context rate keys on a *single request's* input, so summing the six first and
    then choosing a rate overstates the bill whenever the total crosses 128k while no
    individual call does -- which is the normal case here. Falls back to the aggregate only
    when the per-query breakdown is missing, and says so by returning the cruder number.
    """
    if responses:
        # Best source: raw per-request usage, which carries the cache breakdown.
        blocks = [
            (e.get("response") or {}).get("usage")
            for e in responses
            if (e.get("response") or {}).get("usage")
        ]
        if blocks:
            return sum(request_cost_usd(model, u, tier=tier) for u in blocks)
    if per_query:
        # `per_query` has no cache breakdown, so this overstates a cached run.
        return sum(
            run_cost_usd(
                model, q.get("input_tokens"), q.get("output_tokens"), tier=tier
            )
            or 0.0
            for q in per_query
        )
    return run_cost_usd(model, input_tokens, output_tokens, tier=tier)
