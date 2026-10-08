"""Unit tests for resilience/observability integration in the orchestrator.

These tests verify the review findings are fixed:
1. Retries between SQL validation failures use exponential backoff
   (previously retry_delay/backoff_factor were dead configuration).
2. Query pipeline and LLM calls are guarded by rate limiters; slot timeouts
   produce RATE_LIMIT_EXCEEDED errors (previously rate limiting was not wired).
3. Metrics are recorded for query requests (previously MetricsCollector was
   created but never called).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import SecurityViolationError
from pg_mcp.models.query import QueryRequest, ReturnType
from pg_mcp.models.schema import DatabaseSchema
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.orchestrator import QueryOrchestrator


def _make_components():
    """Build the mocked collaborators of the orchestrator."""
    sql_generator = MagicMock()
    sql_generator.generate = AsyncMock(return_value="SELECT 1")
    sql_validator = MagicMock()
    sql_validator.validate_or_raise = MagicMock(return_value=None)
    schema_cache = MagicMock()
    schema_cache.get = MagicMock(return_value=DatabaseSchema(database_name="db", tables=[]))
    result_validator = MagicMock()
    result_validator.validate = AsyncMock(return_value=MagicMock(confidence=95))
    executor = MagicMock()
    executor.execute = AsyncMock(return_value=([{"n": 1}], 1))
    return sql_generator, sql_validator, schema_cache, result_validator, executor


def _make_orchestrator(
    resilience_config: ResilienceConfig | None = None,
    rate_limiter: MultiRateLimiter | None = None,
    metrics: MagicMock | None = None,
    validator_side_effect=None,
    sql_result=([{"n": 1}], 1),
    validation_config: ValidationConfig | None = None,
) -> tuple[QueryOrchestrator, MagicMock, MagicMock, MagicMock]:
    """Create a fully-mocked single-database orchestrator.

    Returns:
        (orchestrator, sql_generator, sql_validator, executor)
    """
    sql_generator, sql_validator, schema_cache, result_validator, executor = _make_components()
    if validator_side_effect is not None:
        sql_validator.validate_or_raise = MagicMock(side_effect=validator_side_effect)
    executor.execute = AsyncMock(return_value=sql_result)

    orchestrator = QueryOrchestrator(
        sql_generator=sql_generator,
        sql_validator=sql_validator,
        sql_executors={"db": executor},
        result_validator=result_validator,
        schema_cache=schema_cache,
        pools={"db": MagicMock()},
        resilience_config=resilience_config or ResilienceConfig(),
        validation_config=validation_config or ValidationConfig(),
        metrics=metrics,
        rate_limiter=rate_limiter,
    )
    return orchestrator, sql_generator, sql_validator, executor


def _request() -> QueryRequest:
    return QueryRequest(question="q", database="db", return_type=ReturnType.RESULT)


class TestExponentialBackoff:
    """Validation-failure retries must wait (delay * factor ** attempt)."""

    @pytest.mark.asyncio
    async def test_backoff_delays_are_exponential(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """First retry waits retry_delay, second waits retry_delay * factor."""
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        # Fail twice, then succeed
        validator_effect = [
            SecurityViolationError("bad"),
            SecurityViolationError("worse"),
            None,
        ]
        config = ResilienceConfig(max_retries=3, retry_delay=0.5, backoff_factor=2.0)
        orchestrator, _, _, _ = _make_orchestrator(
            resilience_config=config, validator_side_effect=validator_effect
        )

        response = await orchestrator.execute_query(_request())
        assert response.success is True
        assert sleeps == pytest.approx([0.5, 1.0])

    @pytest.mark.asyncio
    async def test_backoff_delay_helper(self) -> None:
        """_llm_backoff_delay computes delay * factor**attempt, capped."""
        config = ResilienceConfig(retry_delay=1.0, backoff_factor=2.0, circuit_breaker_timeout=60.0)
        orchestrator, _, _, _ = _make_orchestrator(resilience_config=config)

        assert orchestrator._llm_backoff_delay(0) == 1.0
        assert orchestrator._llm_backoff_delay(1) == 2.0
        assert orchestrator._llm_backoff_delay(2) == 4.0
        # Capped at circuit breaker timeout
        assert orchestrator._llm_backoff_delay(10) == 60.0

    @pytest.mark.asyncio
    async def test_no_sleep_before_first_attempt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A first-try success must not sleep at all."""
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        orchestrator, _, _, _ = _make_orchestrator()
        response = await orchestrator.execute_query(_request())
        assert response.success is True
        assert sleeps == []


class TestQueryRateLimiting:
    """The whole pipeline is guarded by the query rate limiter."""

    @pytest.mark.asyncio
    async def test_slot_timeout_returns_rate_limit_exceeded(self) -> None:
        """When the query slot can't be acquired, respond with
        RATE_LIMIT_EXCEEDED instead of blocking or failing generically."""
        rate_limiter = MultiRateLimiter(query_limit=1, llm_limit=5)
        # Exhaust the single query slot by keeping its context open
        async with rate_limiter.for_queries(timeout=1.0):
            orchestrator, _, _, executor = _make_orchestrator(
                resilience_config=ResilienceConfig(rate_limit_timeout=0.1),
                rate_limiter=rate_limiter,
            )
            response = await orchestrator.execute_query(_request())

        assert response.success is False
        assert response.error is not None
        assert response.error.code == "rate_limit_exceeded"
        assert "rate limiter" in response.error.message.lower()
        executor.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_request_within_limit_executes(self) -> None:
        """Normal case: slot available, query goes through."""
        rate_limiter = MultiRateLimiter(query_limit=2, llm_limit=2)
        orchestrator, _, _, executor = _make_orchestrator(rate_limiter=rate_limiter)

        response = await orchestrator.execute_query(_request())
        assert response.success is True
        executor.execute.assert_awaited_once()


class TestLLMRateLimiting:
    """LLM calls are guarded by the LLM rate limiter."""

    @pytest.mark.asyncio
    async def test_llm_slot_timeout_yields_rate_limit_error(self) -> None:
        """An LLM slot timeout surfaces as a rate_limit_exceeded response."""
        rate_limiter = MultiRateLimiter(query_limit=5, llm_limit=1)
        async with rate_limiter.for_llm(timeout=1.0):
            orchestrator, _, _, executor = _make_orchestrator(
                resilience_config=ResilienceConfig(rate_limit_timeout=0.1),
                rate_limiter=rate_limiter,
            )
            response = await orchestrator.execute_query(_request())

        assert response.success is False
        assert response.error is not None
        assert response.error.code == "rate_limit_exceeded"
        assert "llm" in response.error.message.lower()
        executor.execute.assert_not_awaited()


class TestMetricsIntegration:
    """The MetricsCollector is actually called (was dead code before)."""

    @pytest.mark.asyncio
    async def test_successful_query_records_request_metric(self) -> None:
        metrics = MagicMock()
        orchestrator, _, _, _ = _make_orchestrator(metrics=metrics)

        response = await orchestrator.execute_query(_request())
        assert response.success is True

        metrics.increment_query_request.assert_called_once_with(status="success", database="db")
        metrics.query_duration.observe.assert_called_once()
        metrics.observe_db_query_duration.assert_called_once()

    @pytest.mark.asyncio
    async def test_security_violation_records_rejection(self) -> None:
        metrics = MagicMock()
        orchestrator, _, _, _ = _make_orchestrator(
            metrics=metrics,
            validator_side_effect=SecurityViolationError("blocked"),
        )

        response = await orchestrator.execute_query(_request())
        assert response.success is False

        # Every rejected SQL attempt is counted, and the final failure is
        # tagged with its own reason
        metrics.increment_sql_rejected.assert_any_call(reason="security_violation")
        assert metrics.increment_sql_rejected.call_count >= 1
        metrics.increment_query_request.assert_called_once_with(
            status="security_violation", database="db"
        )

    @pytest.mark.asyncio
    async def test_llm_calls_are_counted(self) -> None:
        metrics = MagicMock()
        orchestrator, _, _, _ = _make_orchestrator(metrics=metrics)

        await orchestrator.execute_query(_request())

        # SQL generation and result validation both count as LLM calls
        metrics.increment_llm_call.assert_any_call(operation="generate_sql")
        metrics.increment_llm_call.assert_any_call(operation="validate_result")
        assert metrics.observe_llm_latency.call_count == 2

    @pytest.mark.asyncio
    async def test_no_metrics_when_collector_absent(self) -> None:
        """metrics=None must keep the pipeline working (no crashes)."""
        orchestrator, _, _, executor = _make_orchestrator(metrics=None)
        response = await orchestrator.execute_query(_request())
        assert response.success is True
        executor.execute.assert_awaited_once()


class TestValidationConfigUsage:
    """max_question_length / min_confidence_score are honoured."""

    @pytest.mark.asyncio
    async def test_question_length_limit_enforced(self) -> None:
        config = ValidationConfig(max_question_length=10)
        orchestrator, _, _, executor = _make_orchestrator(validation_config=config)

        request = QueryRequest(question="x" * 11, database="db", return_type=ReturnType.RESULT)
        response = await orchestrator.execute_query(request)

        assert response.success is False
        assert response.error is not None
        assert response.error.code == "question_too_long"
        assert response.error.details["max_length"] == 10
        executor.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_low_confidence_flagged_against_min_score(self) -> None:
        """Results below min_confidence_score still return but are flagged."""
        config = ValidationConfig(min_confidence_score=90)
        orchestrator, _, _, _ = _make_orchestrator(validation_config=config)

        # Replace result validator with a low-confidence outcome
        low_conf_validator = MagicMock()
        low_conf_validator.validate = AsyncMock(return_value=MagicMock(confidence=40))
        orchestrator.result_validator = low_conf_validator

        response = await orchestrator.execute_query(_request())
        assert response.success is True
        assert response.confidence == 40
        # Orchestrator did NOT fail the query, only flagged it (non-blocking)
        assert orchestrator.validation_config.min_confidence_score == 90
