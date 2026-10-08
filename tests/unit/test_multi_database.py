"""Unit tests for multi-database support and routing.

These tests verify that:
1. Requests are routed to the SQLExecutor registered for the resolved
   database (the core multi-database fix from the review).
2. Invalid database names are rejected.
3. Missing database specification with multiple databases errors out.
4. Legacy single-executor mode still works.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import DatabaseError
from pg_mcp.models.query import QueryRequest, ReturnType
from pg_mcp.models.schema import DatabaseSchema
from pg_mcp.services.orchestrator import QueryOrchestrator


def _make_schema(db_name: str) -> DatabaseSchema:
    """Build a minimal DatabaseSchema for mocking."""
    return DatabaseSchema(
        database_name=db_name,
        tables=[],
    )


def _make_orchestrator(
    executors: dict[str, MagicMock],
    pools: dict[str, MagicMock],
    **kwargs,
) -> QueryOrchestrator:
    """Create an orchestrator with per-database executors and mocks."""
    sql_generator = MagicMock()
    sql_generator.generate = AsyncMock(return_value="SELECT 1")
    sql_validator = MagicMock()
    sql_validator.validate_or_raise = MagicMock(return_value=None)
    schema_cache = MagicMock()
    schema_cache.get = MagicMock(return_value=_make_schema("db"))
    result_validator = MagicMock()
    result_validator.validate = AsyncMock(return_value=MagicMock(confidence=95))

    defaults: dict = {
        "sql_generator": sql_generator,
        "sql_validator": sql_validator,
        "sql_executors": executors,
        "result_validator": result_validator,
        "schema_cache": schema_cache,
        "pools": pools,
        "resilience_config": kwargs.pop("resilience_config", ResilienceConfig()),
        "validation_config": kwargs.pop("validation_config", ValidationConfig()),
    }
    defaults.update(kwargs)
    return QueryOrchestrator(**defaults)


class TestMultiDatabaseRouting:
    """Test that queries execute against the requested database."""

    @pytest.mark.asyncio
    async def test_routes_to_requested_database_executor(self) -> None:
        """A request with database='db_b' must run on db_b's executor."""
        exec_a = MagicMock()
        exec_a.execute = AsyncMock(return_value=([], 0))
        exec_b = MagicMock()
        exec_b.execute = AsyncMock(return_value=([{"n": 1}], 1))

        pools = {"db_a": MagicMock(), "db_b": MagicMock()}
        orchestrator = _make_orchestrator(executors={"db_a": exec_a, "db_b": exec_b}, pools=pools)

        request = QueryRequest(
            question="count rows", database="db_b", return_type=ReturnType.RESULT
        )
        response = await orchestrator.execute_query(request)

        assert response.success is True
        exec_b.execute.assert_awaited_once()
        exec_a.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_routes_each_request_to_its_own_database(self) -> None:
        """Consecutive requests to different databases hit different executors."""
        exec_a = MagicMock()
        exec_a.execute = AsyncMock(return_value=([], 0))
        exec_b = MagicMock()
        exec_b.execute = AsyncMock(return_value=([], 0))

        pools = {"db_a": MagicMock(), "db_b": MagicMock()}
        orchestrator = _make_orchestrator(executors={"db_a": exec_a, "db_b": exec_b}, pools=pools)

        for db, executor in (("db_a", exec_a), ("db_b", exec_b)):
            request = QueryRequest(question="q", database=db, return_type=ReturnType.RESULT)
            response = await orchestrator.execute_query(request)
            assert response.success is True
            executor.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unknown_database_rejected(self) -> None:
        """A request naming a database without a pool fails cleanly."""
        exec_a = MagicMock()
        pools = {"db_a": MagicMock()}
        orchestrator = _make_orchestrator(executors={"db_a": exec_a}, pools=pools)

        request = QueryRequest(question="q", database="db_zzz", return_type=ReturnType.RESULT)
        response = await orchestrator.execute_query(request)

        assert response.success is False
        assert response.error is not None
        assert response.error.code == "database_error"
        assert "db_zzz" in response.error.message
        exec_a.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_executor_is_internal_error(self) -> None:
        """A pool without a matching executor is a configuration error."""
        exec_a = MagicMock()
        pools = {"db_a": MagicMock(), "db_b": MagicMock()}
        orchestrator = _make_orchestrator(executors={"db_a": exec_a}, pools=pools)

        request = QueryRequest(question="q", database="db_b", return_type=ReturnType.RESULT)
        response = await orchestrator.execute_query(request)

        assert response.success is False
        assert response.error is not None
        assert "no sql executor configured" in response.error.message.lower()

    @pytest.mark.asyncio
    async def test_multiple_databases_require_specification(self) -> None:
        """With several databases, omitting `database` must fail (not guess)."""
        exec_a = MagicMock()
        exec_b = MagicMock()
        pools = {"db_a": MagicMock(), "db_b": MagicMock()}
        orchestrator = _make_orchestrator(executors={"db_a": exec_a, "db_b": exec_b}, pools=pools)

        request = QueryRequest(question="q", return_type=ReturnType.RESULT)
        response = await orchestrator.execute_query(request)

        assert response.success is False
        assert response.error is not None
        assert "multiple databases" in response.error.message.lower()
        exec_a.execute.assert_not_called()
        exec_b.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_database_auto_selected(self) -> None:
        """With exactly one database, omitting `database` auto-selects it."""
        executor = MagicMock()
        executor.execute = AsyncMock(return_value=([], 0))
        pools = {"only": MagicMock()}
        orchestrator = _make_orchestrator(executors={"only": executor}, pools=pools)

        request = QueryRequest(question="q", return_type=ReturnType.RESULT)
        response = await orchestrator.execute_query(request)

        assert response.success is True
        executor.execute.assert_awaited_once()


class TestLegacySingleExecutorMode:
    """The deprecated single-executor argument keeps working."""

    @pytest.mark.asyncio
    async def test_single_executor_used_for_any_database(self) -> None:
        """Legacy mode routes every database to the single executor."""
        executor = MagicMock()
        executor.execute = AsyncMock(return_value=([], 0))
        pools = {"db_a": MagicMock(), "db_b": MagicMock()}

        orchestrator = _make_orchestrator(
            executors={},  # unused
            pools=pools,
            sql_executor=executor,
        )

        for db in ("db_a", "db_b"):
            request = QueryRequest(question="q", database=db, return_type=ReturnType.RESULT)
            response = await orchestrator.execute_query(request)
            assert response.success is True

        assert executor.execute.await_count == 2

    def test_requires_executor_argument(self) -> None:
        """Constructing without any executor raises a clear error."""
        with pytest.raises(ValueError, match="sql_executor"):
            QueryOrchestrator(
                sql_generator=MagicMock(),
                sql_validator=MagicMock(),
                result_validator=MagicMock(),
                schema_cache=MagicMock(),
                pools={"db": MagicMock()},
            )


class TestResolverExecutorConsistency:
    """_resolve_executor honours routing rules."""

    def test_legacy_mode_returns_single_executor(self) -> None:
        executor = MagicMock()
        orchestrator = _make_orchestrator(executors={}, pools={}, sql_executor=executor)
        assert orchestrator._resolve_executor("any_db") is executor

    def test_multi_mode_unknown_database_raises(self) -> None:
        orchestrator = _make_orchestrator(executors={"db": MagicMock()}, pools={"db": MagicMock()})
        with pytest.raises(DatabaseError):
            orchestrator._resolve_executor("missing")
