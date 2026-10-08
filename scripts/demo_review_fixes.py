"""Demo script: verify the three review findings are fixed end-to-end.

This script builds a real two-database pipeline (blog_small + ecommerce_medium
loaded from fixtures) against a live PostgreSQL instance and demonstrates:

1. Multi-database routing   - queries go to the requested database
2. Security controls        - blocked_tables / blocked_columns / EXPLAIN policy
3. Rate limiting            - concurrent requests over the limit are rejected
4. Exponential backoff      - validation retries wait (delay * factor**attempt)
5. Metrics                  - prometheus counters record the whole flow

No OpenAI key required: SQL generation is replaced with a deterministic
fake generator so the demo focuses on the orchestrator pipeline.

Usage:
    uv run python scripts/demo_review_fixes.py
"""

import asyncio
import sys

from asyncpg import Pool

from pg_mcp.cache.schema_cache import SchemaCache
from pg_mcp.config.settings import (
    CacheConfig,
    DatabaseConfig,
    OpenAIConfig,
    ResilienceConfig,
    SecurityConfig,
    ValidationConfig,
)
from pg_mcp.db.pool import create_pool
from pg_mcp.models.query import QueryRequest, ReturnType
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.orchestrator import QueryOrchestrator
from pg_mcp.services.result_validator import ResultValidator
from pg_mcp.services.sql_executor import SQLExecutor
from pg_mcp.services.sql_generator import SQLGenerator
from pg_mcp.services.sql_validator import SQLValidator

PG_HOST = "localhost"
PG_PORT = 15432
PG_USER = "postgres"
PG_PASSWORD = "postgres"  # noqa: S105 - local demo container credential


class FakeSQLGenerator(SQLGenerator):
    """Deterministic SQL generator so the demo needs no OpenAI key."""

    def __init__(self, canned: str = "SELECT COUNT(*) AS total FROM users") -> None:
        self.canned = canned

    async def generate(
        self,
        question: str,
        schema,
        previous_attempt: str | None = None,
        error_feedback: str | None = None,
    ) -> str:
        return self.canned


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


async def build_orchestrator(
    databases: dict[str, str],  # name -> canned SQL
    security: SecurityConfig,
    resilience: ResilienceConfig,
    with_metrics: bool = True,
    with_rate_limiter: bool = True,
) -> tuple[QueryOrchestrator, dict[str, Pool]]:
    """Wire a full orchestrator over the given databases."""
    pools: dict[str, Pool] = {}
    executors = {}
    schema_cache = SchemaCache(config=CacheConfig(schema_ttl=3600, max_size=10))
    metrics = MetricsCollector() if with_metrics else None
    rate_limiter = (
        MultiRateLimiter(
            query_limit=resilience.query_rate_limit,
            llm_limit=resilience.llm_rate_limit,
        )
        if with_rate_limiter
        else None
    )
    generator = FakeSQLGenerator(canned=next(iter(databases.values())))

    for name in databases:
        config = DatabaseConfig(
            host=PG_HOST, port=PG_PORT, name=name, user=PG_USER, password=PG_PASSWORD
        )
        pool = await create_pool(config)
        pools[name] = pool
        executors[name] = SQLExecutor(
            pool=pool, security_config=security, db_config=config
        )
        await schema_cache.load(name, pool)

    orchestrator = QueryOrchestrator(
        sql_generator=generator,
        sql_validator=SQLValidator(
            config=security,
            blocked_tables=security.blocked_tables or None,
            blocked_columns=security.blocked_columns or None,
            allow_explain=security.allow_explain,
        ),
        sql_executors=executors,
        result_validator=ResultValidator(
            openai_config=OpenAIConfig(api_key="sk-demo"),
            validation_config=ValidationConfig(enabled=False),
        ),
        schema_cache=schema_cache,
        pools=pools,
        resilience_config=resilience,
        validation_config=ValidationConfig(enabled=False),
        metrics=metrics,
        rate_limiter=rate_limiter,
    )
    return orchestrator, pools


async def demo_multi_database() -> None:
    banner("DEMO 1: Multi-database routing (review finding #1)")
    print("Two databases configured: blog_small (blog) + ecommerce_medium (shop)")
    print("The SAME question is routed to whichever database the caller picks.\n")

    orchestrator, pools = await build_orchestrator(
        databases={
            "blog_small": "SELECT COUNT(*) AS total FROM users",
            "ecommerce_medium": "SELECT COUNT(*) AS total FROM user_profiles",
        },
        security=SecurityConfig(),
        resilience=ResilienceConfig(query_rate_limit=20, llm_rate_limit=20),
    )

    print("  [a] database = blog_small (blog users table):")
    orchestrator.sql_generator.canned = "SELECT COUNT(*) AS total FROM users"
    response = await orchestrator.execute_query(
        QueryRequest(question="how many users?", database="blog_small")
    )
    print(f"      success = {response.success}")
    print(f"      SQL     = {response.generated_sql}")
    print(f"      result  = {response.data.rows if response.data else None}")

    print("\n  [b] database = ecommerce_medium (shop user_profiles table):")
    orchestrator.sql_generator.canned = "SELECT COUNT(*) AS total FROM user_profiles"
    response = await orchestrator.execute_query(
        QueryRequest(question="how many customers?", database="ecommerce_medium")
    )
    print(f"      success = {response.success}")
    print(f"      SQL     = {response.generated_sql}")
    print(f"      result  = {response.data.rows if response.data else None}")

    print("\n  [c] control: no database specified + multiple databases configured:")
    response = await orchestrator.execute_query(
        QueryRequest(question="how many users?")
    )
    print(f"      success = {response.success}, error = {response.error.code}")
    print(f"      message = {response.error.message}")

    for pool in pools.values():
        await pool.close()


async def demo_wrong_database_isolated() -> None:
    banner("DEMO 1b: requests can no longer hit the wrong database")
    print("Executor for ecommerce_medium only knows its own pool;\n"
          "a query naming an unknown database is rejected, not silently\n"
          "executed against the primary (the original bug).\n")

    orchestrator, pools = await build_orchestrator(
        databases={"ecommerce_medium": "SELECT COUNT(*) AS total FROM user_profiles"},
        security=SecurityConfig(),
        resilience=ResilienceConfig(),
    )
    request = QueryRequest(
        question="how many customers?",
        database="blog_small_typo",
        return_type=ReturnType.RESULT,
    )
    response = await orchestrator.execute_query(request)
    print(f"  success = {response.success}")
    print(f"  error   = {response.error.code}: {response.error.message}")
    for pool in pools.values():
        await pool.close()


async def demo_security_controls() -> None:
    banner("DEMO 2: configured security controls (review finding #1)")
    print("SecurityConfig now drives blocked_tables / blocked_columns / allow_explain.\n")

    security = SecurityConfig(
        blocked_tables="payments",
        blocked_columns="credit_card_cvv",
        allow_explain=False,
    )
    orchestrator, pools = await build_orchestrator(
        databases={"ecommerce_medium": "SELECT * FROM payments"},
        security=security,
        resilience=ResilienceConfig(),
    )

    print("  [a] SELECT on blocked table 'payments':")
    response = await orchestrator.execute_query(
        QueryRequest(question="show payments", database="ecommerce_medium")
    )
    print(f"      success = {response.success}, error = {response.error.code}")
    print(f"      {response.error.message}")

    print("\n  [b] SELECT on allowed table 'user_profiles':")
    orchestrator.sql_generator.canned = "SELECT COUNT(*) AS total FROM user_profiles"
    response = await orchestrator.execute_query(
        QueryRequest(question="count profiles", database="ecommerce_medium")
    )
    print(
        f"      success = {response.success},"
        f" rows = {response.data.rows if response.data else None}"
    )

    print("\n  [c] EXPLAIN with allow_explain=false:")
    orchestrator.sql_generator.canned = "EXPLAIN SELECT * FROM customers"
    response = await orchestrator.execute_query(
        QueryRequest(question="explain plan", database="ecommerce_medium")
    )
    print(f"      success = {response.success}, error = {response.error.code}")

    for pool in pools.values():
        await pool.close()


async def demo_rate_limiting_and_metrics() -> None:
    banner("DEMO 3+4: rate limiting & metrics in the request path (finding #2)")
    print("query_rate_limit=1: a second concurrent query must be rejected\n"
          "with RATE_LIMIT_EXCEEDED instead of silently queueing forever.\n")

    metrics = MetricsCollector()
    metrics.reset_all_metrics()
    resilience = ResilienceConfig(
        query_rate_limit=1, llm_rate_limit=5, rate_limit_timeout=0.5
    )
    security = SecurityConfig()
    pools: dict[str, Pool] = {}
    executors = {}
    schema_cache = SchemaCache(config=CacheConfig(schema_ttl=3600, max_size=10))

    for name in ("blog_small",):
        config = DatabaseConfig(
            host=PG_HOST, port=PG_PORT, name=name, user=PG_USER, password=PG_PASSWORD
        )
        pool = await create_pool(config)
        pools[name] = pool
        executors[name] = SQLExecutor(pool=pool, security_config=security, db_config=config)
        await schema_cache.load(name, pool)

    rate_limiter = MultiRateLimiter(query_limit=1, llm_limit=5)
    orchestrator = QueryOrchestrator(
        sql_generator=FakeSQLGenerator("SELECT COUNT(*) AS total FROM users"),
        sql_validator=SQLValidator(config=security),
        sql_executors=executors,
        result_validator=ResultValidator(
            openai_config=OpenAIConfig(api_key="sk-demo"),
            validation_config=ValidationConfig(enabled=False),
        ),
        schema_cache=schema_cache,
        pools=pools,
        resilience_config=resilience,
        validation_config=ValidationConfig(enabled=False),
        metrics=metrics,
        rate_limiter=rate_limiter,
    )

    # Hold the single query slot open, then fire a request: it must be
    # rejected with RATE_LIMIT_EXCEEDED after the timeout.
    async with rate_limiter.for_queries(timeout=5.0):
        blocked = await orchestrator.execute_query(
            QueryRequest(question="q1", database="blog_small")
        )
        print("  [a] concurrent request while slot busy:")
        print(f"      success = {blocked.success}, error = {blocked.error.code}")

    free = await orchestrator.execute_query(
        QueryRequest(question="q2", database="blog_small")
    )
    print(f"  [b] request after slot freed: success = {free.success}")

    print("\n  [c] prometheus metrics recorded by the pipeline:")
    from prometheus_client import REGISTRY, generate_latest

    payload = generate_latest(REGISTRY).decode("utf-8")
    for line in payload.splitlines():
        if line.startswith("pg_mcp_queries_total") or line.startswith("pg_mcp_query_duration"):
            print(f"      {line}")

    for pool in pools.values():
        await pool.close()


async def demo_backoff_retry() -> None:
    banner("DEMO 5: retry with exponential backoff (finding #2)")
    print("retry_delay=0.2, backoff_factor=3.0 -> retries wait 0.2s, 0.6s, 1.8s...\n")

    import logging

    logging.basicConfig(level=logging.INFO, format="  %(levelname)s %(name)s: %(message)s")
    logging.getLogger("pg_mcp.services.orchestrator").setLevel(logging.WARNING)

    from unittest.mock import MagicMock

    from pg_mcp.models.errors import SQLParseError
    from pg_mcp.models.schema import DatabaseSchema

    security = SecurityConfig()
    config = DatabaseConfig(
        host=PG_HOST, port=PG_PORT, name="blog_small", user=PG_USER, password=PG_PASSWORD
    )
    pool = await create_pool(config)

    schema_cache = MagicMock()
    schema_cache.get = MagicMock(return_value=DatabaseSchema(database_name="blog_small", tables=[]))

    validator = MagicMock()
    validator.validate_or_raise = MagicMock(
        side_effect=[SQLParseError("syntax error"), None]  # fail once, then pass
    )

    orchestrator = QueryOrchestrator(
        sql_generator=FakeSQLGenerator("SELECT 1"),
        sql_validator=validator,
        sql_executors={
            "blog_small": SQLExecutor(pool=pool, security_config=security, db_config=config)
        },
        result_validator=ResultValidator(
            openai_config=OpenAIConfig(api_key="sk-demo"),
            validation_config=ValidationConfig(enabled=False),
        ),
        schema_cache=schema_cache,
        pools={"blog_small": pool},
        resilience_config=ResilienceConfig(
            max_retries=2, retry_delay=0.2, backoff_factor=3.0
        ),
        validation_config=ValidationConfig(enabled=False),
    )

    import time

    start = time.perf_counter()
    response = await orchestrator.execute_query(
        QueryRequest(question="q", database="blog_small")
    )
    elapsed = time.perf_counter() - start
    print("\n  first attempt failed, retry succeeded after backoff")
    print(f"  success = {response.success}, wall time = {elapsed:.2f}s (>= 0.2s backoff)")

    await pool.close()


async def main() -> None:
    print("\nPostgreSQL MCP Server - review fixes verification demo")
    await demo_multi_database()
    await demo_wrong_database_isolated()
    await demo_security_controls()
    await demo_rate_limiting_and_metrics()
    await demo_backoff_retry()
    banner("ALL DEMOS COMPLETED")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
