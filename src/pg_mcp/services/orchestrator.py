"""Query orchestrator for coordinating the complete query flow.

This module provides the QueryOrchestrator class that coordinates all components
of the query processing pipeline: SQL generation, validation, execution, and result
validation.

Per the review, this version integrates the previously dormant resilience and
observability layers into the actual request flow:

- **Multi-database routing**: requests are executed by the ``SQLExecutor``
  registered for the *resolved* database (previously every request hit the
  primary executor regardless of the requested database).
- **Rate limiting**: the whole pipeline is guarded by a query rate limiter and
  LLM calls by an LLM rate limiter (from ``MultiRateLimiter``).
- **Retry with exponential backoff**: validation-failure retries now sleep
  ``retry_delay * backoff_factor ** attempt`` seconds between attempts.
- **Metrics**: query requests, durations, SQL rejections, LLM calls/latency and
  token usage are recorded through ``MetricsCollector``.
- **Tracing**: the pipeline runs inside a ``request_context`` so the request ID
  propagates through all async operations.
"""

import asyncio
import logging
import time
import uuid
from typing import Any

from asyncpg import Pool

from pg_mcp.cache.schema_cache import SchemaCache
from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import (
    DatabaseError,
    ErrorCode,
    LLMError,
    PgMcpError,
    RateLimitExceededError,
    SchemaLoadError,
    SecurityViolationError,
    SQLParseError,
)
from pg_mcp.models.query import (
    ErrorDetail,
    QueryRequest,
    QueryResponse,
    QueryResult,
    ReturnType,
    ValidationResult,
)
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.observability.tracing import request_context
from pg_mcp.resilience.circuit_breaker import CircuitBreaker
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.result_validator import ResultValidator
from pg_mcp.services.sql_executor import SQLExecutor
from pg_mcp.services.sql_generator import SQLGenerator
from pg_mcp.services.sql_validator import SQLValidator

logger = logging.getLogger(__name__)


class QueryOrchestrator:
    """Orchestrates the complete query processing pipeline.

    This class coordinates SQL generation, validation, execution, and result
    validation. It implements retry logic with exponential backoff and error
    feedback, the circuit breaker pattern for fault tolerance, rate limiting,
    metrics collection, and comprehensive error handling.

    Example:
        >>> orchestrator = QueryOrchestrator(
        ...     sql_generator=generator,
        ...     sql_validator=validator,
        ...     sql_executors={"mydb": executor},
        ...     result_validator=result_validator,
        ...     schema_cache=cache,
        ...     pools={"mydb": pool},
        ...     resilience_config=resilience_config,
        ...     validation_config=validation_config,
        ...     metrics=metrics,
        ...     rate_limiter=rate_limiter,
        ... )
        >>> response = await orchestrator.execute_query(QueryRequest(
        ...     question="How many users?",
        ...     database="mydb"
        ...  ))
    """

    def __init__(
        self,
        sql_generator: SQLGenerator,
        sql_validator: SQLValidator,
        sql_executor: SQLExecutor | None = None,
        result_validator: ResultValidator | None = None,
        schema_cache: SchemaCache | None = None,
        pools: dict[str, Pool] | None = None,
        resilience_config: ResilienceConfig | None = None,
        validation_config: ValidationConfig | None = None,
        sql_executors: dict[str, SQLExecutor] | None = None,
        metrics: MetricsCollector | None = None,
        rate_limiter: MultiRateLimiter | None = None,
    ) -> None:
        """Initialize query orchestrator.

        Args:
            sql_generator: SQL generation service.
            sql_validator: SQL validation service.
            sql_executor: Single SQL executor (legacy mode; every database is
                routed to it). Deprecated in favour of ``sql_executors``.
            result_validator: Result validation service.
            schema_cache: Schema cache instance.
            pools: Dictionary mapping database names to connection pools.
            resilience_config: Resilience configuration for retries, circuit
                breaker and rate limiting.
            validation_config: Validation configuration including thresholds.
            sql_executors: Executor per database name. When provided, requests
                are routed to the executor matching the resolved database.
            metrics: Optional metrics collector. When ``None``, metrics are
                skipped (useful in tests).
            rate_limiter: Optional multi rate limiter guarding the query
                pipeline and LLM calls. When ``None``, no rate limiting.
        """
        if sql_executors is None and sql_executor is None:
            raise ValueError("Either sql_executor or sql_executors must be provided")

        self.sql_generator = sql_generator
        self.sql_validator = sql_validator
        self.result_validator = result_validator  # type: ignore[assignment]
        self.schema_cache = schema_cache  # type: ignore[assignment]
        self.pools = pools or {}  # type: ignore[assignment]
        self.resilience_config = resilience_config or ResilienceConfig()
        self.validation_config = validation_config or ValidationConfig()

        # Multi-database executors. In legacy single-executor mode every
        # database resolves to the same executor (previous behaviour).
        self.sql_executors = sql_executors if sql_executors is not None else {}
        self._single_executor = sql_executor

        # Observability (optional so tests can run without Prometheus state)
        self.metrics = metrics

        # Rate limiting (optional)
        self.rate_limiter = rate_limiter

        # Create circuit breaker for LLM calls
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.resilience_config.circuit_breaker_threshold,
            recovery_timeout=self.resilience_config.circuit_breaker_timeout,
        )

    # ------------------------------------------------------------------
    # Executor routing
    # ------------------------------------------------------------------

    def _resolve_executor(self, database_name: str) -> SQLExecutor:
        """Return the executor for the given database.

        In multi-executor mode the executor registered for ``database_name`` is
        returned; a missing registration is an internal misconfiguration and
        raises. In legacy single-executor mode the single executor is returned
        for every database.

        Args:
            database_name: Resolved database name (validated against pools).

        Returns:
            SQLExecutor: The executor bound to that database.

        Raises:
            DatabaseError: If no executor is registered for the database.
        """
        if self._single_executor is not None:
            return self._single_executor

        executor = self.sql_executors.get(database_name)
        if executor is None:
            raise DatabaseError(
                message=f"No SQL executor configured for database '{database_name}'",
                details={
                    "database": database_name,
                    "configured_databases": sorted(self.sql_executors.keys()),
                },
            )
        return executor

    # ------------------------------------------------------------------
    # Metrics helpers (no-op when metrics are not wired)
    # ------------------------------------------------------------------

    def _record_query_request(self, status: str, database: str) -> None:
        """Increment the query request counter with the given status."""
        if self.metrics is not None:
            self.metrics.increment_query_request(status=status, database=database)

    def _record_sql_rejected(self, reason: str) -> None:
        """Increment the SQL rejected counter with the given reason."""
        if self.metrics is not None:
            self.metrics.increment_sql_rejected(reason=reason)

    def _record_llm_tokens(self, operation: str, tokens: int) -> None:
        """Record LLM token usage."""
        if self.metrics is not None and tokens > 0:
            self.metrics.increment_llm_tokens(operation=operation, tokens=tokens)

    async def execute_query(self, request: QueryRequest) -> QueryResponse:
        """Execute complete query flow from question to results.

        This method orchestrates the entire pipeline:
        1. Enforce the configured question length limit
        2. Generate request_id and open a tracing context
        3. Acquire a query rate-limiter slot (RATE_LIMIT_EXCEEDED on timeout)
        4. Resolve and validate database name and executor
        5. Load schema from cache
        6. Generate and validate SQL with retry + exponential backoff
        7. Execute SQL via the database-specific executor (if return_type == RESULT)
        8. Validate results (optional)
        9. Record metrics and return structured response

        Args:
            request: Query request containing question and parameters.

        Returns:
            QueryResponse: Complete response with SQL, results, or error information.

        Example:
            >>> response = await orchestrator.execute_query(
            ...     QueryRequest(question="Count all users", return_type="result")
            ... )
            >>> if response.success:
            ...     print(f"Found {response.data.row_count} rows")
        """
        # Step 0: enforce configured question length (uses max_question_length)
        max_len = self.validation_config.max_question_length
        if len(request.question) > max_len:
            logger.warning(
                "Question exceeds configured maximum length",
                extra={"question_length": len(request.question), "max_length": max_len},
            )
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=ErrorCode.QUESTION_TOO_LONG.value,
                    message=(
                        f"Question length {len(request.question)} exceeds the "
                        f"configured maximum of {max_len} characters"
                    ),
                    details={"question_length": len(request.question), "max_length": max_len},
                ),
                confidence=0,
                tokens_used=0,
            )

        # Generate request_id for full-chain tracing
        request_id = str(uuid.uuid4())
        logger.info(
            "Starting query execution",
            extra={"request_id": request_id, "question": request.question[:100]},
        )

        # Rate limiting wraps the whole pipeline; a slot timeout is reported
        # as RATE_LIMIT_EXCEEDED instead of a generic error.
        if self.rate_limiter is not None:
            try:
                async with self.rate_limiter.for_queries(
                    timeout=self.resilience_config.rate_limit_timeout
                ):
                    return await self._execute_query_traced(request, request_id)
            except TimeoutError:
                logger.warning(
                    "Query rejected: rate limiter slot could not be acquired",
                    extra={
                        "request_id": request_id,
                        "rate_limit_timeout": self.resilience_config.rate_limit_timeout,
                    },
                )
                self._record_query_request(
                    status=ErrorCode.RATE_LIMIT_EXCEEDED.value,
                    database=request.database or "auto",
                )
                return QueryResponse(
                    success=False,
                    generated_sql=None,
                    validation=None,
                    data=None,
                    error=ErrorDetail(
                        code=ErrorCode.RATE_LIMIT_EXCEEDED.value,
                        message=(
                            "Too many concurrent queries; the request was rejected "
                            f"after waiting {self.resilience_config.rate_limit_timeout}s "
                            "for a rate limiter slot"
                        ),
                        details={"rate_limit_timeout": self.resilience_config.rate_limit_timeout},
                    ),
                    confidence=0,
                    tokens_used=0,
                )

        return await self._execute_query_traced(request, request_id)

    async def _execute_query_traced(self, request: QueryRequest, request_id: str) -> QueryResponse:
        """Run the pipeline inside a tracing context with duration metrics."""
        async with request_context(request_id):
            start_time = time.perf_counter()
            try:
                return await self._execute_query_internal(request, request_id)
            finally:
                if self.metrics is not None:
                    self.metrics.query_duration.observe(time.perf_counter() - start_time)

    async def _execute_query_internal(
        self, request: QueryRequest, request_id: str
    ) -> QueryResponse:
        """Run the actual query pipeline (assumes rate limit slot acquired)."""
        try:
            # Step 1: Resolve database name
            database_name = self._resolve_database(request.database)
            logger.debug(
                "Resolved database",
                extra={"request_id": request_id, "database": database_name},
            )

            # Step 1b: Resolve the executor bound to this database.
            # Previously a single executor was used for every request, so a
            # request targeting database B could silently execute against
            # database A. Routing by database fixes that.
            executor = self._resolve_executor(database_name)

            # Step 2: Get schema from cache
            schema = self.schema_cache.get(database_name)
            if schema is None:
                # Schema not in cache, load it
                pool = self.pools.get(database_name)
                if pool is None:
                    raise DatabaseError(
                        message=f"No connection pool available for database '{database_name}'",
                        details={"database": database_name},
                    )
                try:
                    schema = await self.schema_cache.load(database_name, pool)
                except Exception as e:
                    raise SchemaLoadError(
                        message=f"Failed to load schema for database '{database_name}': {e!s}",
                        details={"database": database_name, "error": str(e)},
                    ) from e

            logger.debug(
                "Schema loaded",
                extra={
                    "request_id": request_id,
                    "database": database_name,
                    "tables": len(schema.tables),
                },
            )

            # Step 3: Generate and validate SQL with retry logic
            generated_sql, validation_result, tokens_used = await self._generate_sql_with_retry(
                question=request.question,
                schema=schema,
                request_id=request_id,
            )

            # Step 4: If return_type is SQL, return early
            if request.return_type == ReturnType.SQL:
                logger.info(
                    "Returning SQL only",
                    extra={"request_id": request_id, "sql_length": len(generated_sql)},
                )
                self._record_query_request(status="success", database=database_name)
                self._record_llm_tokens("generate_sql", tokens_used or 0)
                return QueryResponse(
                    success=True,
                    generated_sql=generated_sql,
                    validation=validation_result,
                    data=None,
                    error=None,
                    confidence=100,
                    tokens_used=tokens_used,
                )

            # Step 5: Execute SQL via the routed executor
            logger.debug("Executing SQL", extra={"request_id": request_id})
            start_time = self._get_current_time_ms()

            results, total_count = await executor.execute(generated_sql)

            execution_time_ms = self._get_current_time_ms() - start_time
            if self.metrics is not None:
                self.metrics.observe_db_query_duration(execution_time_ms / 1000.0)
            logger.info(
                "SQL executed successfully",
                extra={
                    "request_id": request_id,
                    "database": database_name,
                    "row_count": total_count,
                    "execution_time_ms": execution_time_ms,
                },
            )

            # Step 6: Validate results (non-blocking, failures don't fail the request)
            result_confidence = await self._validate_results_safely(
                question=request.question,
                sql=generated_sql,
                results=results,
                row_count=total_count,
                request_id=request_id,
            )

            self._record_query_request(status="success", database=database_name)
            self._record_llm_tokens("generate_sql", tokens_used or 0)

            # Step 7: Build successful response
            query_result = QueryResult(
                columns=list(results[0].keys()) if results else [],
                rows=results,
                row_count=len(results),  # Limited row count (after max_rows applied)
                execution_time_ms=execution_time_ms,
            )

            return QueryResponse(
                success=True,
                generated_sql=generated_sql,
                validation=validation_result,
                data=query_result,
                error=None,
                confidence=result_confidence,
                tokens_used=tokens_used,
            )

        except PgMcpError as e:
            # Handle known application errors
            if e.code == ErrorCode.SECURITY_VIOLATION:
                self._record_sql_rejected(reason="security_violation")
            logger.warning(
                "Query execution failed with known error",
                extra={
                    "request_id": request_id,
                    "error_code": e.code,
                    "error_message": str(e),
                },
            )
            self._record_query_request(status=e.code.value, database=request.database or "auto")
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=e.code.value,
                    message=e.message,
                    details=e.details,
                ),
                confidence=0,
                tokens_used=0,
            )
        except Exception as e:
            # Handle unexpected errors
            logger.exception(
                "Query execution failed with unexpected error",
                extra={"request_id": request_id},
            )
            self._record_query_request(
                status=ErrorCode.INTERNAL_ERROR.value, database=request.database or "auto"
            )
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=ErrorCode.INTERNAL_ERROR.value,
                    message=f"Internal server error: {e!s}",
                    details={"error_type": type(e).__name__},
                ),
                confidence=0,
                tokens_used=0,
            )

    def _resolve_database(self, database: str | None) -> str:
        """Resolve database name from request or auto-select.

        If database is specified, validate it exists.
        If not specified and only one database available, auto-select it.

        Args:
            database: Database name from request (optional).

        Returns:
            str: Resolved database name.

        Raises:
            DatabaseError: If database is invalid or cannot be auto-selected.

        Example:
            >>> name = orchestrator._resolve_database("mydb")  # Validates "mydb" exists
            >>> name = orchestrator._resolve_database(None)  # Auto-selects if only one DB
        """
        if database is not None:
            # Validate specified database exists
            if database not in self.pools:
                raise DatabaseError(
                    message=f"Database '{database}' not found",
                    details={
                        "requested_database": database,
                        "available_databases": list(self.pools.keys()),
                    },
                )
            return database

        # Auto-select if only one database available
        available_dbs = list(self.pools.keys())
        if len(available_dbs) == 0:
            raise DatabaseError(
                message="No databases configured",
                details={},
            )
        if len(available_dbs) == 1:
            return available_dbs[0]

        # Multiple databases, must specify
        raise DatabaseError(
            message="Multiple databases available, please specify which to query",
            details={"available_databases": available_dbs},
        )

    def _llm_backoff_delay(self, attempt: int) -> float:
        """Compute the exponential backoff delay before a retry attempt.

        Args:
            attempt: Zero-based index of the failed attempt that triggers the
                delay (the first retry waits ``retry_delay``).

        Returns:
            float: Delay in seconds, capped at the circuit breaker timeout to
            avoid unbounded waits.
        """
        delay = self.resilience_config.retry_delay * (
            self.resilience_config.backoff_factor**attempt
        )
        return min(delay, self.resilience_config.circuit_breaker_timeout)

    async def _generate_with_llm_rate_limit(
        self,
        question: str,
        schema: Any,
        previous_sql: str | None,
        error_feedback: str | None,
    ) -> str:
        """Call the SQL generator guarded by the LLM rate limiter.

        Args:
            question: User's natural language question.
            schema: Database schema for context.
            previous_sql: Previously generated SQL that failed (for retry).
            error_feedback: Error message from the previous attempt.

        Returns:
            str: Generated SQL.

        Raises:
            RateLimitExceededError: If no LLM slot could be acquired in time.
        """
        if self.rate_limiter is None:
            return await self.sql_generator.generate(
                question=question,
                schema=schema,
                previous_attempt=previous_sql,
                error_feedback=error_feedback,
            )
        try:
            async with self.rate_limiter.for_llm(timeout=self.resilience_config.rate_limit_timeout):
                return await self.sql_generator.generate(
                    question=question,
                    schema=schema,
                    previous_attempt=previous_sql,
                    error_feedback=error_feedback,
                )
        except TimeoutError as e:
            raise RateLimitExceededError(
                message=(
                    "LLM call rejected: no rate limiter slot acquired within "
                    f"{self.resilience_config.rate_limit_timeout}s"
                ),
                details={"rate_limit_timeout": self.resilience_config.rate_limit_timeout},
            ) from e

    async def _generate_sql_with_retry(
        self,
        question: str,
        schema: Any,
        request_id: str,
    ) -> tuple[str, ValidationResult, int | None]:
        """Generate and validate SQL with retry logic on validation failures.

        This method implements a retry loop that:
        1. Checks circuit breaker state
        2. Generates SQL using LLM (guarded by the LLM rate limiter)
        3. Validates the generated SQL
        4. On validation failure, waits with exponential backoff and retries
           with error feedback
        5. Records success/failure to circuit breaker and metrics

        Args:
            question: User's natural language question.
            schema: Database schema for context.
            request_id: Request ID for tracking.

        Returns:
            tuple: (generated_sql, validation_result, tokens_used)

        Raises:
            LLMError: If circuit breaker is open or generation fails.
            RateLimitExceededError: If the LLM rate limiter rejects the call.
            SecurityViolationError: If SQL fails validation after all retries.
            SQLParseError: If SQL cannot be parsed.

        Example:
            >>> sql, validation, tokens = await orchestrator._generate_sql_with_retry(
            ...     question="Count users",
            ...     schema=db_schema,
            ...     request_id="123",
            ... )
        """
        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            raise LLMError(
                message="SQL generation service is temporarily unavailable (circuit breaker open)",
                details={
                    "circuit_state": self.circuit_breaker.state,
                    "failure_count": self.circuit_breaker.failure_count,
                },
            )

        previous_sql: str | None = None
        error_feedback: str | None = None
        max_retries = self.resilience_config.max_retries
        tokens_used: int | None = None

        for attempt in range(max_retries + 1):
            try:
                logger.debug(
                    "Generating SQL",
                    extra={
                        "request_id": request_id,
                        "attempt": attempt + 1,
                        "max_retries": max_retries + 1,
                    },
                )

                # Generate SQL (LLM call guarded by rate limiter + metrics)
                llm_start = time.perf_counter()
                if self.metrics is not None:
                    self.metrics.increment_llm_call(operation="generate_sql")
                generated_sql = await self._generate_with_llm_rate_limit(
                    question=question,
                    schema=schema,
                    previous_sql=previous_sql,
                    error_feedback=error_feedback,
                )
                if self.metrics is not None:
                    self.metrics.observe_llm_latency(
                        operation="generate_sql", duration=time.perf_counter() - llm_start
                    )

                # Note: tokens_used would come from OpenAI response metadata if available
                # For now, we don't extract it, but it can be added later

                logger.debug(
                    "SQL generated",
                    extra={
                        "request_id": request_id,
                        "sql_length": len(generated_sql),
                    },
                )

                # Validate SQL
                try:
                    self.sql_validator.validate_or_raise(generated_sql)
                except (SecurityViolationError, SQLParseError) as validation_error:
                    self._record_sql_rejected(reason="validation_failed")
                    if attempt < max_retries:
                        # Exponential backoff before retrying with feedback
                        delay = self._llm_backoff_delay(attempt)
                        logger.warning(
                            "SQL validation failed, retrying with feedback after backoff",
                            extra={
                                "request_id": request_id,
                                "attempt": attempt + 1,
                                "backoff_seconds": delay,
                                "error": str(validation_error),
                            },
                        )
                        await asyncio.sleep(delay)
                        previous_sql = generated_sql
                        error_feedback = str(validation_error)
                        continue
                    else:
                        # Out of retries, record failure and raise
                        self.circuit_breaker.record_failure()
                        logger.error(
                            "SQL validation failed after all retries",
                            extra={
                                "request_id": request_id,
                                "attempts": attempt + 1,
                                "error": str(validation_error),
                            },
                        )
                        raise

                # Validation successful
                self.circuit_breaker.record_success()
                logger.info(
                    "SQL generated and validated successfully",
                    extra={
                        "request_id": request_id,
                        "attempts": attempt + 1,
                    },
                )

                # Build validation result
                validation_result = ValidationResult(
                    is_valid=True,
                    is_select=True,
                    allows_data_modification=False,
                    uses_blocked_functions=[],
                    error_message=None,
                )

                return generated_sql, validation_result, tokens_used

            except (LLMError, SecurityViolationError, SQLParseError, RateLimitExceededError):
                # Re-raise known errors (rate limit rejections keep their own
                # error code instead of being masked as a generic LLM error)
                raise
            except Exception as e:
                # Unexpected error during generation
                self.circuit_breaker.record_failure()
                logger.exception(
                    "Unexpected error during SQL generation",
                    extra={"request_id": request_id},
                )
                raise LLMError(
                    message=f"SQL generation failed unexpectedly: {e!s}",
                    details={"error_type": type(e).__name__},
                ) from e

        # Should not reach here, but just in case
        self.circuit_breaker.record_failure()
        raise LLMError(
            message="SQL generation failed after all retry attempts",
            details={"max_retries": max_retries},
        )

    async def _validate_results_safely(
        self,
        question: str,
        sql: str,
        results: list[dict[str, Any]],
        row_count: int,
        request_id: str,
    ) -> int:
        """Validate query results with error handling (non-blocking).

        This method attempts to validate results using LLM, but failures
        don't cause the overall query to fail. Returns a confidence score.
        Confidence below ``min_confidence_score`` is logged as a warning
        (the configured field is now honoured instead of being dead config).

        Args:
            question: User's original question.
            sql: Generated SQL query.
            results: Query results.
            row_count: Total row count.
            request_id: Request ID for tracking.

        Returns:
            int: Confidence score (0-100). Returns 100 if validation disabled/fails.

        Example:
            >>> confidence = await orchestrator._validate_results_safely(
            ...     question="Count users",
            ...     sql="SELECT COUNT(*) FROM users",
            ...     results=[{"count": 42}],
            ...     row_count=1,
            ...     request_id="123",
            ... )
        """
        if not self.validation_config.enabled:
            return 100

        try:
            logger.debug(
                "Validating results",
                extra={"request_id": request_id},
            )

            llm_start = time.perf_counter()
            if self.metrics is not None:
                self.metrics.increment_llm_call(operation="validate_result")
            validation_result = await self.result_validator.validate(
                question=question,
                sql=sql,
                results=results,
                row_count=row_count,
            )
            if self.metrics is not None:
                self.metrics.observe_llm_latency(
                    operation="validate_result", duration=time.perf_counter() - llm_start
                )

            # Honour min_confidence_score: flag low-confidence results.
            if validation_result.confidence < self.validation_config.min_confidence_score:
                logger.warning(
                    "Result confidence below configured minimum",
                    extra={
                        "request_id": request_id,
                        "confidence": validation_result.confidence,
                        "min_confidence_score": self.validation_config.min_confidence_score,
                    },
                )

            logger.info(
                "Result validation completed",
                extra={
                    "request_id": request_id,
                    "confidence": validation_result.confidence,
                    "is_acceptable": validation_result.is_acceptable,
                },
            )

            return validation_result.confidence

        except Exception as e:
            # Log but don't fail the query
            logger.warning(
                "Result validation failed, continuing with default confidence",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                },
            )
            return 100  # Default to high confidence if validation fails

    @staticmethod
    def _get_current_time_ms() -> float:
        """Get current time in milliseconds.

        Returns:
            float: Current time in milliseconds since epoch.
        """
        return time.time() * 1000
