"""Unit tests for configured security controls (tables/columns/EXPLAIN).

These tests verify that SecurityConfig's blocked_tables / blocked_columns /
allow_explain settings are actually honoured by SQLValidator (previously the
server hard-coded them to None/False, making the knobs dead configuration).
"""

import pytest

from pg_mcp.config.settings import SecurityConfig
from pg_mcp.models.errors import SecurityViolationError
from pg_mcp.services.sql_validator import SQLValidator


class TestSecurityConfigParsing:
    """SecurityConfig accepts comma-separated env values."""

    def test_blocked_tables_defaults_empty(self) -> None:
        config = SecurityConfig()
        assert config.blocked_tables == []
        assert config.blocked_columns == []
        assert config.allow_explain is False

    def test_blocked_tables_parsed_from_csv_string(self) -> None:
        config = SecurityConfig(blocked_tables="audit_log, payments , secrets")
        assert config.blocked_tables == ["audit_log", "payments", "secrets"]

    def test_blocked_columns_parsed_from_csv_string(self) -> None:
        config = SecurityConfig(blocked_columns="password_hash,ssn")
        assert config.blocked_columns == ["password_hash", "ssn"]

    def test_allow_explain_flag(self) -> None:
        assert SecurityConfig(allow_explain=True).allow_explain is True


class TestBlockedTablesEnforcement:
    """Configured blocked tables are rejected by the validator."""

    def _validator(self, **kwargs) -> SQLValidator:
        config = SecurityConfig(**kwargs)
        return SQLValidator(
            config=config,
            blocked_tables=config.blocked_tables or None,
            blocked_columns=config.blocked_columns or None,
            allow_explain=config.allow_explain,
        )

    def test_select_from_blocked_table_rejected(self) -> None:
        validator = self._validator(blocked_tables="payments")
        with pytest.raises(SecurityViolationError):
            validator.validate_or_raise("SELECT * FROM payments")

    def test_select_from_allowed_table_ok(self) -> None:
        validator = self._validator(blocked_tables="payments")
        # Should not raise
        validator.validate_or_raise("SELECT * FROM users")

    def test_blocked_table_case_insensitive(self) -> None:
        validator = self._validator(blocked_tables="Payments")
        with pytest.raises(SecurityViolationError):
            validator.validate_or_raise("SELECT * FROM PAYMENTS")

    def test_no_blocked_tables_allows_everything(self) -> None:
        validator = self._validator()
        validator.validate_or_raise("SELECT * FROM payments")  # no raise


class TestBlockedColumnsEnforcement:
    """Configured blocked columns are rejected by the validator."""

    def _validator(self, **kwargs) -> SQLValidator:
        config = SecurityConfig(**kwargs)
        return SQLValidator(
            config=config,
            blocked_tables=config.blocked_tables or None,
            blocked_columns=config.blocked_columns or None,
            allow_explain=config.allow_explain,
        )

    def test_select_blocked_column_rejected(self) -> None:
        validator = self._validator(blocked_columns="password_hash")
        with pytest.raises(SecurityViolationError):
            validator.validate_or_raise("SELECT password_hash FROM users")

    def test_select_allowed_column_ok(self) -> None:
        validator = self._validator(blocked_columns="password_hash")
        validator.validate_or_raise("SELECT username FROM users")  # no raise

    def test_select_star_not_flagged_by_column_rules(self) -> None:
        """SELECT * cannot be attributed to specific columns; it is allowed
        by column blocking (table-level rules are the coarse control)."""
        validator = self._validator(blocked_columns="password_hash")
        validator.validate_or_raise("SELECT * FROM users")  # no raise


class TestExplainPolicy:
    """allow_explain toggles whether EXPLAIN statements are permitted."""

    def _validator(self, allow_explain: bool) -> SQLValidator:
        config = SecurityConfig(allow_explain=allow_explain)
        return SQLValidator(
            config=config,
            blocked_tables=config.blocked_tables or None,
            blocked_columns=config.blocked_columns or None,
            allow_explain=config.allow_explain,
        )

    def test_explain_blocked_by_default(self) -> None:
        validator = self._validator(allow_explain=False)
        with pytest.raises(SecurityViolationError):
            validator.validate_or_raise("EXPLAIN SELECT * FROM users")

    def test_explain_allowed_when_configured(self) -> None:
        validator = self._validator(allow_explain=True)
        validator.validate_or_raise("EXPLAIN SELECT * FROM users")  # no raise

    def test_explain_analyze_treated_as_explain(self) -> None:
        validator = self._validator(allow_explain=True)
        validator.validate_or_raise("EXPLAIN ANALYZE SELECT * FROM users")  # no raise

    def test_explain_delete_follows_explain_policy(self) -> None:
        """EXPLAIN DELETE only shows the plan and never mutates data
        (documented in sql_validator), so it follows allow_explain."""
        validator = self._validator(allow_explain=True)
        validator.validate_or_raise("EXPLAIN DELETE FROM users")  # no raise

    def test_explain_delete_rejected_when_explain_disallowed(self) -> None:
        """With allow_explain=False, EXPLAIN DELETE is also rejected."""
        validator = self._validator(allow_explain=False)
        with pytest.raises(SecurityViolationError):
            validator.validate_or_raise("EXPLAIN DELETE FROM users")
