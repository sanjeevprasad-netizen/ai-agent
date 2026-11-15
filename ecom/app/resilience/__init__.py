"""Resilience package exposing bounded retries, structured errors, and graceful degradation."""

from app.resilience.retry import StructuredToolError, execute_with_bounded_retry
from app.resilience.degradation import create_support_ticket, build_degraded_response

__all__ = [
    "StructuredToolError",
    "execute_with_bounded_retry",
    "create_support_ticket",
    "build_degraded_response",
]

