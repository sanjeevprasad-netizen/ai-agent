"""Resilience: Bounded Retries, Exponential Backoff, and Structured Error Feedback.

Provides:
1. Bounded retry policy with exponential backoff and jitter for transient errors.
2. Structured tool error taxonomy distinguishing retryable transient errors from non-retryable validation errors.
"""

import asyncio
import logging
import random
from typing import Any, Callable, Dict, Optional, Tuple, Type
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StructuredToolError(BaseModel):
    error_type: str = Field(..., description="Category: SchemaMismatch, TransientFailure, AuthorizationError, ResourceNotFound")
    message: str = Field(..., description="Clear human-readable error explanation")
    retryable: bool = Field(False, description="Whether caller can retry operation")
    diagnostic_context: Optional[Dict[str, Any]] = Field(None, description="Debug metadata for logging")


async def execute_with_bounded_retry(
    coro_fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay_ms: float = 100.0,
    max_delay_ms: float = 2000.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Any:
    """Execute async callable with bounded exponential backoff and randomized jitter."""
    attempt = 0
    while True:
        try:
            return await coro_fn()
        except retryable_exceptions as exc:
            attempt += 1
            if attempt >= max_retries:
                logger.warning("Bounded retry exceeded (%d/%d attempts). Re-raising: %s", attempt, max_retries, exc)
                raise exc

            # Calculate backoff with jitter
            backoff_ms = min(max_delay_ms, base_delay_ms * (2 ** (attempt - 1)))
            jitter = random.uniform(0.8, 1.2)
            sleep_sec = (backoff_ms * jitter) / 1000.0

            logger.info(
                "Transient error on attempt %d/%d (%s). Retrying in %.3fs...",
                attempt,
                max_retries,
                str(exc)[:80],
                sleep_sec,
            )
            await asyncio.sleep(sleep_sec)

