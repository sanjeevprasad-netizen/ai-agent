"""Production Observability: Non-Blocking Langfuse Telemetry with Fail-Open Safety.

Implements:
1. Per-interaction tracing (user_id, session_id, trace_id) compliant with Langfuse v5 (OpenTelemetry)
2. Routing spans (Fast-Path vs Slow-Path)
3. Tool call logging with precise duration tracking and observation types (retriever vs tool)
4. LLM reasoning metrics (prompt tokens, completion tokens, latency)
5. PII Redaction before transmission
6. Complete fail-open safety: network or telemetry errors NEVER impact customer requests.
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

from app.config import settings
from app.security.pii import sanitize_telemetry_payload

logger = logging.getLogger(__name__)


class AsyncLangfuseClient:
    """Non-blocking, fail-open Langfuse v5 telemetry client wrapper."""

    def __init__(self):
        self._client = None
        self._configured = False

    def _clean_str(self, val: Optional[str]) -> Optional[str]:
        if val is None:
            return None
        return val.strip("\"'")

    def _init_client(self):
        self._configured = True
        raw_secret = settings.LANGFUSE_SECRET_KEY or os.getenv("LANGFUSE_SECRET_KEY")
        raw_public = settings.LANGFUSE_PUBLIC_KEY or os.getenv("LANGFUSE_PUBLIC_KEY")
        raw_base = settings.langfuse_host

        self.secret_key = self._clean_str(raw_secret)
        self.public_key = self._clean_str(raw_public)
        self.base_url = self._clean_str(raw_base)
        self._enabled = bool(self.secret_key and self.public_key)
        self._client = None

        if self._enabled:
            # Synchronize to os.environ for other OpenTelemetry / Langfuse components
            os.environ["LANGFUSE_PUBLIC_KEY"] = self.public_key
            os.environ["LANGFUSE_SECRET_KEY"] = self.secret_key
            os.environ["LANGFUSE_BASE_URL"] = self.base_url
            os.environ["LANGFUSE_HOST"] = self.base_url

            try:
                from langfuse import Langfuse
                self._client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.base_url,
                )
                logger.info("Langfuse telemetry client initialized for host: %s", self.base_url)
            except Exception as e:
                logger.warning("Langfuse initialization failed: %s", e)
                self._client = None
        else:
            logger.info("Langfuse keys not configured in .env. System operating in local PostgreSQL trace mode.")

    def reload_config(self):
        """Reload configuration from environment / settings."""
        self._init_client()

    def log_interaction_async(
        self,
        trace_id: str,
        session_id: str,
        user_id: str,
        user_email: str,
        user_message: str,
        final_response: str,
        route: str,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        tool_calls: Optional[list] = None,
        status: str = "success",
    ):
        """Schedule background telemetry emission with async flush and accurate timing."""
        # Attempt auto-detection if keys were added post-startup
        if self._client is None:
            self.reload_config()

        if self._client is None:
            return

        def _emit_and_flush():
            try:
                from langfuse import propagate_attributes
                from opentelemetry import trace as otel_trace_api

                # PII Redaction
                safe_query = sanitize_telemetry_payload(user_message, allowed_email=user_email)
                safe_reply = sanitize_telemetry_payload(final_response, allowed_email=user_email)
                safe_tools = sanitize_telemetry_payload(tool_calls or [], allowed_email=user_email)

                # Format trace context if trace_id is 32-hex
                clean_tid = trace_id.replace("-", "").lower() if trace_id else None

                # Compute nanosecond timestamps from latency_ms and tool durations
                now_ns = time.time_ns()
                total_dur_ns = int(max(1.0, latency_ms or 50.0) * 1e6)
                trace_start_ns = now_ns - total_dur_ns
                trace_end_ns = now_ns

                remote_parent = None
                if clean_tid and len(clean_tid) == 32:
                    remote_parent = self._client._create_remote_parent_span(
                        trace_id=clean_tid, parent_span_id=None
                    )

                context_manager = otel_trace_api.use_span(remote_parent) if remote_parent else None

                def _record_trace():
                    root_otel = self._client._otel_tracer.start_span(
                        name=f"ecom-chat-{route}",
                        start_time=trace_start_ns,
                    )
                    with otel_trace_api.use_span(root_otel):
                        with propagate_attributes(
                            user_id=str(user_id),
                            session_id=str(session_id),
                            tags=["ecommerce-support", route, status],
                            trace_name=f"ecom-chat-{route}",
                        ):
                            # Hierarchy: Add child tool/retriever observations with accurate duration
                            if safe_tools:
                                tool_cursor_ns = trace_start_ns
                                for tc in safe_tools:
                                    tool_name = tc.get("tool", "unknown")
                                    obs_type = "retriever" if "search" in tool_name or "policy" in tool_name else "tool"

                                    dur_ms = tc.get("duration_ms") or 10.0
                                    tool_span_dur_ns = int(max(1.0, float(dur_ms)) * 1e6)
                                    tool_start_ns = tool_cursor_ns
                                    tool_end_ns = tool_start_ns + tool_span_dur_ns
                                    tool_cursor_ns = tool_end_ns

                                    tool_otel = self._client._otel_tracer.start_span(
                                        name=f"{obs_type}:{tool_name}",
                                        start_time=tool_start_ns,
                                    )
                                    tool_obs = self._client._create_observation_from_otel_span(
                                        otel_span=tool_otel,
                                        as_type=obs_type,
                                        input=tc.get("args"),
                                        output=tc.get("result"),
                                        metadata={
                                            "fast_path": tc.get("fast_path", False),
                                            "observation_type": obs_type,
                                            "duration_ms": dur_ms,
                                        },
                                    )
                                    tool_obs.end(end_time=tool_end_ns)

                                trace_end_ns = max(trace_end_ns, tool_cursor_ns)

                    root_obs = self._client._create_observation_from_otel_span(
                        otel_span=root_otel,
                        as_type="span",
                        input={"message": safe_query},
                        output={"response": safe_reply},
                        metadata={
                            "user_email": user_email,
                            "route": route,
                            "status": status,
                            "latency_ms": latency_ms,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "tools_count": len(safe_tools),
                        },
                    )
                    root_obs.end(end_time=trace_end_ns)

                if context_manager:
                    with context_manager:
                        _record_trace()
                else:
                    _record_trace()

                # Flush in background executor thread to ensure prompt delivery
                self._client.flush()
            except Exception as exc:
                # Fail-open guarantee: Telemetry failure MUST NEVER impact customer
                logger.debug("Fail-open: Langfuse telemetry emission skipped: %s", exc)

        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, _emit_and_flush)
        except RuntimeError:
            _emit_and_flush()


# Singleton telemetry instance
telemetry = AsyncLangfuseClient()
