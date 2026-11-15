"""Observability and execution tracer for ecom agent.

Tracks per-step latency, token consumption, and calculated USD costs.
"""

from datetime import datetime, timezone
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Pricing per 1,000 tokens (USD)
PRICING_TABLE = {
    "openai/gpt-oss-120b": {"prompt": 0.00060, "completion": 0.00080},
    "llama-3.3-70b-versatile": {"prompt": 0.00059, "completion": 0.00079},
    "llama-3.1-8b-instant": {"prompt": 0.00005, "completion": 0.00008},
    "mixtral-8x7b-32768": {"prompt": 0.00024, "completion": 0.00024},
    "llama3.2": {"prompt": 0.0, "completion": 0.0},
    "llama3.1": {"prompt": 0.0, "completion": 0.0},
    "default": {"prompt": 0.00015, "completion": 0.00060},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate estimated cost in USD based on model pricing."""
    rates = PRICING_TABLE.get(model, PRICING_TABLE["default"])
    cost = (prompt_tokens / 1000.0) * rates["prompt"] + (completion_tokens / 1000.0) * rates["completion"]
    return round(cost, 6)


class TraceStep:
    def __init__(self, step_number: int, step_type: str, name: str, details: Optional[Dict[str, Any]] = None):
        self.step_number = step_number
        self.step_type = step_type  # 'llm_reasoning' | 'tool_call' | 'triage_direct'
        self.name = name
        self.start_time = time.perf_counter()
        self.duration_ms: float = 0.0
        self.details: Dict[str, Any] = details or {}
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    def finish(
        self,
        output: Any = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        is_error: bool = False,
    ):
        self.duration_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        if output is not None:
            self.details["output"] = output
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.details["is_error"] = is_error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "type": self.step_type,
            "name": self.name,
            "duration_ms": self.duration_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "details": self.details,
        }


class ActiveTrace:
    def __init__(
        self,
        session_id: Optional[str],
        user_message: str,
        model_name: str = "openai/gpt-oss-120b",
        user_details: Optional[Dict[str, Any]] = None,
    ):
        # Generate a standard 32-character hex UUID compliant with OpenTelemetry and Langfuse trace standards
        self.trace_id = uuid.uuid4().hex
        self.session_id = session_id
        self.user_message = user_message
        self.model_name = model_name
        self.user_details = user_details or {}
        self.start_time = time.perf_counter()
        self.steps: List[TraceStep] = []
        self.step_counter = 0

    def start_step(self, step_type: str, name: str, input_data: Any = None) -> TraceStep:
        self.step_counter += 1
        details = {}
        if input_data is not None:
            details["input"] = input_data
        step = TraceStep(self.step_counter, step_type, name, details)
        self.steps.append(step)
        return step

    def finalize(self, final_response: str, status: str = "success") -> Dict[str, Any]:
        total_latency_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        total_prompt_tokens = sum(s.prompt_tokens for s in self.steps)
        total_completion_tokens = sum(s.completion_tokens for s in self.steps)
        cost_usd = estimate_cost(self.model_name, total_prompt_tokens, total_completion_tokens)

        steps_data = [s.to_dict() for s in self.steps]

        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "user_details": self.user_details,
            "user_message": self.user_message,
            "final_response": final_response,
            "total_latency_ms": total_latency_ms,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "estimated_cost_usd": cost_usd,
            "status": status,
            "steps": steps_data,
            "steps_json": json.dumps(steps_data),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

