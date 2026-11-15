"""Router package for Two-Tier Intent Routing and Dynamic Tool Scoping."""

from app.router.intent_router import (
    DeterministicIntentRouter,
    FastPathIntent,
    RouteDecision,
    RouteTarget,
    execute_fast_path,
)
from app.router.tool_registry import (
    CUSTOMER_ALLOWED_TOOLS,
    ADMIN_ALLOWED_TOOLS,
    BLOCKED_FROM_CUSTOMER,
    ALL_TOOL_DEFINITIONS,
    get_tools_for_role,
    validate_tool_authorization,
    is_tool_allowed,
)

__all__ = [
    "DeterministicIntentRouter",
    "FastPathIntent",
    "RouteDecision",
    "RouteTarget",
    "execute_fast_path",
    "CUSTOMER_ALLOWED_TOOLS",
    "ADMIN_ALLOWED_TOOLS",
    "BLOCKED_FROM_CUSTOMER",
    "ALL_TOOL_DEFINITIONS",
    "get_tools_for_role",
    "validate_tool_authorization",
    "is_tool_allowed",
]

