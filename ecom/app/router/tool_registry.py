"""Dynamic Role-Based Tool Registry & Capability Scoping with Defense-in-Depth.

Enforces:
1. Customer session only receives customer-authorized tools in LLM schemas.
2. Admin session receives customer tools + admin tools.
3. Defense-in-Depth: Even if an unauthorized tool call is attempted (via prompt injection or direct call),
   the execution boundary rejects it based on the authenticated user's role.
4. Output contracts validated via Pydantic v2.
"""

import logging
from typing import Any, Dict, List, Optional, Set
from sqlalchemy.ext.asyncio import AsyncSession

from app.contracts.tool_contracts import (
    OrderLookupInput,
    OrderLookupOutput,
    InventoryCheckInput,
    InventoryCheckOutput,
    PlaceOrderInput,
    PlaceOrderOutput,
    RequestReturnInput,
    RequestReturnOutput,
    ProcessRefundInput,
    ProcessRefundOutput,
    PolicySearchInput,
    PolicySearchOutput,
    ApproveRefundInput,
    ApproveRefundOutput,
    ReembedPoliciesInput,
    ReembedPoliciesOutput,
    InspectTracesInput,
    InspectTracesOutput,
)
from app.db.models import User

logger = logging.getLogger(__name__)

# ─── Role-Based Scopes ────────────────────────────────────────────────────────

CUSTOMER_ALLOWED_TOOLS: Set[str] = {
    "order_lookup",
    "inventory_check",
    "policy_search",
    "place_order",
    "request_return",
}

ADMIN_ALLOWED_TOOLS: Set[str] = CUSTOMER_ALLOWED_TOOLS | {
    "process_refund",
    "approve_refund",
    "reembed_policies",
    "inspect_traces",
}

BLOCKED_FROM_CUSTOMER: Set[str] = {
    "approve_refund",
    "reembed_policies",
    "inspect_traces",
}

# ─── Tool Definitions (OpenAI Function Calling Format) ────────────────────────

ALL_TOOL_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "order_lookup": {
        "type": "function",
        "function": {
            "name": "order_lookup",
            "description": "Look up existing order status, delivery date, line items, and tracking number by order_id, email, or status. Customers can only view their own orders.",
            "parameters": OrderLookupInput.model_json_schema(),
        },
    },
    "inventory_check": {
        "type": "function",
        "function": {
            "name": "inventory_check",
            "description": "Check current stock level, unit price, category, and warranty duration for products in the catalog.",
            "parameters": InventoryCheckInput.model_json_schema(),
        },
    },
    "policy_search": {
        "type": "function",
        "function": {
            "name": "policy_search",
            "description": "Search return, shipping, compensation, and warranty policies using semantic vector search with date filtering.",
            "parameters": PolicySearchInput.model_json_schema(),
        },
    },
    "place_order": {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": "Place a new customer order. Validates inventory, decrements stock, and creates order records.",
            "parameters": PlaceOrderInput.model_json_schema(),
        },
    },
    "request_return": {
        "type": "function",
        "function": {
            "name": "request_return",
            "description": "Initiate a product return request. Verifies order is 'delivered' and generates an official RMA tracking code.",
            "parameters": RequestReturnInput.model_json_schema(),
        },
    },
    "process_refund": {
        "type": "function",
        "function": {
            "name": "process_refund",
            "description": "Process a monetary refund for an RMA return. Auto-approves under $50; requires admin review if > $50.",
            "parameters": ProcessRefundInput.model_json_schema(),
        },
    },
    # Admin-Only Tools
    "approve_refund": {
        "type": "function",
        "function": {
            "name": "approve_refund",
            "description": "Administrator only: Approve a pending refund that exceeded the automatic $50 threshold.",
            "parameters": ApproveRefundInput.model_json_schema(),
        },
    },
    "reembed_policies": {
        "type": "function",
        "function": {
            "name": "reembed_policies",
            "description": "Administrator only: Trigger policy re-indexing and embedding calculation into vector storage.",
            "parameters": ReembedPoliciesInput.model_json_schema(),
        },
    },
    "inspect_traces": {
        "type": "function",
        "function": {
            "name": "inspect_traces",
            "description": "Administrator only: Inspect recent agent execution traces, token usage, and latency metrics.",
            "parameters": InspectTracesInput.model_json_schema(),
        },
    },
}


def get_tools_for_role(role: str) -> List[Dict[str, Any]]:
    """Build dynamic OpenAI tool schemas permitted strictly for the given role."""
    allowed_names = ADMIN_ALLOWED_TOOLS if role == "admin" else CUSTOMER_ALLOWED_TOOLS
    return [ALL_TOOL_DEFINITIONS[name] for name in allowed_names if name in ALL_TOOL_DEFINITIONS]


def validate_tool_authorization(tool_name: str, user_role: str) -> None:
    """Defense-in-depth authorization validation.

    Ensures that even if an attacker attempts prompt injection or directly submits a tool call,
    the tool execution is rejected before accessing any service or database.
    """
    allowed_tools = ADMIN_ALLOWED_TOOLS if user_role == "admin" else CUSTOMER_ALLOWED_TOOLS
    if tool_name not in allowed_tools:
        logger.warning(
            "SECURITY VIOLATION: Role '%s' attempted to invoke unauthorized tool '%s'.",
            user_role,
            tool_name,
        )
        raise PermissionError(
            f"Access Denied: The tool '{tool_name}' is restricted and cannot be executed by users with role '{user_role}'."
        )


def is_tool_allowed(tool_name: str, user_role: str) -> bool:
    """Check if a tool is permitted for a role without raising an exception."""
    allowed_tools = ADMIN_ALLOWED_TOOLS if user_role == "admin" else CUSTOMER_ALLOWED_TOOLS
    return tool_name in allowed_tools

