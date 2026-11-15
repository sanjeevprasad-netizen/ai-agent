"""Deterministic Two-Tier Intent Router.

Routes incoming queries into:
1. FAST_PATH:
   - High-confidence, deterministic, read-only requests (Order status/tracking, Product inventory/price).
   - Zero LLM calls, zero ReAct loops, zero vector search overhead.
   - Pydantic v2 output contract validation.
   - Sub-80ms execution target.
2. SLOW_PATH:
   - Ambiguous, multi-step, reasoning-heavy requests (Returns, refunds, complaints, policy interpretation).
   - Falls back to the full ReAct agent with role-scoped tools.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.contracts.tool_contracts import (
    InventoryCheckInput,
    InventoryCheckOutput,
    OrderLookupInput,
    OrderLookupOutput,
)
from app.db.models import User
from app.tools.mcp_tools import tool_inventory_check, tool_order_lookup

logger = logging.getLogger(__name__)


class RouteTarget(str, Enum):
    FAST_PATH = "fast_path"
    SLOW_PATH = "slow_path"


class FastPathIntent(str, Enum):
    ORDER_LOOKUP = "order_lookup"
    INVENTORY_CHECK = "inventory_check"


@dataclass
class RouteDecision:
    target: RouteTarget
    intent: Optional[FastPathIntent] = None
    extracted_params: Optional[Dict[str, Any]] = None
    reason: str = ""


# ─── Regular Expressions & Pattern Matchers ─────────────────────────────────

# Order ID patterns: ORD0210788, ORD-1001, ORD-2024-001
ORDER_ID_REGEX = re.compile(r"\b(ORD[-_A-Za-z0-9]{3,24})\b", re.IGNORECASE)

# Product ID / SKU patterns: PRD0000001, PROD-001, SKU-123
PRODUCT_ID_REGEX = re.compile(r"\b((?:PRD|PROD|SKU)[-_A-Za-z0-9]{2,20})\b", re.IGNORECASE)

# Email pattern
EMAIL_REGEX = re.compile(r"\b([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b")

# Complex reasoning / policy keywords that MUST divert to Slow Path ReAct
SLOW_PATH_KEYWORDS = re.compile(
    r"\b(return|refund|exchange|cancel|damaged|broken|defective|wrong\s+item|"
    r"policy|warranty|why|reason|how\s+do\s+i|can\s+i|complaint|dispute|"
    r"compensat|late|delayed|lost|stolen)\b",
    re.IGNORECASE,
)

# Deterministic Order Lookup Intent Patterns
ORDER_LOOKUP_PATTERNS = [
    re.compile(r"^\s*where\s+is\s+(?:the\s+)?(?:my\s+)?(?:order\s+)?(ORD[-_A-Za-z0-9]+)[\s?!.]*$", re.IGNORECASE),
    re.compile(r"^\s*(what\s+is\s+the\s+)?status\s+of\s+(?:the\s+)?(?:my\s+)?(?:order\s+)?(ORD[-_A-Za-z0-9]+)[\s?!.]*$", re.IGNORECASE),
    re.compile(r"^\s*track\s+(?:the\s+)?(?:my\s+)?(?:order\s+)?(ORD[-_A-Za-z0-9]+)[\s?!.]*$", re.IGNORECASE),
    re.compile(r"^\s*check\s+(the\s+status\s+of\s+)?(?:my\s+)?(?:order\s+)?(ORD[-_A-Za-z0-9]+)[\s?!.]*$", re.IGNORECASE),
    re.compile(r"^\s*(check|show|get|view)\s+(the\s+status\s+of\s+)?my\s+order(s)?[\s?!.]*$", re.IGNORECASE),
    re.compile(r"^\s*where\s+is\s+my\s+order[\s?!.]*$", re.IGNORECASE),
]

# Deterministic Inventory Check Intent Patterns
INVENTORY_PATTERNS = [
    re.compile(r"^\s*is\s+((?:PRD|PROD|SKU)[-_A-Za-z0-9]+)\s+in\s+stock[\s?!.]*$", re.IGNORECASE),
    re.compile(r"^\s*(what\s+is\s+the\s+)?price\s+of\s+((?:PRD|PROD|SKU)[-_A-Za-z0-9]+)[\s?!.]*$", re.IGNORECASE),
    re.compile(r"^\s*check\s+(stock|inventory)\s+for\s+((?:PRD|PROD|SKU)[-_A-Za-z0-9]+)[\s?!.]*$", re.IGNORECASE),
    re.compile(r"^\s*how\s+much\s+is\s+((?:PRD|PROD|SKU)[-_A-Za-z0-9]+)[\s?!.]*$", re.IGNORECASE),
    re.compile(r"^\s*do\s+you\s+have\s+((?:PRD|PROD|SKU)[-_A-Za-z0-9]+)\s+(in\s+stock|available)[\s?!.]*$", re.IGNORECASE),
]


class DeterministicIntentRouter:
    """Classifies user queries into Fast Path vs Slow Path deterministically."""

    def classify(self, message: str, current_user: Optional[User] = None) -> RouteDecision:
        clean = message.strip()

        # 1. Check for complex reasoning / return / refund / policy keywords
        # If any are present, query MUST go through Slow Path ReAct
        if SLOW_PATH_KEYWORDS.search(clean):
            return RouteDecision(
                target=RouteTarget.SLOW_PATH,
                reason="Query contains policy, return, refund, or multi-step reasoning triggers.",
            )

        # 2. Check Order Lookup Fast Path Patterns
        for pattern in ORDER_LOOKUP_PATTERNS:
            match = pattern.search(clean)
            if match:
                order_id_match = ORDER_ID_REGEX.search(clean)
                order_id = order_id_match.group(1).upper() if order_id_match else None
                email_match = EMAIL_REGEX.search(clean)
                target_email = email_match.group(1).lower() if email_match else (current_user.email if current_user else None)

                return RouteDecision(
                    target=RouteTarget.FAST_PATH,
                    intent=FastPathIntent.ORDER_LOOKUP,
                    extracted_params={
                        "order_id": order_id,
                        "email": target_email,
                    },
                    reason="Deterministic order lookup intent matched.",
                )

        # 3. Check Inventory / Stock Fast Path Patterns
        for pattern in INVENTORY_PATTERNS:
            match = pattern.search(clean)
            if match:
                prod_id_match = PRODUCT_ID_REGEX.search(clean)
                prod_id = prod_id_match.group(1).upper() if prod_id_match else None

                return RouteDecision(
                    target=RouteTarget.FAST_PATH,
                    intent=FastPathIntent.INVENTORY_CHECK,
                    extracted_params={
                        "id": prod_id,
                    },
                    reason="Deterministic inventory/stock inquiry matched.",
                )

        # 4. Fall back to Slow Path ReAct Agent for anything ambiguous
        return RouteDecision(
            target=RouteTarget.SLOW_PATH,
            reason="Ambiguous or conversational query; deferring to ReAct orchestrator.",
        )


# ─── Fast-Path Dispatcher ───────────────────────────────────────────────────

def format_fast_order_response(data: Dict[str, Any], current_user: Optional[User]) -> str:
    """Format OrderLookupOutput into a concise markdown response."""
    if not data.get("success", False) or data.get("error"):
        return f"Unable to retrieve order: {data.get('error', 'Unknown error')}"

    orders = data.get("orders", [])
    if not orders:
        return "No orders were found matching your request. If you have an order number, please share it!"

    order = orders[0]
    status = order.get("status", "processing").title()
    order_id = order.get("order_id", "N/A")
    tracking = order.get("tracking_number") or "Pending assignment"
    delivery = order.get("delivery_date")
    total = order.get("total_amount", 0.0)

    delivery_str = delivery[:10] if delivery else "Processing delivery schedule"

    lines = [
        f"### Order Status: **{order_id}**",
        f"- **Status:** {status}",
        f"- **Total Amount:** ${total:.2f}",
        f"- **Tracking Number:** {tracking}",
        f"- **Delivery Date:** {delivery_str}",
    ]

    items = order.get("items", [])
    if items:
        lines.append("\n**Items:**")
        for it in items:
            lines.append(f"- {it.get('quantity', 1)}x {it.get('product_name', 'Item')} (${it.get('unit_price', 0.0):.2f})")

    returns = order.get("returns", [])
    if returns:
        lines.append("\n**Active Returns:**")
        for r in returns:
            lines.append(f"- RMA `{r.get('rma_code')}`: {r.get('status', 'requested').title()} ({r.get('reason')})")

    return "\n".join(lines)


def format_fast_inventory_response(data: Dict[str, Any]) -> str:
    """Format InventoryCheckOutput into a concise markdown response."""
    if not data.get("success", False) or data.get("error"):
        return f"Unable to check inventory: {data.get('error', 'Unknown error')}"

    products = data.get("products", [])
    if not products:
        return "No matching products were found in our catalog."

    p = products[0]
    stock = p.get("stock_quantity", 0)
    in_stock = p.get("in_stock", False)
    status_str = "In Stock" if in_stock else "Out of Stock"

    return (
        f"### Product: **{p.get('name')}** (`{p.get('product_id')}`)\n"
        f"- **Price:** ${p.get('price', 0.0):.2f}\n"
        f"- **Availability:** **{status_str}** ({stock} units available)\n"
        f"- **Category:** {p.get('category', 'General').title()}\n"
        f"- **Warranty:** {p.get('warranty_months', 0)} months manufacturer warranty\n"
        f"- **Description:** {p.get('description', '')}"
    )


async def execute_fast_path(
    decision: RouteDecision,
    db_session: AsyncSession,
    current_user: Optional[User],
) -> Tuple[str, List[Dict[str, Any]], float]:
    """Execute fast-path tool directly with customer scoping and Pydantic v2 validation.

    Returns:
        (response_markdown, list_of_executed_tool_calls, elapsed_ms)
    """
    t_start = time.perf_counter()
    params = decision.extracted_params or {}

    if decision.intent == FastPathIntent.ORDER_LOOKUP:
        lookup_input = OrderLookupInput(
            order_id=params.get("order_id"),
            email=params.get("email"),
        )
        tool_res = await tool_order_lookup(db_session, current_user, lookup_input)
        response_text = format_fast_order_response(tool_res, current_user)
        tool_name = "order_lookup"
        tool_args = lookup_input.model_dump(exclude_none=True)

    elif decision.intent == FastPathIntent.INVENTORY_CHECK:
        inv_input = InventoryCheckInput(
            id=params.get("id"),
        )
        tool_res = await tool_inventory_check(db_session, inv_input)
        response_text = format_fast_inventory_response(tool_res)
        tool_name = "inventory_check"
        tool_args = inv_input.model_dump(exclude_none=True)

    else:
        raise ValueError(f"Unsupported fast path intent: {decision.intent}")

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    tool_calls = [{
        "tool": tool_name,
        "args": tool_args,
        "result": tool_res,
        "duration_ms": round(elapsed_ms, 2),
        "fast_path": True,
    }]

    return response_text, tool_calls, elapsed_ms
