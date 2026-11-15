"""Graceful Degradation and Automated Support Ticket Generation.

When a tool or reasoning operation persistently fails after retries:
1. Catches failure gracefully without exposing stack traces to the customer.
2. Creates an official support ticket in the PostgreSQL `support_tickets` table.
3. Deduplicates open tickets for the same user and failure category.
4. Returns a polite customer-facing message with their ticket reference.
"""

from datetime import datetime, timezone
import json
import logging
import random
from typing import Any, Dict, Optional

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import SupportTicket, User

logger = logging.getLogger(__name__)


async def create_support_ticket(
    session: AsyncSession,
    user: Optional[User],
    tool_name: str,
    failure_category: str,
    request_summary: str,
    diagnostic_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Create and persist an automated support ticket, deduplicating open tickets."""
    cust_id = (getattr(user, "customer_id", None) or getattr(user, "id", None) or "CUST-ANON") if user else "CUST-ANON"

    # Check for existing open ticket for same customer & tool
    existing_res = await session.execute(
        select(SupportTicket).where(
            and_(
                SupportTicket.customer_id == cust_id,
                SupportTicket.failed_tool == tool_name,
                SupportTicket.status == "open",
            )
        )
    )
    existing_ticket = existing_res.scalar_one_or_none()
    if existing_ticket:
        logger.info("Found existing open ticket %s for customer %s. Avoiding duplicate.", existing_ticket.id, cust_id)
        return existing_ticket.id

    ticket_id = f"TCK-{random.randint(10000, 99999)}"
    ticket = SupportTicket(
        id=ticket_id,
        customer_id=cust_id,
        request_summary=request_summary[:500],
        failure_category=failure_category,
        failed_tool=tool_name,
        retry_count=3,
        diagnostic_context=json.dumps(diagnostic_context or {}),
        status="open",
        created_at=datetime.now(timezone.utc),
    )
    session.add(ticket)
    await session.commit()
    logger.info("Created support ticket %s for customer %s on failure of %s", ticket_id, cust_id, tool_name)
    return ticket_id


def build_degraded_response(ticket_id: str, tool_name: str) -> str:
    """Construct a polite, professional degradation message for the customer."""
    tool_friendly_names = {
        "order_lookup": "order inquiry",
        "inventory_check": "product catalog lookup",
        "place_order": "order checkout",
        "request_return": "return processing",
        "process_refund": "refund authorization",
        "policy_search": "policy consultation",
    }
    action = tool_friendly_names.get(tool_name, "request")

    return (
        f"I apologize, but our automated system encountered a temporary technical issue while "
        f"processing your {action}. To ensure this is handled promptly, I have automatically created "
        f"support ticket **{ticket_id}** for our customer resolution team. A representative will "
        f"review this and follow up directly. You may also refer to ticket **{ticket_id}** in future inquiries."
    )

