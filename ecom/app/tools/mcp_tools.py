"""Official Python MCP tool definitions with strict dual-sided Pydantic v2 contracts.

Implements transactional e-commerce lifecycle:
1. order_lookup (with customer role scoping)
2. inventory_check
3. place_order
4. request_return
5. process_refund (with idempotency and $50 approval threshold)
6. policy_search (with pgvector and date-filtering)
7. approve_refund (admin only)
8. reembed_policies (admin only)
9. inspect_traces (admin only)
"""

from datetime import datetime, timezone
import json
import logging
import random
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import select, or_, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.contracts.tool_contracts import (
    OrderLookupInput,
    OrderLookupOutput,
    OrderDetail,
    OrderItemDetail,
    ReturnSummary,
    InventoryCheckInput,
    InventoryCheckOutput,
    ProductDetail,
    PlaceOrderInput,
    PlaceOrderOutput,
    RequestReturnInput,
    RequestReturnOutput,
    ProcessRefundInput,
    ProcessRefundOutput,
    PolicySearchInput,
    PolicySearchOutput,
    PolicyChunkResult,
    ApproveRefundInput,
    ApproveRefundOutput,
    ReembedPoliciesInput,
    ReembedPoliciesOutput,
    InspectTracesInput,
    InspectTracesOutput,
)
from app.db.models import Order, OrderItem, Product, Return, Refund, User, ExecutionTrace

logger = logging.getLogger(__name__)

# Optional MCPServer integration
try:
    from mcp.server.mcpserver import MCPServer
    mcp_server = MCPServer("ecom-support")
except Exception:
    mcp_server = None


# ─── 1. Order Lookup Tool ─────────────────────────────────────────────────────

async def tool_order_lookup(
    session: AsyncSession,
    current_user: Optional[User],
    params: OrderLookupInput,
) -> Dict[str, Any]:
    """Execute order lookup with customer security scoping and Pydantic v2 output validation."""
    stmt = select(Order).options(selectinload(Order.items), selectinload(Order.returns))
    conditions = []

    # Security Scoping: Customer role can ONLY view their own orders
    if current_user and current_user.role == "customer":
        conditions.append(Order.customer_email == current_user.email)
        # If user explicitly asked for a different email, deny
        if params.email and params.email.lower().strip() != current_user.email.lower():
            raw_err = {
                "success": False,
                "count": 0,
                "orders": [],
                "error": "Access Denied: Customers can only query orders associated with their own verified email address.",
            }
            return OrderLookupOutput.model_validate(raw_err).model_dump()
    elif params.email:
        conditions.append(Order.customer_email.ilike(f"%{params.email.strip()}%"))

    if params.order_id:
        conditions.append(Order.id == params.order_id.strip().upper())

    if params.status:
        conditions.append(Order.status == params.status.strip().lower())

    if not conditions:
        raw_err = {
            "success": False,
            "count": 0,
            "orders": [],
            "error": "Please provide an order_id, customer email, or status to search orders.",
        }
        return OrderLookupOutput.model_validate(raw_err).model_dump()

    stmt = stmt.where(and_(*conditions)).order_by(Order.created_at.desc()).limit(10)
    result = await session.execute(stmt)
    orders = result.scalars().all()

    if not orders:
        raw_empty = {
            "success": True,
            "count": 0,
            "orders": [],
            "message": "No orders found matching the specified search criteria.",
        }
        return OrderLookupOutput.model_validate(raw_empty).model_dump()

    output = []
    for o in orders:
        items_summary = [
            OrderItemDetail(
                product_id=item.product_id,
                product_name=item.product_name,
                quantity=item.quantity,
                unit_price=item.unit_price,
                net_amount=getattr(item, "net_amount", None) or (item.quantity * item.unit_price),
            )
            for item in o.items
        ]
        returns_summary = [
            ReturnSummary(rma_code=r.id, reason=r.reason, status=r.status)
            for r in o.returns
        ]
        output.append(
            OrderDetail(
                order_id=o.id,
                customer_email=o.customer_email,
                status=o.status,
                total_amount=o.total_amount,
                tracking_number=o.tracking_number,
                delivery_date=o.delivery_date.isoformat() if o.delivery_date else None,
                created_at=o.created_at.isoformat() if o.created_at else None,
                items=items_summary,
                returns=returns_summary,
            )
        )

    raw_success = {
        "success": True,
        "count": len(output),
        "orders": output,
        "message": f"Successfully retrieved {len(output)} order(s).",
    }
    validated = OrderLookupOutput.model_validate(raw_success)
    return validated.model_dump()


# ─── 2. Inventory Check Tool ──────────────────────────────────────────────────

async def tool_inventory_check(
    session: AsyncSession,
    params: InventoryCheckInput,
) -> Dict[str, Any]:
    """Execute product catalog inventory search with Pydantic v2 output validation."""
    stmt = select(Product)
    conditions = []

    if params.id:
        conditions.append(Product.id == params.id.strip().upper())
    if params.name:
        conditions.append(Product.name.ilike(f"%{params.name.strip()}%"))
    if params.category:
        conditions.append(Product.category.ilike(f"%{params.category.strip()}%"))

    if conditions:
        stmt = stmt.where(or_(*conditions))

    stmt = stmt.order_by(Product.name.asc()).limit(15)
    result = await session.execute(stmt)
    products = result.scalars().all()

    if not products:
        raw_empty = {
            "success": True,
            "count": 0,
            "products": [],
            "message": "No matching products found in catalog.",
        }
        return InventoryCheckOutput.model_validate(raw_empty).model_dump()

    output = [
        ProductDetail(
            product_id=p.id,
            name=p.name,
            category=p.category,
            price=p.price,
            stock_quantity=p.stock_quantity,
            in_stock=p.stock_quantity > 0,
            warranty_months=p.warranty_months,
            description=p.description,
        )
        for p in products
    ]
    raw_success = {
        "success": True,
        "count": len(output),
        "products": output,
        "message": f"Found {len(output)} matching product(s).",
    }
    validated = InventoryCheckOutput.model_validate(raw_success)
    return validated.model_dump()


# ─── 3. Place Order Tool ──────────────────────────────────────────────────────

async def tool_place_order(
    session: AsyncSession,
    current_user: Optional[User],
    params: PlaceOrderInput,
) -> Dict[str, Any]:
    """Place a new customer order with inventory validation and Pydantic v2 contract."""
    target_email = params.customer_email.strip().lower()

    # Customer scoping check
    if current_user and current_user.role == "customer":
        if target_email != current_user.email.lower():
            raw_err = {
                "success": False,
                "status": "error",
                "error": "Customers cannot place orders under a different email address.",
            }
            return PlaceOrderOutput.model_validate(raw_err).model_dump()
        user = current_user
    else:
        res = await session.execute(select(User).where(User.email == target_email))
        user = res.scalar_one_or_none()
        if not user:
            raw_err = {
                "success": False,
                "status": "error",
                "error": f"No registered customer account found for email: {target_email}.",
            }
            return PlaceOrderOutput.model_validate(raw_err).model_dump()

    total_amount = 0.0
    items_to_create = []

    for item_req in params.items:
        prod_res = await session.execute(select(Product).where(Product.id == item_req.product_id.strip().upper()))
        product = prod_res.scalar_one_or_none()
        if not product:
            raw_err = {
                "success": False,
                "status": "error",
                "error": f"Product '{item_req.product_id}' does not exist in catalog.",
            }
            return PlaceOrderOutput.model_validate(raw_err).model_dump()

        if product.stock_quantity < item_req.quantity:
            raw_err = {
                "success": False,
                "status": "error",
                "error": f"Insufficient stock for '{product.name}' ({product.id}). Available: {product.stock_quantity}, Requested: {item_req.quantity}.",
            }
            return PlaceOrderOutput.model_validate(raw_err).model_dump()

        # Decrement inventory
        product.stock_quantity -= item_req.quantity
        line_total = product.price * item_req.quantity
        total_amount += line_total

        items_to_create.append({
            "product_id": product.id,
            "product_name": product.name,
            "quantity": item_req.quantity,
            "unit_price": product.price,
        })

    order_id = f"ORD-{random.randint(1000, 9999)}"
    tracking_no = f"TRK-{random.randint(10000000, 99999999)}"
    now_utc = datetime.now(timezone.utc)

    cust_id = getattr(user, "customer_id", None) or user.id
    new_order = Order(
        id=order_id,
        customer_id=cust_id,
        customer_email=user.email,
        order_date=now_utc,
        status="processing",
        total_amount=round(total_amount, 2),
        tracking_number=tracking_no,
        actual_delivery_date=None,
        created_at=now_utc,
    )
    session.add(new_order)
    await session.flush()

    for item_data in items_to_create:
        line_id = f"OL-{uuid.uuid4().hex[:10].upper()}"
        line_net = round(item_data["quantity"] * item_data["unit_price"], 2)
        order_item = OrderItem(
            id=line_id,
            order_id=order_id,
            product_id=item_data["product_id"],
            product_name=item_data["product_name"],
            quantity=item_data["quantity"],
            unit_price=item_data["unit_price"],
            net_amount=line_net,
        )
        session.add(order_item)

    await session.commit()

    raw_success = {
        "success": True,
        "status": "success",
        "order_id": order_id,
        "customer_email": user.email,
        "order_status": "processing",
        "total_amount": round(total_amount, 2),
        "tracking_number": tracking_no,
        "items_count": len(items_to_create),
        "message": f"Order {order_id} placed successfully. Warehouse status: processing.",
    }
    return PlaceOrderOutput.model_validate(raw_success).model_dump()


# ─── 4. Request Return Tool ───────────────────────────────────────────────────

async def tool_request_return(
    session: AsyncSession,
    current_user: Optional[User],
    params: RequestReturnInput,
) -> Dict[str, Any]:
    """Initiate a return request for an item in a delivered order."""
    order_res = await session.execute(
        select(Order)
        .options(selectinload(Order.items), selectinload(Order.returns))
        .where(Order.id == params.order_id.strip().upper())
    )
    order = order_res.scalar_one_or_none()

    if not order:
        raw_err = {
            "success": False,
            "status": "error",
            "error": f"Order '{params.order_id}' was not found in our system.",
        }
        return RequestReturnOutput.model_validate(raw_err).model_dump()

    # Ownership check
    if current_user and current_user.role == "customer":
        if order.customer_email.lower() != current_user.email.lower():
            raw_err = {
                "success": False,
                "status": "error",
                "error": "Access Denied: You cannot request returns for orders belonging to another customer.",
            }
            return RequestReturnOutput.model_validate(raw_err).model_dump()

    # State validation: must be 'delivered'
    if order.status.lower() != "delivered":
        raw_err = {
            "success": False,
            "status": "error",
            "error": f"Cannot return order '{order.id}'. Returns are only permitted for delivered orders. Current order status: '{order.status}'.",
        }
        return RequestReturnOutput.model_validate(raw_err).model_dump()

    product_in_order = any(item.product_id == params.product_id.strip().upper() for item in order.items)
    if not product_in_order:
        valid_items = [f"{item.product_id} ({item.product_name})" for item in order.items]
        raw_err = {
            "success": False,
            "status": "error",
            "error": f"Product '{params.product_id}' is not in order '{order.id}'. Valid items in this order: {', '.join(valid_items)}.",
        }
        return RequestReturnOutput.model_validate(raw_err).model_dump()

    rma_code = f"RMA-{random.randint(1000, 9999)}"
    return_record = Return(
        id=rma_code,
        order_id=order.id,
        customer_id=order.customer_id,
        reason=params.reason.strip(),
        status="requested",
        created_at=datetime.now(timezone.utc),
    )
    session.add(return_record)
    await session.commit()

    raw_success = {
        "success": True,
        "status": "success",
        "rma_code": rma_code,
        "order_id": order.id,
        "product_id": params.product_id.upper(),
        "return_status": "requested",
        "message": f"Return request initiated successfully with authorization code {rma_code}. You may print a prepaid label or proceed to refund evaluation.",
    }
    return RequestReturnOutput.model_validate(raw_success).model_dump()


# ─── 5. Process Refund Tool ───────────────────────────────────────────────────

async def tool_process_refund(
    session: AsyncSession,
    current_user: Optional[User],
    params: ProcessRefundInput,
) -> Dict[str, Any]:
    """Process a refund with idempotency protection and a $50 admin approval limit."""
    # 1. Idempotency check
    existing_refund_res = await session.execute(
        select(Refund).where(Refund.idempotency_key == params.idempotency_key.strip())
    )
    existing_refund = existing_refund_res.scalar_one_or_none()
    if existing_refund:
        raw_idem = {
            "success": True,
            "status": "idempotent_duplicate",
            "refund_id": existing_refund.id,
            "rma_code": existing_refund.return_id,
            "order_id": existing_refund.order_id,
            "amount": existing_refund.amount,
            "refund_status": existing_refund.status,
            "requires_admin_approval": existing_refund.status == "pending_admin_approval",
            "message": f"Refund with idempotency key '{params.idempotency_key}' was already processed with status '{existing_refund.status}'. Duplicate transaction prevented.",
        }
        return ProcessRefundOutput.model_validate(raw_idem).model_dump()

    # 2. Validate Order & Return
    order_res = await session.execute(select(Order).where(Order.id == params.order_id.strip().upper()))
    order = order_res.scalar_one_or_none()
    if not order:
        raw_err = {
            "success": False,
            "status": "error",
            "error": f"Order '{params.order_id}' was not found.",
        }
        return ProcessRefundOutput.model_validate(raw_err).model_dump()

    return_res = await session.execute(select(Return).where(Return.id == params.rma_code.strip().upper()))
    return_record = return_res.scalar_one_or_none()
    if not return_record or return_record.order_id != order.id:
        raw_err = {
            "success": False,
            "status": "error",
            "error": f"No valid return request matching RMA code '{params.rma_code}' for order '{order.id}'. Please initiate a return request first.",
        }
        return ProcessRefundOutput.model_validate(raw_err).model_dump()

    # Ownership check
    if current_user and current_user.role == "customer":
        if order.customer_email.lower() != current_user.email.lower():
            raw_err = {
                "success": False,
                "status": "error",
                "error": "Access Denied: You cannot process refunds for another customer's order.",
            }
            return ProcessRefundOutput.model_validate(raw_err).model_dump()

    refund_id = f"REF-{random.randint(1000, 9999)}"
    amount = round(params.amount, 2)

    # Threshold check ($50 limit)
    if amount <= 50.0:
        refund_status = "completed"
        order.status = "returned"
        return_record.status = "approved"
        message = f"Refund of ${amount:.2f} processed and completed automatically (under the $50 threshold). Order '{order.id}' status set to 'returned'."
    else:
        refund_status = "pending_admin_approval"
        message = f"Refund amount ${amount:.2f} exceeds automatic threshold ($50.00). Refund registered with status 'pending_admin_approval'. An administrator must approve the transaction."

    refund_obj = Refund(
        id=refund_id,
        return_id=return_record.id,
        order_id=order.id,
        amount=amount,
        status=refund_status,
        idempotency_key=params.idempotency_key.strip(),
        created_at=datetime.now(timezone.utc),
    )
    session.add(refund_obj)
    await session.commit()

    raw_success = {
        "success": True,
        "status": "success",
        "refund_id": refund_id,
        "rma_code": return_record.id,
        "order_id": order.id,
        "amount": amount,
        "refund_status": refund_status,
        "requires_admin_approval": refund_status == "pending_admin_approval",
        "message": message,
    }
    return ProcessRefundOutput.model_validate(raw_success).model_dump()


# ─── 6. Policy Search Tool ────────────────────────────────────────────────────

async def tool_policy_search(
    session: AsyncSession,
    params: PolicySearchInput,
) -> Dict[str, Any]:
    """Execute pgvector semantic search over store policies with active date filtering."""
    from app.rag.retriever import search_policies

    results = await search_policies(
        session=session,
        query=params.query,
        top_k=params.top_k,
        order_date=params.order_date,
    )
    if not results:
        raw_empty = {
            "success": True,
            "count": 0,
            "query": params.query,
            "order_date_filtered": params.order_date,
            "results": [],
            "message": "No relevant policies found matching the query.",
        }
        return PolicySearchOutput.model_validate(raw_empty).model_dump()

    formatted_results = [
        PolicyChunkResult(
            title=r.get("title", "Policy Section"),
            chunk_index=r.get("chunk_index", 0),
            content=r.get("content", ""),
            score=float(r.get("score", 0.0)),
            category=r.get("category"),
        )
        for r in results
    ]

    raw_success = {
        "success": True,
        "count": len(formatted_results),
        "query": params.query,
        "order_date_filtered": params.order_date,
        "results": formatted_results,
        "message": f"Retrieved {len(formatted_results)} relevant policy section(s).",
    }
    return PolicySearchOutput.model_validate(raw_success).model_dump()


# ─── 7. Admin Tools ───────────────────────────────────────────────────────────

async def tool_approve_refund(
    session: AsyncSession,
    current_user: Optional[User],
    params: ApproveRefundInput,
) -> Dict[str, Any]:
    """Admin tool: Approve a pending refund."""
    ref_res = await session.execute(
        select(Refund)
        .options(selectinload(Refund.order), selectinload(Refund.return_req))
        .where(Refund.id == params.refund_id.strip().upper())
    )
    refund = ref_res.scalar_one_or_none()
    if not refund:
        raw_err = {
            "success": False,
            "status": "error",
            "error": f"Refund '{params.refund_id}' was not found.",
        }
        return ApproveRefundOutput.model_validate(raw_err).model_dump()

    refund.status = "completed"
    if refund.order:
        refund.order.status = "returned"
    if refund.return_req:
        refund.return_req.status = "approved"

    await session.commit()

    raw_success = {
        "success": True,
        "status": "success",
        "refund_id": refund.id,
        "order_id": refund.order_id,
        "amount": refund.amount,
        "new_status": "completed",
        "message": f"Refund {refund.id} approved by admin {current_user.name if current_user else 'Admin'}. Order status set to returned.",
    }
    return ApproveRefundOutput.model_validate(raw_success).model_dump()


async def tool_reembed_policies(
    session: AsyncSession,
    params: ReembedPoliciesInput,
) -> Dict[str, Any]:
    """Admin tool: Trigger policy re-embedding."""
    from app.rag.ingest import ingest_policies

    count = await ingest_policies(session)
    raw_success = {
        "success": True,
        "total_chunks": count,
        "message": f"Successfully re-embedded {count} policy chunks into vector storage.",
    }
    return ReembedPoliciesOutput.model_validate(raw_success).model_dump()


async def tool_inspect_traces(
    session: AsyncSession,
    params: InspectTracesInput,
) -> Dict[str, Any]:
    """Admin tool: Inspect recent execution traces."""
    stmt = select(ExecutionTrace).order_by(desc(ExecutionTrace.created_at)).limit(params.limit)
    if params.status_filter:
        stmt = stmt.where(ExecutionTrace.status == params.status_filter)

    res = await session.execute(stmt)
    records = res.scalars().all()

    trace_list = [
        {
            "trace_id": t.trace_id,
            "session_id": t.session_id,
            "query": t.user_message,
            "latency_ms": t.total_latency_ms,
            "tokens": (t.prompt_tokens or 0) + (t.completion_tokens or 0),
            "status": t.status,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        }
        for t in records
    ]

    raw_success = {
        "success": True,
        "count": len(trace_list),
        "traces": trace_list,
    }
    return InspectTracesOutput.model_validate(raw_success).model_dump()


# ─── Tool Router with Defense-in-Depth Authorization ─────────────────────────

from app.router.tool_registry import validate_tool_authorization, ALL_TOOL_DEFINITIONS

# Expose OPENAI_TOOLS for legacy consumers
OPENAI_TOOLS = list(ALL_TOOL_DEFINITIONS.values())


async def execute_tool(
    name: str,
    args: Dict[str, Any],
    session: AsyncSession,
    current_user: Optional[User] = None,
) -> Dict[str, Any]:
    """Route and execute tool call with strict authorization and dual-sided Pydantic v2 validation."""
    user_role = current_user.role if current_user else "customer"
    logger.info(
        "Executing tool '%s' with args: %s (User: %s, Role: %s)",
        name,
        args,
        current_user.email if current_user else "anonymous",
        user_role,
    )

    # Defense-in-Depth Authorization Check
    try:
        validate_tool_authorization(name, user_role)
    except PermissionError as perm_err:
        return {"error": str(perm_err), "success": False}

    try:
        if name == "order_lookup":
            parsed = OrderLookupInput.model_validate(args)
            return await tool_order_lookup(session, current_user, parsed)

        elif name == "inventory_check":
            parsed = InventoryCheckInput.model_validate(args)
            return await tool_inventory_check(session, parsed)

        elif name == "place_order":
            parsed = PlaceOrderInput.model_validate(args)
            return await tool_place_order(session, current_user, parsed)

        elif name == "request_return":
            parsed = RequestReturnInput.model_validate(args)
            return await tool_request_return(session, current_user, parsed)

        elif name == "process_refund":
            parsed = ProcessRefundInput.model_validate(args)
            return await tool_process_refund(session, current_user, parsed)

        elif name == "policy_search":
            parsed = PolicySearchInput.model_validate(args)
            return await tool_policy_search(session, parsed)

        elif name == "approve_refund":
            parsed = ApproveRefundInput.model_validate(args)
            return await tool_approve_refund(session, current_user, parsed)

        elif name == "reembed_policies":
            parsed = ReembedPoliciesInput.model_validate(args)
            return await tool_reembed_policies(session, parsed)

        elif name == "inspect_traces":
            parsed = InspectTracesInput.model_validate(args)
            return await tool_inspect_traces(session, parsed)

        else:
            return {"error": f"Unknown tool name: '{name}'", "success": False}
    except Exception as exc:
        logger.exception("Error executing tool '%s': %s", name, exc)
        return {"error": f"Failed to execute {name}: {str(exc)}", "success": False}
