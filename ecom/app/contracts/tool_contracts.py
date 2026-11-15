"""Strict dual-sided Pydantic v2 contracts for all tools and internal service boundaries.

Enforces:
1. Validated tool inputs
2. Validated tool outputs (defense-in-depth, never blindly trusting internal data)
3. Strong typing for order items, returns, refunds, inventory, and policy results
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


# ─── 1. Order Lookup Contracts ───────────────────────────────────────────────

class OrderLookupInput(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    order_id: Optional[str] = Field(None, description="Order ID (e.g., 'ORD-1001' or 'ORD0210788')")
    email: Optional[str] = Field(None, description="Customer email address")
    status: Optional[str] = Field(
        None,
        description="Filter by order status: 'pending', 'processing', 'shipped', 'delivered', 'returned', 'cancelled'",
    )


class OrderItemDetail(BaseModel):
    model_config = ConfigDict(extra="ignore")

    product_id: str = Field(..., description="Unique product ID")
    product_name: str = Field(..., description="Product title / name")
    quantity: int = Field(..., ge=1, description="Quantity ordered")
    unit_price: float = Field(..., ge=0.0, description="Unit price in USD")
    net_amount: Optional[float] = Field(None, description="Total line item price")


class ReturnSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rma_code: str = Field(..., description="Return authorization RMA identifier")
    reason: str = Field(..., description="Customer-provided reason for return")
    status: str = Field(..., description="Return status ('requested', 'approved', 'rejected')")


class OrderDetail(BaseModel):
    model_config = ConfigDict(extra="ignore")

    order_id: str = Field(..., description="Order identifier")
    customer_email: str = Field(..., description="Customer email address")
    status: str = Field(..., description="Order status ('delivered', 'processing', etc.)")
    total_amount: float = Field(..., ge=0.0, description="Total order amount in USD")
    tracking_number: Optional[str] = Field(None, description="Shipping tracking number")
    delivery_date: Optional[str] = Field(None, description="Expected or actual delivery date ISO string")
    created_at: Optional[str] = Field(None, description="Order creation date ISO string")
    items: List[OrderItemDetail] = Field(default_factory=list, description="Line items in this order")
    returns: List[ReturnSummary] = Field(default_factory=list, description="Returns associated with order")


class OrderLookupOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    success: bool = Field(True, description="Whether query executed successfully")
    count: int = Field(..., ge=0, description="Number of orders matched")
    orders: List[OrderDetail] = Field(default_factory=list, description="List of matching orders")
    message: Optional[str] = Field(None, description="Human-readable informational message")
    error: Optional[str] = Field(None, description="Error message if operation failed")


# ─── 2. Inventory Check Contracts ───────────────────────────────────────────

class InventoryCheckInput(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    id: Optional[str] = Field(None, description="Product ID (e.g., 'PROD-001' or 'PRD0000001')")
    name: Optional[str] = Field(None, description="Product name or keyword (e.g., 'Headphones')")
    category: Optional[str] = Field(None, description="Product category")


class ProductDetail(BaseModel):
    model_config = ConfigDict(extra="ignore")

    product_id: str = Field(..., description="Product ID")
    name: str = Field(..., description="Product name")
    category: str = Field(..., description="Catalog category")
    price: float = Field(..., ge=0.0, description="Unit price in USD")
    stock_quantity: int = Field(..., ge=0, description="Available stock quantity")
    in_stock: bool = Field(..., description="Whether product is currently available")
    warranty_months: int = Field(..., ge=0, description="Manufacturer warranty duration in months")
    description: str = Field(..., description="Product description")


class InventoryCheckOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    success: bool = Field(True, description="Whether inventory query succeeded")
    count: int = Field(..., ge=0, description="Number of products matched")
    products: List[ProductDetail] = Field(default_factory=list, description="Matching products list")
    message: Optional[str] = Field(None, description="Informational message")
    error: Optional[str] = Field(None, description="Error message if query failed")


# ─── 3. Place Order Contracts ───────────────────────────────────────────────

class OrderItemInput(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    product_id: str = Field(..., description="Product ID (e.g. 'PROD-001' or 'PRD0000001')")
    quantity: int = Field(..., gt=0, description="Quantity to purchase")


class PlaceOrderInput(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    customer_email: str = Field(..., description="Customer email address")
    items: List[OrderItemInput] = Field(..., min_length=1, description="List of items to order")


class PlaceOrderOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    success: bool = Field(True, description="Whether order placement succeeded")
    status: str = Field(..., description="Operation status ('success' or 'error')")
    order_id: Optional[str] = Field(None, description="Newly generated order identifier")
    customer_email: Optional[str] = Field(None, description="Customer email")
    order_status: Optional[str] = Field(None, description="Initial order state ('processing')")
    total_amount: Optional[float] = Field(None, description="Total monetary charge")
    tracking_number: Optional[str] = Field(None, description="Assigned shipping tracking code")
    items_count: Optional[int] = Field(None, description="Number of line items ordered")
    message: Optional[str] = Field(None, description="Result message for user")
    error: Optional[str] = Field(None, description="Error details if rejected")


# ─── 4. Request Return Contracts ────────────────────────────────────────────

class RequestReturnInput(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    order_id: str = Field(..., description="Order ID to return (must be in 'delivered' status)")
    product_id: str = Field(..., description="Product ID to return")
    reason: str = Field(..., min_length=3, description="Reason for return request")


class RequestReturnOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    success: bool = Field(True, description="Whether return request succeeded")
    status: str = Field(..., description="Status ('success' or 'error')")
    rma_code: Optional[str] = Field(None, description="Generated return RMA authorization code")
    order_id: Optional[str] = Field(None, description="Referenced order ID")
    product_id: Optional[str] = Field(None, description="Product ID being returned")
    return_status: Optional[str] = Field(None, description="Current return lifecycle status")
    message: Optional[str] = Field(None, description="Instructions or confirmation message")
    error: Optional[str] = Field(None, description="Error details if rejected")


# ─── 5. Process Refund Contracts ────────────────────────────────────────────

class ProcessRefundInput(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    order_id: str = Field(..., description="Associated Order ID (e.g. 'ORD-1001')")
    rma_code: str = Field(..., description="Approved Return RMA authorization code")
    amount: float = Field(..., gt=0.0, description="Refund amount in USD")
    idempotency_key: str = Field(..., min_length=8, description="Unique idempotency key to prevent double refunds")


class ProcessRefundOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    success: bool = Field(True, description="Whether refund operation succeeded")
    status: str = Field(..., description="Status ('success', 'idempotent_duplicate', or 'error')")
    refund_id: Optional[str] = Field(None, description="Refund identifier")
    rma_code: Optional[str] = Field(None, description="Referenced RMA authorization code")
    order_id: Optional[str] = Field(None, description="Referenced Order ID")
    amount: Optional[float] = Field(None, description="Refund amount in USD")
    refund_status: Optional[str] = Field(None, description="Status ('completed' or 'pending_admin_approval')")
    requires_admin_approval: bool = Field(False, description="True if amount > $50 threshold")
    message: Optional[str] = Field(None, description="Informational message")
    error: Optional[str] = Field(None, description="Error details if rejected")


# ─── 6. Policy Search Contracts ─────────────────────────────────────────────

class PolicySearchInput(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    query: str = Field(..., description="Policy question or search keywords")
    top_k: int = Field(3, ge=1, le=10, description="Maximum number of policy sections to retrieve")
    order_date: Optional[str] = Field(None, description="Order purchase date to filter applicable policies")


class PolicyChunkResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(..., description="Policy title or document name")
    chunk_index: int = Field(..., ge=0, description="Chunk sequence index")
    content: str = Field(..., description="Policy text chunk")
    score: float = Field(..., description="Cosine similarity score")
    category: Optional[str] = Field(None, description="Policy category")


class PolicySearchOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    success: bool = Field(True, description="Whether policy search succeeded")
    count: int = Field(..., ge=0, description="Number of policy chunks matched")
    query: str = Field(..., description="Original search query")
    order_date_filtered: Optional[str] = Field(None, description="Order date used for temporal filtering")
    results: List[PolicyChunkResult] = Field(default_factory=list, description="Matching policy sections")
    message: Optional[str] = Field(None, description="Informational message")
    error: Optional[str] = Field(None, description="Error details if search failed")


# ─── 7. Admin-Only Tool Contracts ───────────────────────────────────────────

class ApproveRefundInput(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    refund_id: str = Field(..., description="Refund ID to approve (e.g., 'REF-5001')")


class ApproveRefundOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    success: bool = Field(True, description="Whether approval succeeded")
    status: str = Field(..., description="Status ('success' or 'error')")
    refund_id: Optional[str] = Field(None, description="Approved refund ID")
    order_id: Optional[str] = Field(None, description="Associated order ID")
    amount: Optional[float] = Field(None, description="Approved amount")
    new_status: Optional[str] = Field(None, description="New status ('completed')")
    message: Optional[str] = Field(None, description="Approval message")
    error: Optional[str] = Field(None, description="Error details if rejected")


class ReembedPoliciesInput(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    force_all: bool = Field(False, description="Whether to bypass incremental hashing and re-embed all documents")


class ReembedPoliciesOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    success: bool = Field(True, description="Whether re-embedding succeeded")
    total_chunks: int = Field(..., ge=0, description="Number of policy chunks embedded")
    message: str = Field(..., description="Result message")
    error: Optional[str] = Field(None, description="Error details if failed")


class InspectTracesInput(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    limit: int = Field(10, ge=1, le=50, description="Maximum number of recent traces to return")
    status_filter: Optional[str] = Field(None, description="Filter traces by execution status")


class InspectTracesOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    success: bool = Field(True, description="Whether trace inspection succeeded")
    count: int = Field(..., ge=0, description="Number of traces retrieved")
    traces: List[Dict[str, Any]] = Field(default_factory=list, description="Recent trace records")
    error: Optional[str] = Field(None, description="Error details if failed")

