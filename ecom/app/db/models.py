"""SQLAlchemy models for ecom application.

Includes:
- Kaggle dataset entities: Location, PaymentMethod, FulfillmentMethod, Seller, Customer, Product, Order, OrderItem, Return
- Authentication & Security: User (credentials linked to Customer)
- Transactional financial records: Refund
- Knowledge base: Policy (pgvector), IngestionMetadata
- Observability & Chat: ChatSession, ChatMessage, ExecutionTrace, SupportTicket
"""

from datetime import datetime, timezone
import uuid
from typing import Any, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Boolean,
    func,
    JSON,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class VectorType(TypeDecorator):
    """Vector type using pgvector on PostgreSQL when available and JSON otherwise."""

    impl = JSON
    cache_ok = True
    HAS_PGVECTOR: bool = False

    def __init__(self, dim: int = 384, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dim = dim

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql" and VectorType.HAS_PGVECTOR:
            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "tolist"):
            return value.tolist()
        return list(value)

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        return value


# ─── 1. Dimension Tables ──────────────────────────────────────────────────────

class Location(Base):
    __tablename__ = "locations"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # e.g., 'LOC0001'
    state: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    city: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    postal_code: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    region: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    area_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    location_category: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)


class PaymentMethod(Base):
    __tablename__ = "payment_methods"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # e.g., 'PAY001'
    payment_method: Mapped[str] = mapped_column(String(50), nullable=False)
    payment_provider: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class FulfillmentMethod(Base):
    __tablename__ = "fulfillment_methods"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # e.g., 'FUL001'
    shipping_method: Mapped[str] = mapped_column(String(50), nullable=False)
    service_level: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    delivery_sla_days: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    base_shipping_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


class Seller(Base):
    __tablename__ = "sellers"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # e.g., 'SELL0001'
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    phone_number: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    join_date: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    location_id: Mapped[Optional[str]] = mapped_column(String(16), ForeignKey("locations.id", ondelete="SET NULL"), nullable=True)
    rating: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=0.0)
    category_focus: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)


# ─── 2. Customer & Authentication Models ──────────────────────────────────────

class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # e.g., 'CUST000001'
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    phone_number: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    gender: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    date_of_birth: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    registration_date: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    income_bracket: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    marital_status: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    location_id: Mapped[Optional[str]] = mapped_column(String(16), ForeignKey("locations.id", ondelete="SET NULL"), nullable=True)
    upi_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    credit_card_number: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Relationships
    orders: Mapped[List["Order"]] = relationship("Order", back_populates="customer", cascade="all, delete-orphan")
    returns: Mapped[List["Return"]] = relationship("Return", back_populates="customer", cascade="all, delete-orphan")
    user: Mapped[Optional["User"]] = relationship("User", back_populates="customer", uselist=False)


class User(Base):
    """Authentication and credentials table.
    
    Stores login credentials (email/password) mapped to Kaggle Customer records or Admin profiles.
    """
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    customer_id: Mapped[Optional[str]] = mapped_column(String(32), ForeignKey("customers.id", ondelete="CASCADE"), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="customer")  # 'admin' | 'customer'
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    customer: Mapped[Optional["Customer"]] = relationship("Customer", back_populates="user")
    sessions: Mapped[List["ChatSession"]] = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")


# ─── 3. Catalog Products ──────────────────────────────────────────────────────

class Product(Base):
    __tablename__ = "products"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # e.g., 'PROD000001'
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    brand: Mapped[str] = mapped_column(String(100), nullable=False, index=True, default="Generic")
    category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    sub_category: Mapped[str] = mapped_column(String(100), nullable=False, index=True, default="General")
    price: Mapped[float] = mapped_column(Float, nullable=False)
    stock_quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=50)
    warranty_months: Mapped[int] = mapped_column(Integer, nullable=False, default=12)
    description: Mapped[str] = mapped_column(Text, nullable=False)


# ─── 4. Orders & Order Items ──────────────────────────────────────────────────

class Order(Base):
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # e.g., 'ORD0210788'
    customer_id: Mapped[str] = mapped_column(String(32), ForeignKey("customers.id"), nullable=False, index=True)
    customer_email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    order_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="processing",
        index=True,
    )  # 'completed', 'delivered', 'shipped', 'processing', 'returned', 'cancelled'
    total_amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    shipping_fee: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    tax_amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    discount_amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    tracking_number: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    expected_delivery_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    actual_delivery_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    delivery_sla_days: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    delivery_delay_days: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    payment_id: Mapped[Optional[str]] = mapped_column(String(16), ForeignKey("payment_methods.id", ondelete="SET NULL"), nullable=True)
    fulfillment_id: Mapped[Optional[str]] = mapped_column(String(16), ForeignKey("fulfillment_methods.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    @property
    def delivery_date(self) -> Optional[datetime]:
        return self.actual_delivery_date or self.expected_delivery_date

    # Relationships
    customer: Mapped["Customer"] = relationship("Customer", back_populates="orders")
    items: Mapped[List["OrderItem"]] = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan", lazy="selectin")
    returns: Mapped[List["Return"]] = relationship("Return", back_populates="order", cascade="all, delete-orphan", lazy="selectin")
    refunds: Mapped[List["Refund"]] = relationship("Refund", back_populates="order", cascade="all, delete-orphan", lazy="selectin")


class OrderItem(Base):
    __tablename__ = "order_items"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # e.g., 'OL00000001'
    order_id: Mapped[str] = mapped_column(String(32), ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, index=True)
    product_id: Mapped[str] = mapped_column(String(32), ForeignKey("products.id"), nullable=False, index=True)
    product_name: Mapped[str] = mapped_column(String(255), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    unit_price: Mapped[float] = mapped_column(Float, nullable=False)
    discount_percentage: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    tax_percentage: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    net_amount: Mapped[float] = mapped_column(Float, nullable=False)

    # Relationships
    order: Mapped["Order"] = relationship("Order", back_populates="items")
    product: Mapped["Product"] = relationship("Product")


# ─── 5. Returns & Refunds ─────────────────────────────────────────────────────

class Return(Base):
    __tablename__ = "returns"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # e.g., 'RMA-9001' or 'RET-OL00000008'
    order_id: Mapped[str] = mapped_column(String(32), ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, index=True)
    order_line_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    customer_id: Mapped[str] = mapped_column(String(32), ForeignKey("customers.id"), nullable=False, index=True)
    product_id: Mapped[Optional[str]] = mapped_column(String(32), ForeignKey("products.id"), nullable=True, index=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="requested",
    )  # 'requested', 'approved', 'rejected'
    days_to_return: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    restocking_fee: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    order: Mapped["Order"] = relationship("Order", back_populates="returns")
    customer: Mapped["Customer"] = relationship("Customer", back_populates="returns")
    product: Mapped[Optional["Product"]] = relationship("Product")
    refunds: Mapped[List["Refund"]] = relationship("Refund", back_populates="return_req", cascade="all, delete-orphan", lazy="selectin")


class Refund(Base):
    __tablename__ = "refunds"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # e.g., 'REF-5001'
    return_id: Mapped[str] = mapped_column(String(32), ForeignKey("returns.id", ondelete="CASCADE"), nullable=False, index=True)
    order_id: Mapped[str] = mapped_column(String(32), ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, index=True)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(
        String(35),
        nullable=False,
        default="pending_admin_approval",
    )  # 'completed', 'pending_admin_approval'
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    return_req: Mapped["Return"] = relationship("Return", back_populates="refunds")
    order: Mapped["Order"] = relationship("Order", back_populates="refunds")


# ─── 6. Knowledge Base & Ingestion Metadata ───────────────────────────────────

class Policy(Base):
    __tablename__ = "policies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    format: Mapped[str] = mapped_column(String(20), nullable=False)  # 'markdown', 'json', 'pdf'
    source_file: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(20), nullable=False, default="1.0")
    effective_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    effective_to: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    embedding: Mapped[List[float]] = mapped_column(VectorType(384), nullable=False)
    document_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    chunk_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    breadcrumbs: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)


class IngestionMetadata(Base):
    __tablename__ = "ingestion_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    source_name: Mapped[str] = mapped_column(String(255), nullable=False)
    document_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    content_version: Mapped[str] = mapped_column(String(20), nullable=False, default="1.0")
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_model_version: Mapped[str] = mapped_column(String(50), nullable=False, default="1.0")
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    last_modified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ingestion_status: Mapped[str] = mapped_column(String(20), nullable=False, default="success")


# ─── 7. Chat, Observability & Tickets ─────────────────────────────────────────

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="New Conversation")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="sessions")
    messages: Mapped[List["ChatMessage"]] = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="ChatMessage.id",
    )
    traces: Mapped[List["ExecutionTrace"]] = relationship(
        "ExecutionTrace",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="ExecutionTrace.created_at.desc()",
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tool_calls_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")


class ExecutionTrace(Base):
    __tablename__ = "execution_traces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("chat_sessions.id"), nullable=True, index=True)
    user_details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    final_response: Mapped[str] = mapped_column(Text, nullable=False)
    total_latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="success")
    steps_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    # Relationships
    session: Mapped[Optional["ChatSession"]] = relationship("ChatSession", back_populates="traces")


class SupportTicket(Base):
    __tablename__ = "support_tickets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # e.g., 'TCK-1001'
    customer_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    request_summary: Mapped[str] = mapped_column(Text, nullable=False)
    failure_category: Mapped[str] = mapped_column(String(50), nullable=False)
    failed_tool: Mapped[str] = mapped_column(String(50), nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    diagnostic_context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="open")  # 'open', 'resolved', 'closed'
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
