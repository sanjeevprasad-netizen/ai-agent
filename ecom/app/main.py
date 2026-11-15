"""FastAPI application entrypoint for ecom.

Mounts static UI files, registers authentication, agent orchestration,
and administrative refund & trace inspection endpoints.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user, require_admin, router as auth_router
from app.config import settings, STATIC_DIR
from app.db.models import ChatMessage, ChatSession, ExecutionTrace, Order, Refund, Return, User, SupportTicket
from app.db.session import get_db_session, init_db
from app.agent.orchestrator import orchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ecom.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: initialize schema and seed on startup."""
    logger.info("Starting ecom service (Unified Python Architecture)...")
    try:
        from app.db.seed import seed_database
        await init_db()
        await seed_database(force=False)
    except Exception as e:
        logger.error("Database startup initialization encountered an issue: %s", e)
    yield
    logger.info("Shutting down ecom service...")


app = FastAPI(
    title="ecom - Unified E-Commerce ",
    version="2.0.0",
    description="Unified Python e-commerce support system with PostgreSQL + pgvector, MCP tools & ReAct agent.",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Auth Router
app.include_router(auth_router)


# ─── Pydantic Request/Response Models ─────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Customer question or instruction")
    session_id: Optional[str] = Field(None, description="Active chat session UUID")


class ChatResponse(BaseModel):
    session_id: str
    response: str
    tool_calls: List[Dict[str, Any]] = []
    trace: Dict[str, Any]


class SessionSummary(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class MessageItem(BaseModel):
    id: int
    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    created_at: datetime


# ─── Chat & Agent Endpoints ───────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Send message to AI , run ReAct loop with guardrails, and return grounded answer."""
    result = await orchestrator.run(
        db_session=db,
        user=current_user,
        user_message=request.message,
        session_id=request.session_id,
    )
    return ChatResponse(
        session_id=result["session_id"],
        response=result["response"],
        tool_calls=result["tool_calls"],
        trace=result["trace"],
    )


@app.get("/api/chat/sessions", response_model=List[SessionSummary])
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Retrieve chat history sessions for current user (or all if admin)."""
    stmt = select(ChatSession)
    if current_user.role != "admin":
        stmt = stmt.where(ChatSession.user_id == current_user.id)
    stmt = stmt.order_by(desc(ChatSession.updated_at)).limit(30)

    result = await db.execute(stmt)
    sessions = result.scalars().all()
    return [
        SessionSummary(
            id=s.id,
            title=s.title,
            created_at=s.created_at,
            updated_at=s.updated_at,
        )
        for s in sessions
    ]


@app.get("/api/chat/sessions/{session_id}/messages", response_model=List[MessageItem])
async def get_session_messages(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Retrieve all messages in a specific session, verifying ownership."""
    sess_res = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    chat_sess = sess_res.scalar_one_or_none()

    if not chat_sess:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    if current_user.role != "admin" and chat_sess.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden: Cannot view another customer's conversation.")

    msg_res = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.asc())
    )
    messages = msg_res.scalars().all()

    output = []
    for m in messages:
        t_calls = None
        if m.tool_calls_json:
            try:
                t_calls = json.loads(m.tool_calls_json)
            except Exception:
                t_calls = None
        output.append(
            MessageItem(
                id=m.id,
                role=m.role,
                content=m.content,
                tool_calls=t_calls,
                created_at=m.created_at,
            )
        )
    return output


# ─── Admin Endpoints ──────────────────────────────────────────────────────────

@app.get("/api/admin/refunds/pending")
async def list_pending_refunds(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """List all refunds currently awaiting administrator approval (> $50 limit)."""
    stmt = (
        select(Refund)
        .options(selectinload(Refund.order), selectinload(Refund.return_req))
        .where(Refund.status == "pending_admin_approval")
        .order_by(desc(Refund.created_at))
    )
    result = await db.execute(stmt)
    refunds = result.scalars().all()

    output = []
    for r in refunds:
        output.append({
            "refund_id": r.id,
            "rma_code": r.return_id,
            "order_id": r.order_id,
            "amount": r.amount,
            "status": r.status,
            "customer_email": r.order.customer_email if r.order else "unknown",
            "reason": r.return_req.reason if r.return_req else "",
            "created_at": r.created_at.isoformat(),
        })
    return {"pending_count": len(output), "refunds": output}


@app.post("/api/admin/refunds/{refund_id}/approve")
async def approve_refund(
    refund_id: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """Admin endpoint: approve a pending refund."""
    ref_res = await db.execute(
        select(Refund)
        .options(selectinload(Refund.order), selectinload(Refund.return_req))
        .where(Refund.id == refund_id.strip().upper())
    )
    refund_obj = ref_res.scalar_one_or_none()

    if not refund_obj:
        raise HTTPException(status_code=404, detail=f"Refund '{refund_id}' not found.")

    if refund_obj.status == "completed":
        return {"message": "Refund is already completed.", "refund_id": refund_obj.id, "status": "completed"}

    refund_obj.status = "completed"
    if refund_obj.order:
        refund_obj.order.status = "returned"
    if refund_obj.return_req:
        refund_obj.return_req.status = "approved"

    await db.commit()
    logger.info("Admin %s approved refund %s for $%s", admin.email, refund_obj.id, refund_obj.amount)

    return {
        "status": "success",
        "refund_id": refund_obj.id,
        "order_id": refund_obj.order_id,
        "amount": refund_obj.amount,
        "new_status": "completed",
        "message": f"Refund {refund_obj.id} approved by admin {admin.name}. Order status updated to returned.",
    }


@app.post("/api/admin/reembed")
@app.post("/api/policies/reindex")
async def trigger_reembed(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """Admin endpoint: dynamically re-chunk, re-embed, and update all policies in pgvector."""
    logger.info("Admin %s requested dynamic re-embedding of policies...", admin.email)
    try:
        from app.rag.ingest import ingest_policies
        count = await ingest_policies(db)
        return {
            "status": "success",
            "message": f"Successfully re-chunked, embedded, and indexed {count} policy sections into PostgreSQL.",
            "chunks_count": count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception("Dynamic re-embedding failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Dynamic re-embedding failed: {str(e)}",
        )


@app.get("/api/admin/traces")
async def list_execution_traces(
    limit: int = 15,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """Admin endpoint: retrieve recent agent execution traces for observability."""
    stmt = select(ExecutionTrace).order_by(desc(ExecutionTrace.created_at)).limit(limit)
    res = await db.execute(stmt)
    traces = res.scalars().all()

    output = []
    for t in traces:
        steps = []
        try:
            steps = json.loads(t.steps_json)
        except Exception:
            steps = []
        u_details = None
        if t.user_details:
            try:
                u_details = json.loads(t.user_details)
            except Exception:
                u_details = {"raw": t.user_details}
        output.append({
            "trace_id": t.trace_id,
            "session_id": t.session_id,
            "user_details": u_details,
            "user_message": t.user_message,
            "final_response": t.final_response,
            "total_latency_ms": t.total_latency_ms,
            "prompt_tokens": t.prompt_tokens,
            "completion_tokens": t.completion_tokens,
            "estimated_cost_usd": t.estimated_cost_usd,
            "status": t.status,
            "steps": steps,
            "created_at": t.created_at.isoformat(),
        })
    return {"count": len(output), "traces": output}


@app.get("/api/admin/tickets")
async def list_support_tickets(
    limit: int = 20,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """Admin endpoint: retrieve auto-generated support tickets."""
    stmt = select(SupportTicket).order_by(desc(SupportTicket.created_at)).limit(limit)
    res = await db.execute(stmt)
    tickets = res.scalars().all()
    output = [
        {
            "ticket_id": t.id,
            "customer_id": t.customer_id,
            "request_summary": t.request_summary,
            "failure_category": t.failure_category,
            "failed_tool": t.failed_tool,
            "status": t.status,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        }
        for t in tickets
    ]
    return {"count": len(output), "tickets": output}


@app.get("/api/health")
async def health():
    """Healthcheck endpoint."""
    return {
        "status": "healthy",
        "service": "ecom",
        "version": "2.0.0",
        "database": "postgresql+asyncpg" if settings.is_postgres else "sqlite",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
    }


# ─── Static Files & Frontend SPA ─────────────────────────────────────────────

# Mount /static directory
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    """Serve frontend SPA index.html."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "ecom API is active. Frontend static files pending."}

