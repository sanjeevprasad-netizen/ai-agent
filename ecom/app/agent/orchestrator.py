"""Stateful Agent Orchestrator with ReAct loop, two-tier intent routing, guardrails, resilience, and telemetry.

Implements:
1. Security Guardrails:
   - Multi-vector prompt injection & jailbreak defense.
   - Business scope enforcement (Strictly 5 domains: Product Info, Shipping, Orders, Returns, Refunds).
2. Two-Tier Intent Router:
   - FAST_PATH: Deterministic sub-80ms execution for routine order/inventory queries (Zero LLM calls).
   - SLOW_PATH: ReAct agent for complex multi-step reasoning, returns, refunds, and policy queries.
3. Dynamic Role-Based Tool Scoping:
   - Exposes only authorized tools to the LLM based on authenticated user's role.
4. Resilience & Graceful Degradation:
   - Bounded retries for transient operations.
   - Automated support ticket generation on persistent failures.
5. Observability & Telemetry:
   - PII-redacted, non-blocking telemetry logging with fail-open safety.
   - ExecutionTrace and ChatMessage persistence in PostgreSQL.
"""

from datetime import datetime, timezone
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings

try:
    from langfuse.openai import AsyncOpenAI
except ImportError:
    from openai import AsyncOpenAI
import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import ChatMessage, ChatSession, ExecutionTrace, User
from app.tools.mcp_tools import execute_tool
from app.router.intent_router import (
    DeterministicIntentRouter,
    RouteTarget,
    execute_fast_path,
)
from app.router.tool_registry import get_tools_for_role
from app.security.guardrails import SecurityGuardrails
from app.security.pii import redact_pii_text
from app.resilience.retry import execute_with_bounded_retry
from app.resilience.degradation import create_support_ticket, build_degraded_response
from app.observability.langfuse_client import telemetry
from app.agent.prompts import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    OFF_TOPIC_RESPONSE,
    GREETING_RESPONSE,
)
from app.agent.tracer import ActiveTrace

logger = logging.getLogger(__name__)

# Greeting regex
GREETING_REGEX = re.compile(
    r"^(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening|day))[\s!.,?]*$",
    re.IGNORECASE,
)


def triage_greeting(user_message: str) -> Optional[str]:
    """Check for simple greetings or thank yous."""
    clean_msg = user_message.strip()
    if GREETING_REGEX.match(clean_msg):
        return GREETING_RESPONSE
    if re.match(r"^(thanks|thank\s+you|thx)[\s!.,]*$", clean_msg, re.IGNORECASE):
        return "You're very welcome! Let me know if you need any other assistance with your orders, returns, or products."
    return None


def normalize_tool_arguments(tool_name: str, raw_args: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and normalize tool arguments."""
    args = dict(raw_args)
    for key in ("properties", "parameters", "arguments", "input"):
        if key in args and isinstance(args[key], dict) and not isinstance(args[key], list):
            args = dict(args[key])

    clean: Dict[str, Any] = {}
    for k, v in args.items():
        if isinstance(v, dict) and not v:
            continue
        clean[k] = v

    return clean


class AgentOrchestrator:
    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None
        self.model_name = settings.LLM_MODEL
        self.router = DeterministicIntentRouter()
        self.guardrails = SecurityGuardrails()

    @property
    def client(self) -> AsyncOpenAI:
        """Lazy-load the AsyncOpenAI client within the active event loop."""
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=settings.LLM_BASE_URL,
                api_key=settings.LLM_API_KEY,
                timeout=httpx.Timeout(settings.LLM_TIMEOUT, connect=5.0),
            )
        return self._client

    async def get_or_create_session(
        self,
        db_session: AsyncSession,
        user: User,
        session_id: Optional[str] = None,
        first_message: str = "New Chat",
    ) -> ChatSession:
        """Fetch existing session or create a new session for the user."""
        if session_id:
            res = await db_session.execute(
                select(ChatSession).where(
                    ChatSession.id == session_id,
                    ChatSession.user_id == user.id if user.role != "admin" else True,
                )
            )
            chat_sess = res.scalar_one_or_none()
            if chat_sess:
                return chat_sess

        title = first_message.strip()[:40] if first_message else "New Conversation"
        chat_sess = ChatSession(
            user_id=user.id,
            title=title,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db_session.add(chat_sess)
        await db_session.flush()
        return chat_sess

    async def run(
        self,
        db_session: AsyncSession,
        user: User,
        user_message: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute full request lifecycle: Guardrails -> Triage -> Fast Path -> ReAct Agent."""
        chat_session = await self.get_or_create_session(
            db_session, user, session_id, first_message=user_message
        )
        user_info = {
            "user_id": user.id,
            "name": user.name,
            "email": user.email,
            "role": user.role,
        }
        user_details_json = json.dumps(user_info)
        trace = ActiveTrace(
            chat_session.id,
            user_message,
            model_name=self.model_name,
            user_details=user_info,
        )

        # ── 1. Security & Business Scope Guardrails ───────────────────────────
        guardrail_eval = await self.guardrails.evaluate_async(user_message)
        if not guardrail_eval.allowed:
            step = trace.start_step(
                "security_guardrail",
                f"Guardrail Refusal: {guardrail_eval.refusal_reason}",
                {"query": user_message},
            )
            step.finish(output=guardrail_eval.refusal_message)

            user_msg_record = ChatMessage(
                session_id=chat_session.id,
                role="user",
                content=user_message,
                created_at=datetime.now(timezone.utc),
            )
            asst_msg_record = ChatMessage(
                session_id=chat_session.id,
                role="assistant",
                content=guardrail_eval.refusal_message,
                created_at=datetime.now(timezone.utc),
            )
            db_session.add(user_msg_record)
            db_session.add(asst_msg_record)

            final_trace = trace.finalize(guardrail_eval.refusal_message, status="guardrail_blocked")
            db_trace = ExecutionTrace(
                session_id=chat_session.id,
                user_details=user_details_json,
                trace_id=final_trace["trace_id"],
                user_message=user_message,
                final_response=guardrail_eval.refusal_message,
                total_latency_ms=final_trace["total_latency_ms"],
                prompt_tokens=0,
                completion_tokens=0,
                estimated_cost_usd=0.0,
                status="guardrail_blocked",
                steps_json=final_trace["steps_json"],
                created_at=datetime.now(timezone.utc),
            )
            db_session.add(db_trace)
            await db_session.commit()

            telemetry.log_interaction_async(
                trace_id=final_trace["trace_id"],
                session_id=chat_session.id,
                user_id=user.id,
                user_email=user.email,
                user_message=user_message,
                final_response=guardrail_eval.refusal_message,
                route="guardrail_blocked",
                latency_ms=final_trace["total_latency_ms"],
                status="guardrail_blocked",
            )

            return {
                "session_id": chat_session.id,
                "response": guardrail_eval.refusal_message,
                "tool_calls": [],
                "trace": final_trace,
                "fast_path": False,
            }

        # ── 2. Greeting Triage ────────────────────────────────────────────────
        greeting_reply = triage_greeting(user_message)
        if greeting_reply:
            step = trace.start_step("triage_direct", "Greeting response", {"query": user_message})
            step.finish(output=greeting_reply)

            user_msg_record = ChatMessage(
                session_id=chat_session.id,
                role="user",
                content=user_message,
                created_at=datetime.now(timezone.utc),
            )
            asst_msg_record = ChatMessage(
                session_id=chat_session.id,
                role="assistant",
                content=greeting_reply,
                created_at=datetime.now(timezone.utc),
            )
            db_session.add(user_msg_record)
            db_session.add(asst_msg_record)

            final_trace = trace.finalize(greeting_reply, status="triage_direct")
            db_trace = ExecutionTrace(
                session_id=chat_session.id,
                user_details=user_details_json,
                trace_id=final_trace["trace_id"],
                user_message=user_message,
                final_response=greeting_reply,
                total_latency_ms=final_trace["total_latency_ms"],
                prompt_tokens=0,
                completion_tokens=0,
                estimated_cost_usd=0.0,
                status="triage_direct",
                steps_json=final_trace["steps_json"],
                created_at=datetime.now(timezone.utc),
            )
            db_session.add(db_trace)
            await db_session.commit()

            telemetry.log_interaction_async(
                trace_id=final_trace["trace_id"],
                session_id=chat_session.id,
                user_id=user.id,
                user_email=user.email,
                user_message=user_message,
                final_response=greeting_reply,
                route="triage",
                latency_ms=final_trace["total_latency_ms"],
                status="triage_direct",
            )

            return {
                "session_id": chat_session.id,
                "response": greeting_reply,
                "tool_calls": [],
                "trace": final_trace,
                "fast_path": False,
            }

        # ── 3. Deterministic Fast Path ────────────────────────────────────────
        decision = self.router.classify(user_message, current_user=user)
        if decision.target == RouteTarget.FAST_PATH:
            logger.info("Fast-Path routed for intent '%s' (User: %s)", decision.intent.value if decision.intent else "unknown", user.email)
            step = trace.start_step(
                "fast_path_execution",
                f"Fast-Path Dispatch: {decision.intent.value if decision.intent else 'unknown'}",
                decision.extracted_params or {},
            )
            fast_reply, fast_tool_calls, elapsed_tool_ms = await execute_fast_path(
                decision, db_session, user
            )
            step.finish(output={"response": fast_reply, "tool_calls": fast_tool_calls})

            user_msg_record = ChatMessage(
                session_id=chat_session.id,
                role="user",
                content=user_message,
                created_at=datetime.now(timezone.utc),
            )
            asst_msg_record = ChatMessage(
                session_id=chat_session.id,
                role="assistant",
                content=fast_reply,
                tool_calls_json=json.dumps(fast_tool_calls),
                created_at=datetime.now(timezone.utc),
            )
            db_session.add(user_msg_record)
            db_session.add(asst_msg_record)

            final_trace = trace.finalize(fast_reply, status="fast_path_success")
            db_trace = ExecutionTrace(
                session_id=chat_session.id,
                user_details=user_details_json,
                trace_id=final_trace["trace_id"],
                user_message=user_message,
                final_response=fast_reply,
                total_latency_ms=final_trace["total_latency_ms"],
                prompt_tokens=0,
                completion_tokens=0,
                estimated_cost_usd=0.0,
                status="fast_path_success",
                steps_json=final_trace["steps_json"],
                created_at=datetime.now(timezone.utc),
            )
            db_session.add(db_trace)
            await db_session.commit()

            telemetry.log_interaction_async(
                trace_id=final_trace["trace_id"],
                session_id=chat_session.id,
                user_id=user.id,
                user_email=user.email,
                user_message=user_message,
                final_response=fast_reply,
                route="fast_path",
                latency_ms=final_trace["total_latency_ms"],
                tool_calls=fast_tool_calls,
                status="fast_path_success",
            )

            return {
                "session_id": chat_session.id,
                "response": fast_reply,
                "tool_calls": fast_tool_calls,
                "trace": final_trace,
                "fast_path": True,
            }

        # ── 4. Slow Path: ReAct Agent with Role Scoping & Resilience ──────────
        logger.info("Routing query to Slow Path ReAct Agent (User: %s, Role: %s)", user.email, user.role)
        role_restriction = (
            "- Customer Scoping: You are scoped ONLY to orders for " + user.email +
            ". Deny requests to view, return, or refund orders for any other customer."
            if user.role == "customer"
            else "- Admin Privileges: You can inspect any order, view pending refunds, and examine traces."
        )

        sys_prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(
            user_name=user.name,
            user_email=user.email,
            user_role=user.role,
            role_restriction=role_restriction,
        )

        role_scoped_tools = get_tools_for_role(user.role)

        res_history = await db_session.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == chat_session.id)
            .order_by(ChatMessage.id.desc())
            .limit(10)
        )
        recent_messages = list(reversed(res_history.scalars().all()))

        messages: List[Dict[str, Any]] = [{"role": "system", "content": sys_prompt}]
        for m in recent_messages:
            messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": user_message})

        executed_tool_calls: List[Dict[str, Any]] = []
        executed_signatures: set[str] = set()
        iterations = 0
        final_text = ""

        while iterations < settings.MAX_ITERATIONS:
            iterations += 1
            llm_step = trace.start_step(
                "llm_reasoning",
                f"LLM Step {iterations}",
                {"messages_count": len(messages)},
            )

            try:
                completion = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=role_scoped_tools,
                    tool_choice="auto",
                    temperature=0.1,
                    name="react-llm-reasoning",
                    trace_id=trace.trace_id,
                )
            except Exception as e:
                logger.warning("LLM API call unavailable or timed out (%s). Activating local RAG degradation...", e)
                llm_step.finish(output={"error": str(e)}, is_error=True)
                try:
                    from app.rag.retriever import retrieve_policies
                    policy_matches = await retrieve_policies(db_session, user_message, top_k=2)
                    if policy_matches:
                        top_p = policy_matches[0]
                        final_text = (
                            f"### Policy Information: **{top_p.get('title', 'Store Policy')}**\n\n"
                            f"{top_p.get('content', '')}\n\n"
                            f"*(Retrieved via Offline Hybrid RAG from `{top_p.get('source_file')}`)*"
                        )
                    else:
                        ticket_id = await create_support_ticket(
                            session=db_session,
                            user=user,
                            tool_name="llm_reasoning",
                            failure_category="ai_reasoning_outage",
                            request_summary=f"AI reasoning outage on user query: {user_message[:200]}",
                            diagnostic_context={"error": str(e)},
                        )
                        final_text = build_degraded_response(
                            ticket_id=ticket_id,
                            tool_name="llm_reasoning",
                        )
                except Exception as fallback_err:
                    logger.exception("Fallback degradation failed: %s", fallback_err)
                    final_text = "Our customer support team is available to assist you. A support ticket has been scheduled."
                break

            response_msg = completion.choices[0].message
            usage = completion.usage
            prompt_tok = usage.prompt_tokens if usage else 0
            comp_tok = usage.completion_tokens if usage else 0

            llm_step.finish(
                output={"content": response_msg.content, "tool_calls_count": len(response_msg.tool_calls or [])},
                prompt_tokens=prompt_tok,
                completion_tokens=comp_tok,
            )

            if not response_msg.tool_calls:
                final_text = response_msg.content or "How else can I assist you?"
                break

            tool_calls_data = []
            for tc in response_msg.tool_calls:
                tool_calls_data.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

            messages.append({
                "role": "assistant",
                "content": response_msg.content,
                "tool_calls": tool_calls_data,
            })

            for tc in response_msg.tool_calls:
                fn_name = tc.function.name
                raw_args_str = tc.function.arguments or "{}"
                try:
                    raw_args = json.loads(raw_args_str)
                except Exception:
                    raw_args = {}

                norm_args = normalize_tool_arguments(fn_name, raw_args)
                sig = f"{fn_name}:{json.dumps(norm_args, sort_keys=True)}"

                if sig in executed_signatures:
                    logger.warning("Loop protection: duplicate call %s skipped.", sig)
                    tool_result = {"error": f"Tool '{fn_name}' was already executed with identical parameters in this turn."}
                else:
                    executed_signatures.add(sig)
                    t_step = trace.start_step("tool_call", f"Tool: {fn_name}", norm_args)

                    # Bounded Retry & Degradation protection
                    try:
                        async def call_tool():
                            return await execute_tool(fn_name, norm_args, db_session, current_user=user)

                        tool_result = await execute_with_bounded_retry(
                            call_tool, max_retries=2, base_delay_ms=50.0
                        )
                    except Exception as exc:
                        logger.exception("Persistent failure on tool %s: %s", fn_name, exc)
                        ticket_id = await create_support_ticket(
                            db_session, user, fn_name, "tool_execution_failure", user_message, {"error": str(exc)}
                        )
                        tool_result = {
                            "error": f"Tool failed after retries. Created support ticket {ticket_id}.",
                            "ticket_id": ticket_id,
                            "degraded_message": build_degraded_response(ticket_id, fn_name),
                        }

                    t_step.finish(output=tool_result)

                executed_tool_calls.append({
                    "tool": fn_name,
                    "args": norm_args,
                    "result": tool_result,
                    "duration_ms": t_step.duration_ms,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result),
                })

        user_msg_record = ChatMessage(
            session_id=chat_session.id,
            role="user",
            content=user_message,
            created_at=datetime.now(timezone.utc),
        )
        asst_msg_record = ChatMessage(
            session_id=chat_session.id,
            role="assistant",
            content=final_text,
            tool_calls_json=json.dumps(executed_tool_calls) if executed_tool_calls else None,
            created_at=datetime.now(timezone.utc),
        )
        db_session.add(user_msg_record)
        db_session.add(asst_msg_record)

        final_trace = trace.finalize(final_text, status="completed")
        db_trace = ExecutionTrace(
            session_id=chat_session.id,
            user_details=user_details_json,
            trace_id=final_trace["trace_id"],
            user_message=user_message,
            final_response=final_text,
            total_latency_ms=final_trace["total_latency_ms"],
            prompt_tokens=final_trace["prompt_tokens"],
            completion_tokens=final_trace["completion_tokens"],
            estimated_cost_usd=final_trace["estimated_cost_usd"],
            status=final_trace["status"],
            steps_json=final_trace["steps_json"],
            created_at=datetime.now(timezone.utc),
        )
        db_session.add(db_trace)
        await db_session.commit()

        telemetry.log_interaction_async(
            trace_id=final_trace["trace_id"],
            session_id=chat_session.id,
            user_id=user.id,
            user_email=user.email,
            user_message=user_message,
            final_response=final_text,
            route="slow_path",
            latency_ms=final_trace["total_latency_ms"],
            prompt_tokens=final_trace["prompt_tokens"],
            completion_tokens=final_trace["completion_tokens"],
            tool_calls=executed_tool_calls,
            status="completed",
        )

        return {
            "session_id": chat_session.id,
            "response": final_text,
            "tool_calls": executed_tool_calls,
            "trace": final_trace,
            "fast_path": False,
        }


# Singleton orchestrator instance
orchestrator = AgentOrchestrator()
