# Agentic AI E-Commerce Support Resolution
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%7C%20pgvector-blue.svg)](https://github.com/pgvector/pgvector)
[![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-orange.svg)](https://modelcontextprotocol.io)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2.6+-red.svg)](https://docs.pydantic.dev/)

An enterprise-ready, stateful Agentic AI customer support resolution system for e-commerce. Built to solve multi-step operational inquiries across product catalog, shipping, orders, returns, and refunds with mathematical retrieval ranking, Model Context Protocol (MCP) tools, deterministic sub-80ms routing, and multi-tier security guardrails.

---

## 1. Project Overview

Customer support teams face high ticket volumes across operational inquiries (where is my order?, is this in stock?) and policy interpretations (can I return this opened camera?). Traditional chatbots either hallucinate or rely on rigid decision trees.

This system combines:
1. **Deterministic Fast-Path Routing (< 80ms, 0 tokens)** for routine order status and inventory checks.
2. **Autonomous ReAct Agent (Slow Path)** for complex, multi-step queries (e.g., verifying order details, consulting temporal policy rules, creating RMAs, and issuing refunds).
3. **Model Context Protocol (MCP) Tools** with strict dual-sided Pydantic v2 validation contracts and defense-in-depth authorization.
4. **Hybrid RAG (pgvector + BM25 + Reciprocal Rank Fusion + Cross-Encoder Reranking)** with strict purchase-date temporal policy boundaries.
5. **Multi-Vector Security Guardrails & Privacy**: Zero-latency regex + Meta Prompt Guard injection defense, strict 5-domain boundary filter, and automated PCI-DSS PII scrubbing.
6. **Resilience & Full Observability**: Bounded retries with exponential backoff and full jitter, automated support ticket generation (`TCK-XXXXX`) on persistent failures, offline RAG fallback, and asynchronous Langfuse + PostgreSQL execution tracing.

---

## 2. Technology Stack

| Category | Technology | Purpose & Justification |
| :--- | :--- | :--- |
| **Backend Framework** | **FastAPI 0.110+ & Uvicorn** | High-concurrency async ASGI web framework for REST API, WebSockets, and SPA serving |
| **Agent Core & LLMs** | **ReAct Loop + Groq API / Ollama** | Multi-step reasoning and function calling (`openai/gpt-oss-120b`, `meta-llama/llama-prompt-guard-2-86m`, or local `llama3.2`) |
| **Relational Database** | **PostgreSQL 16 + asyncpg** | High-scale operational database hosting 1.3M+ Kaggle e-commerce records with foreign keys and ACID transactions |
| **Vector Search Engine** | **pgvector (0.2.5+)** | In-database cosine similarity vector indexing (384 dimensions) for policy chunk retrieval |
| **Embeddings & Ranking**| **SentenceTransformers & Cross-Encoder** | `all-MiniLM-L6-v2` for dense embeddings, BM25 for lexical matching, and `ms-marco-MiniLM-L-6-v2` for cross-encoder reranking |
| **Tool Protocol** | **Model Context Protocol (MCP)** | Standardized tool interface with strict dual-sided Pydantic v2 input/output schema validation |
| **ORM & Data Layer** | **SQLAlchemy 2.0 (Async)** | Async ORM managing dimension tables, fact tables, chat sessions, messages, and execution traces |
| **Security & Guardrails** | **Pydantic v2, Jose (JWT), Bcrypt** | Multi-vector prompt injection defense, 5-domain business bounding, customer role scoping, and PCI-DSS PII redaction |
| **Observability** | **Langfuse & PostgreSQL Tracing** | Non-blocking telemetry capturing per-step execution traces, latency (ms), token consumption, and routing decisions |
| **Frontend UI** | **HTML5 / CSS3 / Vanilla JS SPA** | Responsive chat copilot interface with live trace inspector, order cards, and admin controls |

---

## 3. Quickstart & Setup Guide

### Prerequisites
- **Python 3.11+** installed
- **PostgreSQL 16+** with `pgvector` extension enabled (or standard PostgreSQL / SQLite fallback)
- **Groq Cloud API Key** (or local Ollama running)

### Step 1: Clone & Navigate to Project Directory
```bash
cd c:/Users/Unthinkable-lap/Desktop/ecommerce-support/ecom
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy `.env.example` to `.env` and configure your credentials:
```bash
cp .env.example .env
```
Key settings in `.env`:
```ini
# PostgreSQL Connection
DATABASE_URL=postgresql+asyncpg://postgres:your_password@localhost:5432/ecom

# JWT Authentication Secret
JWT_SECRET=your_super_secret_jwt_key_at_least_32_chars_long
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=480

# LLM Provider (Groq API or local Ollama)
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=your_groq_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Security Guardrails & Classifier
GUARDRAIL_ENABLED=true
GUARDRAIL_INTENT_MODEL=openai/gpt-oss-20b
GUARDRAIL_PROMPT_GUARD_MODEL=meta-llama/llama-prompt-guard-2-86m

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Optional Langfuse Observability
# LANGFUSE_PUBLIC_KEY=pk-lf-...
# LANGFUSE_SECRET_KEY=sk-lf-...
# LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

### Step 5: Database Seeding & Ingestion Pipeline
Seed the Kaggle operational dataset (1.3M+ records) and chunk/embed the policy knowledge base:
```bash
# Run database schema migration and seed
python -m app.db.seed --reset
```
*Note: Ingestion completes in ~180s using streaming PostgreSQL `COPY` commands and incremental hash-based policy caching.*

### Step 6: Start the Application Server
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Web Chat Application**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Healthcheck**: [http://localhost:8000/api/health](http://localhost:8000/api/health)

---


---

## 5. Verification & Testing

### Run Master Production Verification Suite (All 9 Suites)
Executes end-to-end tests validating the entire system without mock data:
```bash
python tests/test_suite.py
```

### Run Retrieval Benchmark (Golden Dataset)
Evaluates Dense Only vs. Hybrid RRF vs. Cross-Encoder Reranking across Hit@3 and MRR@3:
```bash
python benchmarks/evaluate.py
```

### Run Intent Safety & Guardrails Test Suite
Verifies prompt injection defense, baseless question refusal, and 5-domain boundary enforcement:
```bash
python tests/test_intent_safety_guardrails.py
```

---

## 6. Project Architecture Summary

```
ecom/
├── app/
│   ├── agent/               # ReAct agent loop, orchestrator, system prompts, tracing
│   ├── contracts/           # Dual-sided Pydantic v2 input/output tool schemas
│   ├── db/                  # SQLAlchemy models, asyncpg session manager, Kaggle seed pipeline
│   ├── observability/       # Langfuse async client and telemetry scrubber
│   ├── rag/                 # Ingest, chunkers (PDF/MD/JSON), hybrid RRF, reranker, retriever
│   ├── resilience/          # Bounded retries with jitter, support ticket fallback degradation
│   ├── router/              # Fast-Path deterministic router (<80ms) and tool registry
│   ├── security/            # Prompt injection filters, 5-domain classifier, PII redaction
│   ├── static/              # Frontend Single Page App (index.html, app.js, style.css)
│   ├── tools/               # 9 MCP business tools with customer/admin role authorization
│   ├── auth.py              # JWT authentication and customer registration endpoint
│   ├── config.py            # Pydantic BaseSettings environment configuration
│   └── main.py              # FastAPI application initialization and API routes
├── benchmarks/              # Retrieval evaluation harness and golden dataset
├── data/policies/           # Markdown, JSON FAQ, and PDF policy documents
└── tests/                   # 9 comprehensive master test suites and guardrail verifications
```

---
