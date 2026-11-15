"""Production-Grade Hybrid Policy Retriever with RRF and Cross-Encoder Reranking.

Implements:
1. Temporal policy filtering (effective_from <= order_date <= effective_to).
2. Dual-channel retrieval:
   - Dense channel: Cosine similarity against SentenceTransformers embeddings.
   - Sparse channel: Lexical term frequency / BM25-style keyword matching.
3. Reciprocal Rank Fusion (RRF with k=60) producing Top-N candidates (N=20).
4. Cross-Encoder / High-Precision Composite Reranking filtering candidates to final Top-K (K=3).
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Policy
from app.rag.ingest import get_embedding_model
from app.rag.hybrid import (
    tokenize_text,
    compute_bm25_lexical_score,
    reciprocal_rank_fusion,
)
from app.rag.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

_reranker: Optional[CrossEncoderReranker] = None


def get_reranker() -> CrossEncoderReranker:
    """Lazy initialization for CrossEncoder reranker to minimize import overhead."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def parse_order_date(order_date_input: Union[str, datetime]) -> Optional[datetime]:
    """Parse various date formats into a timezone-aware UTC datetime."""
    if isinstance(order_date_input, datetime):
        if order_date_input.tzinfo is None:
            return order_date_input.replace(tzinfo=timezone.utc)
        return order_date_input

    if not isinstance(order_date_input, str) or not order_date_input.strip():
        return None

    raw = order_date_input.strip()
    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(raw.replace("Z", ""), fmt.replace("Z", ""))
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    try:
        from dateutil import parser
        dt = parser.parse(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


async def search_policies(
    session: AsyncSession,
    query: str,
    top_k: int = 3,
    order_date: Optional[Union[str, datetime]] = None,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Execute policy retrieval with configurable pipelines: Dense, Sparse (BM25), Hybrid RRF, and Cross-Encoder."""
    if not query.strip():
        return []

    parsed_date = parse_order_date(order_date) if order_date else None

    # Temporal Filtering: effective_from <= order_date <= effective_to
    conditions = []
    if parsed_date:
        conditions.append(Policy.effective_from <= parsed_date)
        conditions.append(
            or_(
                Policy.effective_to.is_(None),
                Policy.effective_to >= parsed_date,
            )
        )

    # 1. Fetch eligible policy pool from database
    stmt = select(Policy)
    if conditions:
        stmt = stmt.where(and_(*conditions))

    result = await session.execute(stmt)
    policies = result.scalars().all()

    if not policies:
        logger.info("No policy documents match temporal criteria for date: %s", parsed_date)
        return []

    # Determine pipeline execution mode
    if mode == "sparse":
        run_dense = False
        run_sparse = True
        run_reranker = False
    elif mode == "dense":
        run_dense = True
        run_sparse = False
        run_reranker = False
    elif mode == "hybrid":
        run_dense = True
        run_sparse = True
        run_reranker = False
    elif mode == "hybrid_rerank":
        run_dense = True
        run_sparse = True
        run_reranker = True
    else:
        # Backward-compatible boolean flags
        run_dense = True
        run_sparse = use_hybrid
        run_reranker = use_hybrid and use_reranker

    # 2. Sparse Lexical Scoring (BM25 / Keyword Matching) if required
    sparse_candidates: List[Dict[str, Any]] = []
    if run_sparse:
        query_tokens = tokenize_text(query)
        for p in policies:
            lex_score = compute_bm25_lexical_score(query_tokens, p.content, p.title)
            if lex_score > 0.0 or not query_tokens:
                sparse_candidates.append({
                    "id": p.id,
                    "title": p.title,
                    "breadcrumbs": getattr(p, "breadcrumbs", "") or p.title,
                    "content": p.content,
                    "format": p.format,
                    "source_file": p.source_file,
                    "version": p.version,
                    "effective_from": p.effective_from.isoformat() if p.effective_from else None,
                    "effective_to": p.effective_to.isoformat() if p.effective_to else None,
                    "score": lex_score,
                    "similarity": lex_score,
                    "source": f"[Source: {p.source_file}]",
                })
        sparse_candidates.sort(key=lambda x: x["score"], reverse=True)

        # If Sparse-only mode, return top-k directly
        if not run_dense:
            return sparse_candidates[:top_k]

    # 3. Dense Vector Scoring if required
    dense_candidates: List[Dict[str, Any]] = []
    if run_dense:
        model = get_embedding_model()
        query_vector = model.encode(query, normalize_embeddings=True)
        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec) or 1.0

        for p in policies:
            if not p.embedding:
                continue
            p_vec = np.array(p.embedding, dtype=np.float32)
            p_norm = np.linalg.norm(p_vec) or 1.0
            cos_sim = float(np.dot(q_vec, p_vec) / (q_norm * p_norm))
            sim = max(0.0, min(1.0, cos_sim))

            dense_candidates.append({
                "id": p.id,
                "title": p.title,
                "breadcrumbs": getattr(p, "breadcrumbs", "") or p.title,
                "content": p.content,
                "format": p.format,
                "source_file": p.source_file,
                "version": p.version,
                "effective_from": p.effective_from.isoformat() if p.effective_from else None,
                "effective_to": p.effective_to.isoformat() if p.effective_to else None,
                "score": round(sim, 4),
                "similarity": round(sim, 4),
                "source": f"[Source: {p.source_file}]",
            })
        dense_candidates.sort(key=lambda x: x["score"], reverse=True)

        # If Dense-only mode, return top-k directly
        if not run_sparse:
            return dense_candidates[:top_k]

    # 4. Reciprocal Rank Fusion (RRF, k=60) -> Top-20 Candidates
    top_candidates = reciprocal_rank_fusion(
        dense_results=dense_candidates[:25],
        sparse_results=sparse_candidates[:25],
        k=60,
        top_n=20,
    )

    # 5. Cross-Encoder Reranking -> Top-K
    if run_reranker:
        final_results = get_reranker().rerank(query, top_candidates[:10], top_k=top_k)
    else:
        final_results = top_candidates[:top_k]

    return final_results


# Alias for compatibility
retrieve_policies = search_policies
