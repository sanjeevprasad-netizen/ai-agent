"""Two-Stage Retrieval Reranker.

Reranks candidate documents from hybrid search (Top-N, e.g. 20) to final top-K (e.g. 3-5).
Applies cross-attention scoring with graceful high-precision fallback.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_cross_encoder_model = None
_tried_loading_cross_encoder = False


def get_cross_encoder():
    """Lazy-load singleton cross-encoder model if available."""
    global _cross_encoder_model, _tried_loading_cross_encoder
    if not _tried_loading_cross_encoder:
        _tried_loading_cross_encoder = True
        try:
            from sentence_transformers import CrossEncoder
            # Lightweight cross-encoder
            _cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("Cross-encoder model loaded successfully.")
        except Exception as e:
            logger.info("CrossEncoder model unavailable (%s). Using composite ranking engine.", e)
            _cross_encoder_model = None
    return _cross_encoder_model


def composite_rerank_fallback(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """High-precision heuristic composite reranker when neural cross-encoder is offline."""
    query_lower = query.lower()
    query_words = set(re.findall(r"\b\w{3,}\b", query_lower))

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for c in candidates:
        content_lower = c.get("content", "").lower()
        title_lower = c.get("title", "").lower()

        # 1. Base RRF score normalized
        rrf = c.get("rrf_score", 0.0) * 100.0

        # 2. Exact phrase bonus
        exact_bonus = 5.0 if query_lower in content_lower or query_lower in title_lower else 0.0

        # 3. Keyword density
        matches = sum(1 for w in query_words if w in content_lower or w in title_lower)
        coverage = (matches / len(query_words)) * 10.0 if query_words else 0.0

        # 4. Dense similarity
        dense_sim = c.get("score", 0.0) * 10.0

        total_score = rrf + exact_bonus + coverage + dense_sim
        item = dict(c)
        item["rerank_score"] = round(total_score, 4)
        scored.append((total_score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored]


class CrossEncoderReranker:
    """Reranks candidate passages using neural cross-encoder or composite reranking."""

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        model = get_cross_encoder()
        if model is not None:
            try:
                pairs = [[query, c.get("content", "")] for c in candidates]
                scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
                ranked = []
                for score, item in zip(scores, candidates):
                    c_copy = dict(item)
                    c_copy["rerank_score"] = round(float(score), 4)
                    ranked.append(c_copy)
                ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
                return ranked[:top_k]
            except Exception as e:
                logger.warning("Neural cross-encoder inference failed: %s. Falling back to composite scoring.", e)

        # Fallback to composite reranking
        ranked = composite_rerank_fallback(query, candidates)
        return ranked[:top_k]

