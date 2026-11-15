"""Hybrid Retrieval Architecture with Reciprocal Rank Fusion (RRF).

Combines:
1. Dense Semantic Vector Search (SentenceTransformers cosine similarity)
2. Sparse Lexical Search (Keyword term-frequency / BM25-style lexical scoring)
3. Reciprocal Rank Fusion (RRF) with constant k=60
"""

from collections import defaultdict
import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Stopwords for lexical term extraction
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves",
}


def tokenize_text(text: str) -> List[str]:
    """Tokenize query or text into clean lowercased alphanumeric stems/words."""
    words = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())
    return [w for w in words if w not in STOPWORDS]


def compute_bm25_lexical_score(query_tokens: List[str], doc_text: str, doc_title: str) -> float:
    """Compute lexical term matching score with title and breadcrumb boosting."""
    if not query_tokens:
        return 0.0

    text_tokens = tokenize_text(doc_text)
    title_tokens = tokenize_text(doc_title)
    if not text_tokens and not title_tokens:
        return 0.0

    score = 0.0
    text_len = len(text_tokens)

    for q in query_tokens:
        # Title match bonus (3x)
        title_count = title_tokens.count(q)
        # Content match
        body_count = text_tokens.count(q)

        tf = (title_count * 3.0) + body_count
        if tf > 0:
            # BM25 term saturation approximation
            k1 = 1.2
            b = 0.75
            norm_len = text_len / 150.0 if text_len > 0 else 1.0
            term_score = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * norm_len))
            score += term_score

    return round(score, 4)


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """Merge ranked lists using Reciprocal Rank Fusion (RRF).

    Formula: RRF(d) = sum( 1.0 / (k + rank) ) across dense and sparse runs.
    """
    rrf_scores: Dict[int, float] = defaultdict(float)
    doc_map: Dict[int, Dict[str, Any]] = {}
    dense_ranks: Dict[int, int] = {}
    sparse_ranks: Dict[int, int] = {}

    # 1. Score Dense Ranks
    for rank, item in enumerate(dense_results, start=1):
        doc_id = item["id"]
        doc_map[doc_id] = item
        dense_ranks[doc_id] = rank
        rrf_scores[doc_id] += 1.0 / (k + rank)

    # 2. Score Sparse Ranks
    for rank, item in enumerate(sparse_results, start=1):
        doc_id = item["id"]
        doc_map[doc_id] = item
        sparse_ranks[doc_id] = rank
        rrf_scores[doc_id] += 1.0 / (k + rank)

    # 3. Sort candidates by combined RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda did: rrf_scores[did], reverse=True)

    fused_results: List[Dict[str, Any]] = []
    for did in sorted_ids[:top_n]:
        item = dict(doc_map[did])
        item["rrf_score"] = round(rrf_scores[did], 6)
        item["dense_rank"] = dense_ranks.get(did, 999)
        item["sparse_rank"] = sparse_ranks.get(did, 999)
        fused_results.append(item)

    return fused_results

