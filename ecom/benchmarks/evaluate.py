"""Retrieval Evaluation and Benchmark Harness.

Compares:
1. Dense Vector Search Only
2. Sparse (BM25) Search Only
3. Hybrid Search with Reciprocal Rank Fusion (RRF k=60)
4. Hybrid Search + Cross-Encoder Reranking

Evaluates:
- Hit@3 (Recall at top-3)
- MRR@3 (Mean Reciprocal Rank at top-3)
- Average latency (ms) per query
"""

import asyncio
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

# Add ecom directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.session import AsyncSessionLocal
from app.rag.retriever import search_policies

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ecom.benchmark")

BENCHMARK_DIR = Path(__file__).resolve().parent
GOLDEN_DATASET_PATH = BENCHMARK_DIR / "golden_dataset.json"
RESULTS_DIR = BENCHMARK_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


async def evaluate_pipeline(
    dataset: List[Dict[str, Any]],
    pipeline_name: str,
    mode: str,
    use_hybrid: bool = True,
    use_reranker: bool = False,
) -> Dict[str, Any]:
    """Run evaluation for a specific retrieval pipeline configuration."""
    hits_at_3 = 0
    reciprocal_ranks = []
    latencies_ms = []

    async with AsyncSessionLocal() as session:
        for entry in dataset:
            query = entry["query"]
            expected_src = entry["expected_source"].lower()
            order_date = entry.get("order_date")

            t0 = time.perf_counter()
            results = await search_policies(
                session=session,
                query=query,
                top_k=3,
                order_date=order_date,
                use_hybrid=use_hybrid,
                use_reranker=use_reranker,
                mode=mode,
            )
            elapsed = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed)

            # Check rank of expected source
            hit_rank = 0
            for rank, r in enumerate(results, start=1):
                src = (r.get("source_file") or "").lower()
                if expected_src in src or expected_src.replace(".md", "") in src:
                    hit_rank = rank
                    break

            if hit_rank > 0:
                hits_at_3 += 1
                reciprocal_ranks.append(1.0 / hit_rank)
            else:
                reciprocal_ranks.append(0.0)

    total = len(dataset)
    hit_rate = round((hits_at_3 / total) * 100.0, 2) if total else 0.0
    mrr = round(sum(reciprocal_ranks) / total, 4) if total else 0.0
    avg_latency = round(sum(latencies_ms) / total, 2) if total else 0.0

    return {
        "pipeline": pipeline_name,
        "queries_count": total,
        "hit_at_3_pct": hit_rate,
        "mrr_at_3": mrr,
        "avg_latency_ms": avg_latency,
    }


async def run_benchmark():
    if not GOLDEN_DATASET_PATH.exists():
        logger.error("Golden dataset not found at: %s", GOLDEN_DATASET_PATH)
        return

    dataset = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    logger.info("Loaded %d golden evaluation queries.", len(dataset))

    configs = [
        ("Dense Vector Only", "dense", False, False),
        ("Sparse (BM25) Only", "sparse", False, False),
        ("Hybrid (Dense + Sparse RRF)", "hybrid", True, False),
        ("Hybrid + Cross-Encoder Reranking", "hybrid_rerank", True, True),
    ]

    all_results = []
    for name, mode, use_hybrid, use_reranker in configs:
        logger.info("Evaluating: %s...", name)
        res = await evaluate_pipeline(dataset, name, mode, use_hybrid, use_reranker)
        all_results.append(res)
        logger.info(
            "--> %s: Hit@3=%.1f%% | MRR@3=%.4f | Avg Latency=%.2fms",
            name,
            res["hit_at_3_pct"],
            res["mrr_at_3"],
            res["avg_latency_ms"],
        )

    out_file = RESULTS_DIR / "retrieval_benchmark.json"
    out_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    logger.info("Benchmark report written to: %s", out_file)
    return all_results


if __name__ == "__main__":
    asyncio.run(run_benchmark())
