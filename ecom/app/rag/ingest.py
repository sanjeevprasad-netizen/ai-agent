"""Production-Grade Incremental Policy Ingestion Pipeline with Content Hashing.

Features:
1. Deterministic SHA-256 content hashing of raw source documents.
2. Ingestion metadata tracking in `ingestion_metadata` table.
3. Incremental change detection:
   - Unchanged document: Skip parsing, chunking, and embedding entirely.
   - Changed/New document: Parse into hierarchy-aware chunks with breadcrumbs, embed, and upsert.
   - Deleted document: Remove obsolete policy chunks from DB.
4. Preserves vector storage and embedding model version compatibility.
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.config import settings, POLICIES_DIR
from app.db.models import Policy, IngestionMetadata
from app.rag.chunkers import (
    MarkdownHierarchyChunker,
    JsonKnowledgeChunker,
    StructuredPdfExtractor,
    DocumentChunk,
    compute_sha256,
)

logger = logging.getLogger(__name__)

_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Get or initialize singleton SentenceTransformer model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model '%s'...", settings.EMBEDDING_MODEL)
        try:
            _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL, local_files_only=True)
        except Exception:
            _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully.")
    return _embedding_model


def calculate_file_sha256(file_path: Path) -> str:
    """Compute deterministic SHA-256 hash of a file's raw bytes."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


async def ingest_policies(
    session: AsyncSession,
    policies_dir: Optional[Path] = None,
    force_reembed: bool = False,
) -> int:
    """Incrementally discover, hash, chunk, embed, and store policies."""
    target_dir = policies_dir or POLICIES_DIR
    if not target_dir.exists():
        logger.warning("Policies directory %s does not exist. Skipping ingestion.", target_dir)
        return 0

    model = get_embedding_model()
    md_chunker = MarkdownHierarchyChunker(max_chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    json_chunker = JsonKnowledgeChunker()
    pdf_extractor = StructuredPdfExtractor()

    base_effective_date = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    now_utc = datetime.now(timezone.utc)

    # 1. Fetch current ingestion metadata
    meta_res = await session.execute(select(IngestionMetadata))
    existing_meta: Dict[str, IngestionMetadata] = {
        m.document_id: m for m in meta_res.scalars().all()
    }

    # Discover policy files (.md, .json, .pdf)
    discovered_files: List[Path] = []
    for ext in ("*.md", "*.json", "*.pdf"):
        discovered_files.extend(sorted(target_dir.glob(ext)))

    discovered_doc_ids: Set[str] = {f.name for f in discovered_files}

    # 2. Detect deleted source documents and prune obsolete chunks
    for doc_id, meta in list(existing_meta.items()):
        if doc_id not in discovered_doc_ids:
            logger.info("Source document '%s' was deleted. Removing obsolete chunks...", doc_id)
            await session.execute(delete(Policy).where(Policy.source_file == doc_id))
            await session.execute(delete(IngestionMetadata).where(IngestionMetadata.document_id == doc_id))

    total_chunks_processed = 0

    # 3. Process each discovered file
    for file_path in discovered_files:
        doc_id = file_path.name
        current_hash = calculate_file_sha256(file_path)
        stored_meta = existing_meta.get(doc_id)

        # Change Detection: Check if document is unchanged and model version matches
        if (
            not force_reembed
            and stored_meta is not None
            and stored_meta.document_hash == current_hash
            and stored_meta.embedding_model == settings.EMBEDDING_MODEL
        ):
            # Check if chunks actually exist in DB
            chunk_check = await session.execute(
                select(Policy.id).where(Policy.source_file == doc_id).limit(1)
            )
            if chunk_check.scalar_one_or_none() is not None:
                logger.info("Document '%s' is unchanged (SHA-256 match). Skipping embedding.", doc_id)
                continue

        logger.info("Ingesting new/modified document: %s (SHA-256: %s...)", doc_id, current_hash[:12])

        # Parse into chunks
        chunks: List[DocumentChunk] = []
        if file_path.suffix.lower() == ".md":
            text = file_path.read_text(encoding="utf-8")
            chunks = md_chunker.chunk(text, source_file=doc_id)
        elif file_path.suffix.lower() == ".json":
            text = file_path.read_text(encoding="utf-8")
            chunks = json_chunker.chunk(text, source_file=doc_id)
        elif file_path.suffix.lower() == ".pdf":
            chunks = pdf_extractor.chunk(file_path)

        if not chunks:
            logger.warning("No chunks extracted from document: %s", doc_id)
            continue

        # Embed all extracted chunks
        chunk_texts = [c.content for c in chunks]
        embeddings = model.encode(chunk_texts, normalize_embeddings=True)

        # Clear old chunks for this document
        await session.execute(delete(Policy).where(Policy.source_file == doc_id))

        # Insert fresh chunks with breadcrumbs and chunk hashes
        records: List[Policy] = []
        for chunk_obj, emb in zip(chunks, embeddings):
            rec = Policy(
                title=chunk_obj.title,
                content=chunk_obj.content,
                format=chunk_obj.format,
                source_file=doc_id,
                version="2.0",
                effective_from=base_effective_date,
                effective_to=None,
                embedding=emb.tolist(),
                document_hash=current_hash,
                chunk_hash=chunk_obj.chunk_hash,
                breadcrumbs=chunk_obj.breadcrumbs,
            )
            records.append(rec)

        session.add_all(records)
        total_chunks_processed += len(records)

        # Update IngestionMetadata
        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
        if stored_meta:
            stored_meta.document_hash = current_hash
            stored_meta.embedding_model = settings.EMBEDDING_MODEL
            stored_meta.ingested_at = now_utc
            stored_meta.last_modified_at = file_mtime
            stored_meta.ingestion_status = "success"
        else:
            new_meta = IngestionMetadata(
                document_id=doc_id,
                source_name=file_path.stem.replace("_", " ").title(),
                document_hash=current_hash,
                content_version="2.0",
                embedding_model=settings.EMBEDDING_MODEL,
                embedding_model_version="1.0",
                ingested_at=now_utc,
                last_modified_at=file_mtime,
                ingestion_status="success",
            )
            session.add(new_meta)

    await session.commit()

    # Query total active policy chunks
    count_res = await session.execute(select(Policy.id))
    total_active_chunks = len(count_res.scalars().all())
    logger.info("Policy ingestion complete: %d chunks active in database.", total_active_chunks)
    return total_active_chunks
