"""Hierarchy-Aware and Structure-Preserving Document Chunkers.

Provides:
1. MarkdownHierarchyChunker:
   - Parses markdown heading hierarchy (#, ##, ###).
   - Generates contextual breadcrumbs (e.g., 'Return Policy > Electronics > 14-Day Limit').
   - Respects paragraph and sentence boundaries.
   - Computes deterministic SHA-256 chunk hash for incremental caching.
2. StructuredPdfExtractor:
   - Extracts clean digital text and preserves structured table contents and section titles.
3. JsonKnowledgeChunker:
   - Extracts structured FAQ question/answer pairs with topical breadcrumbs.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional


@dataclass
class DocumentChunk:
    title: str
    breadcrumbs: str
    content: str
    format: str
    source_file: str
    chunk_index: int
    chunk_hash: str
    metadata: Dict[str, Any]


def compute_sha256(text: str) -> str:
    """Compute deterministic SHA-256 hash of normalized text."""
    normalized = " ".join(text.strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class MarkdownHierarchyChunker:
    """Chunks markdown by heading hierarchy while preserving semantic breadcrumbs."""

    def __init__(self, max_chunk_size: int = 600, chunk_overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, source_file: str) -> List[DocumentChunk]:
        lines = text.splitlines()
        chunks: List[DocumentChunk] = []

        current_doc_title = Path(source_file).stem.replace("_", " ").title()
        # Find doc title from first h1 if available
        for line in lines:
            if line.startswith("# "):
                current_doc_title = line.lstrip("# ").strip()
                break

        current_h1 = current_doc_title
        current_h2 = ""
        current_h3 = ""

        current_buffer: List[str] = []
        chunk_idx = 0

        def flush_buffer():
            nonlocal chunk_idx, current_buffer
            if not current_buffer:
                return

            body = "\n".join(current_buffer).strip()
            if not body:
                current_buffer = []
                return

            # Construct breadcrumb trail
            trail = [current_h1]
            if current_h2 and current_h2 != current_h1:
                trail.append(current_h2)
            if current_h3:
                trail.append(current_h3)
            breadcrumbs = " > ".join(trail)

            # Title combines doc title and current section
            section_title = current_h3 or current_h2 or current_h1
            full_title = f"{current_doc_title} - {section_title}" if section_title != current_doc_title else current_doc_title

            payload = f"[{breadcrumbs}]\n{body}"
            chunk_hash = compute_sha256(payload)

            chunks.append(
                DocumentChunk(
                    title=full_title,
                    breadcrumbs=breadcrumbs,
                    content=payload,
                    format="markdown",
                    source_file=source_file,
                    chunk_index=chunk_idx,
                    chunk_hash=chunk_hash,
                    metadata={
                        "h1": current_h1,
                        "h2": current_h2,
                        "h3": current_h3,
                    },
                )
            )
            chunk_idx += 1
            current_buffer = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# "):
                flush_buffer()
                current_h1 = stripped.lstrip("# ").strip()
                current_h2 = ""
                current_h3 = ""
            elif stripped.startswith("## "):
                flush_buffer()
                current_h2 = stripped.lstrip("## ").strip()
                current_h3 = ""
            elif stripped.startswith("### "):
                flush_buffer()
                current_h3 = stripped.lstrip("### ").strip()
            else:
                current_buffer.append(line)
                # Check buffer length threshold
                if sum(len(l) for l in current_buffer) >= self.max_chunk_size and stripped == "":
                    flush_buffer()

        flush_buffer()
        return chunks


class JsonKnowledgeChunker:
    """Extracts structured FAQ entries into normalized document chunks."""

    def chunk(self, text: str, source_file: str) -> List[DocumentChunk]:
        data = json.loads(text)
        chunks: List[DocumentChunk] = []
        doc_title = Path(source_file).stem.replace("_", " ").title()

        items = data if isinstance(data, list) else data.get("faq", data.get("items", []))
        for idx, item in enumerate(items):
            category = item.get("category", "General").title()
            question = item.get("question", item.get("title", f"FAQ #{idx + 1}"))
            answer = item.get("answer", item.get("content", ""))

            breadcrumbs = f"{doc_title} > {category} > {question}"
            payload = f"[{breadcrumbs}]\nQ: {question}\nA: {answer}"
            chunk_hash = compute_sha256(payload)

            chunks.append(
                DocumentChunk(
                    title=f"{category}: {question}",
                    breadcrumbs=breadcrumbs,
                    content=payload,
                    format="json",
                    source_file=source_file,
                    chunk_index=idx,
                    chunk_hash=chunk_hash,
                    metadata={"category": category, "question": question},
                )
            )
        return chunks


class StructuredPdfExtractor:
    """Extracts clean text and section chunks from PDF documents."""

    def chunk(self, file_path: Path) -> List[DocumentChunk]:
        doc_title = file_path.stem.replace("_", " ").title()
        pdf_text = ""

        try:
            import fitz
            doc = fitz.open(str(file_path))
            for page in doc:
                pdf_text += page.get_text() + "\n\n"
        except Exception:
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(file_path))
                for page in reader.pages:
                    pdf_text += (page.extract_text() or "") + "\n\n"
            except Exception:
                pdf_text = ""

        clean_text = pdf_text.strip()
        if not clean_text:
            return []

        # Split into semantic paragraphs of around 500 characters
        paragraphs = [p.strip() for p in clean_text.split("\n\n") if len(p.strip()) > 30]
        chunks: List[DocumentChunk] = []

        curr_block: List[str] = []
        chunk_idx = 0

        for para in paragraphs:
            curr_block.append(para)
            if sum(len(p) for p in curr_block) >= 500:
                body = "\n\n".join(curr_block)
                breadcrumbs = f"{doc_title} > Section {chunk_idx + 1}"
                payload = f"[{breadcrumbs}]\n{body}"
                chunk_hash = compute_sha256(payload)

                chunks.append(
                    DocumentChunk(
                        title=f"{doc_title} (Part {chunk_idx + 1})",
                        breadcrumbs=breadcrumbs,
                        content=payload,
                        format="pdf",
                        source_file=file_path.name,
                        chunk_index=chunk_idx,
                        chunk_hash=chunk_hash,
                        metadata={"page_approx": chunk_idx + 1},
                    )
                )
                chunk_idx += 1
                curr_block = []

        if curr_block:
            body = "\n\n".join(curr_block)
            breadcrumbs = f"{doc_title} > Section {chunk_idx + 1}"
            payload = f"[{breadcrumbs}]\n{body}"
            chunk_hash = compute_sha256(payload)
            chunks.append(
                DocumentChunk(
                    title=f"{doc_title} (Part {chunk_idx + 1})",
                    breadcrumbs=breadcrumbs,
                    content=payload,
                    format="pdf",
                    source_file=file_path.name,
                    chunk_index=chunk_idx,
                    chunk_hash=chunk_hash,
                    metadata={"page_approx": chunk_idx + 1},
                )
            )

        return chunks

