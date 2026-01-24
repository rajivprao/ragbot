
# -*- coding: utf-8 -*-
"""
rag/chunker.py

Role-aware, metadata-preserving chunking utilities for EnterpriseRAG.

Goals
-----
- Normalize and split heterogeneous documents (PDF/DOCX/Web/ERD summaries)
- Use larger chunks for technical/schema/ERD materials to keep relationships together
- Filter out tiny/noisy chunks
- Preserve source metadata and add stable chunk identifiers (hash-based)
- Provide simple knobs for future tuning without touching the rest of the pipeline

Design
------
- Character-based recursive splitting via LangChain's RecursiveCharacterTextSplitter
  (robust, dependency-light; token splitters can be added later if needed)
- Two profiles by default:
    * Technical-like:   chunk_size=1200, overlap=120  (schema, technical, erd)
    * Standard/general: chunk_size=600,  overlap=60
- Chunk post-processing:
    * Whitespace normalization
    * Tiny-chunk filter
    * Optional short-chunk merging (disabled by default; enable if your corpus is very granular)

Usage
-----
    from rag.chunker import Chunker
    chunker = Chunker(min_text_len=20)
    chunks = chunker.split(doc_list)  # returns List[langchain_core.documents.Document]

Each output chunk inherits metadata and includes:
    - 'chunk_index' (1-based within its parent doc)
    - 'chunk_total' (count within its parent doc)
    - 'chunk_kind' (carried-through if set by extractors: "text" | "ocr" | "erd" | ...)
    - 'hash_id'     (stable 32-hex; sha256 of text + source + page_number)

If you later add token-aware splitting, keep the interface intact so the rest of the system remains unchanged.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple


# LangChain Document + text splitters
try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    class Document:  # minimal shim for static checks
        def __init__(self, page_content: str, metadata: dict):
            self.page_content = page_content
            self.metadata = metadata

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Shared helpers
try:
    from vector_store.base import normalize_text, sha256_id
except Exception:  # pragma: no cover
    import hashlib
    def normalize_text(txt: Optional[str]) -> str:
        return " ".join((txt or "").split())
    def sha256_id(text: str, source: str, page_number: int = 0) -> str:
        base = f"{normalize_text(text)}|{source}|{page_number}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()[:32]


class Chunker:
    """
    Role-aware chunker with per-doc_type profiles and metadata preservation.

    Parameters
    ----------
    std_chunk_size : int
        Chunk size for non-technical docs (functional/product/web/etc.).
    std_overlap : int
        Overlap for non-technical docs.
    tech_chunk_size : int
        Chunk size for technical-like docs (schema/technical/erd).
    tech_overlap : int
        Overlap for technical-like docs.
    min_text_len : int
        Minimal normalized length for a chunk to be retained.
    merge_short_tail : bool
        If True, tries to merge the last tiny chunk with the previous one (per source doc).
    technical_like_types : Tuple[str, ...]
        Doc types to be treated as "technical-like" for larger chunk sizes.
    """

    def __init__(
        self,
        std_chunk_size: int = 600,
        std_overlap: int = 60,
        tech_chunk_size: int = 1200,
        tech_overlap: int = 120,
        min_text_len: int = 20,
        merge_short_tail: bool = False,
        technical_like_types: Tuple[str, ...] = ("technical", "schema", "erd"),
    ) -> None:
        self.std_chunk_size = int(std_chunk_size)
        self.std_overlap = int(std_overlap)
        self.tech_chunk_size = int(tech_chunk_size)
        self.tech_overlap = int(tech_overlap)
        self.min_text_len = int(min_text_len)
        self.merge_short_tail = bool(merge_short_tail)
        self.technical_like_types = tuple(technical_like_types)

        # Prepare splitters
        self._std_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.std_chunk_size,
            chunk_overlap=self.std_overlap,
        )
        self._tech_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.tech_chunk_size,
            chunk_overlap=self.tech_overlap,
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def split(self, docs: List[Document]) -> List[Document]:
        """
        Split a list of Documents into retrieval-sized chunks.
        """
        if not docs:
            return []

        out_chunks: List[Document] = []
        for d in docs:
            out_chunks.extend(self._split_one(d))

        # Assign stable chunk ids (hash) and finalize metadata
        finalized: List[Document] = []
        for c in out_chunks:
            text = normalize_text(c.page_content)
            src = c.metadata.get("source") or c.metadata.get("file_name") or "unknown"
            page = int(c.metadata.get("page_number", 0) or 0)
            c.metadata["hash_id"] = sha256_id(text, str(src), page)
            finalized.append(c)

        return finalized

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _split_one(self, doc: Document) -> List[Document]:
        """
        Split a single input Document while preserving and enriching metadata.
        """
        text = normalize_text(getattr(doc, "page_content", "") or "")
        meta = dict(getattr(doc, "metadata", {}) or {})

        if len(text) < self.min_text_len:
            return []

        # Choose splitter by doc_type
        doc_type = (meta.get("doc_type") or "").lower()
        splitter = self._tech_splitter if doc_type in self.technical_like_types else self._std_splitter

        # Split
        pieces = splitter.split_text(text)

        # Merge tiny trailing chunk if enabled
        if self.merge_short_tail and len(pieces) >= 2:
            if len(normalize_text(pieces[-1])) < max(self.min_text_len, int(0.25 * self.std_chunk_size)):
                pieces[-2] = normalize_text(pieces[-2] + " " + pieces[-1])
                pieces.pop()

        # Build chunk docs
        chunks: List[Document] = []
        total = len(pieces)
        kind = meta.get("chunk_kind") or "text"  # carried-through from extractors when set

        for idx, piece in enumerate(pieces, start=1):
            piece_norm = normalize_text(piece)
            if len(piece_norm) < self.min_text_len:
                continue

            c_meta = {
                **meta,
                "chunk_index": idx,
                "chunk_total": total,
                "chunk_kind": kind,
            }
            chunks.append(Document(page_content=piece_norm, metadata=c_meta))

        return chunks


# -----------------------------------------------------------------------------
# Convenience function (optional) – one-off splitter
# -----------------------------------------------------------------------------

def split_documents(
    docs: List[Document],
    *,
    std_chunk_size: int = 600,
    std_overlap: int = 60,
    tech_chunk_size: int = 1200,
    tech_overlap: int = 120,
    min_text_len: int = 20,
    merge_short_tail: bool = False,
    technical_like_types: Tuple[str, ...] = ("technical", "schema", "erd"),
) -> List[Document]:
    """
    Functional wrapper around Chunker for quick use.

    Example:
        from rag.chunker import split_documents
        chunks = split_documents(docs, min_text_len=40)
    """
    return Chunker(
        std_chunk_size=std_chunk_size,
        std_overlap=std_overlap,
        tech_chunk_size=tech_chunk_size,
        tech_overlap=tech_overlap,
        min_text_len=min_text_len,
        merge_short_tail=merge_short_tail,
        technical_like_types=technical_like_types,
    ).split(docs)
