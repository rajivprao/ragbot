
# -*- coding: utf-8 -*-
"""
vector_store/base.py

A lightweight abstraction for pluggable vector databases used by EnterpriseRAG.

This module defines:
- StoredDoc: minimal container for text + metadata (and optional vector)
- SearchResult: normalized return type for vector searches
- VectorStoreBase: backend-agnostic interface every concrete store must implement
- Common helper utilities: normalize_text, now_iso, sha256_id, ensure_dir

Concrete backends (e.g., Qdrant) must subclass VectorStoreBase and implement the abstract methods.
"""

from __future__ import annotations

import abc
import datetime
import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


# ----------------------------
# Helpers (shared across backends)
# ----------------------------

def now_iso() -> str:
    """UTC timestamp in ISO format with 'Z' suffix (no microseconds)."""
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def normalize_text(txt: Optional[str]) -> str:
    """Whitespace-normalize text; return empty string for None."""
    if not txt:
        return ""
    # Collapse all whitespace to single spaces and strip ends
    return " ".join(txt.split())


def sha256_id(text: str, source: str, page_number: int = 0) -> str:
    """
    Stable 32-hex identifier from normalized text + source + page number.
    Makes upserts idempotent across runs and supports deduplication.
    """
    base = f"{normalize_text(text)}|{source}|{page_number}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:32]


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist (idempotent)."""
    os.makedirs(path, exist_ok=True)


# ----------------------------
# Data contracts
# ----------------------------

Metadata = Dict[str, Any]
FilterDict = Dict[str, Any]  # simple equality-based filters; richer backends can accept nested expressions


@dataclass
class StoredDoc:
    """
    Canonical unit we store in a vector DB.
    - id:     stable identifier (e.g., sha256 hash of text+source+page)
    - text:   the text that is embedded and searched
    - metadata: arbitrary payload stored alongside the vector (for filters / citations)
    - vector: optional precomputed embedding (some backends accept direct vectors)
    """
    id: str
    text: str
    metadata: Metadata
    vector: Optional[Sequence[float]] = None


@dataclass
class SearchResult:
    """
    Normalized search hit returned by vector stores.
    - id, text, metadata: the original payload
    - score: similarity score (higher is better). Scale is backend-dependent but normalized to float.
    """
    id: str
    text: str
    metadata: Metadata
    score: float


# ----------------------------
# Vector store interface
# ----------------------------

class VectorStoreError(RuntimeError):
    """Base exception for vector store errors."""


class VectorStoreBase(abc.ABC):
    """
    Abstract interface for a vector database backend.

    Backends should:
      - Perform embeddings internally (using a provided embeddings object), OR
      - Accept precomputed vectors via StoredDoc.vector.

    All methods should be side-effect free (idempotent) where practical.
    """

    def __init__(
        self,
        collection_alias: str,
        embeddings: Any = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            collection_alias: Stable alias/name used by the application (e.g., "enterprise_corpus").
                              Backends can map this alias to a concrete collection for blue/green swaps.
            embeddings:       Optional embeddings object with .embed_query(text: str) -> List[float]
                              and .embed_documents(texts: List[str]) -> List[List[float]]
            **kwargs:         Backend-specific parameters (paths, dimensions, distance metric, etc.)
        """
        self.collection_alias = collection_alias
        self.embeddings = embeddings
        self.backend_kwargs = kwargs

    # ---- Lifecycle / collections ----

    @abc.abstractmethod
    def ensure_collection(self, collection: Optional[str] = None, **kwargs: Any) -> None:
        """
        Ensure that the (aliased) collection (or a concrete collection name) exists with correct schema.
        Called at init or ingest start.
        """

    def load(self) -> None:
        """
        Optional hook for restoring a previously created index.
        For remote DBs, this can be a no-op; for embedded backends, re-open handles if needed.
        """
        # Default no-op
        return

    def save(self) -> None:
        """
        Optional hook for persisting any in-memory changes (if the backend requires it).
        Remote DBs often persist automatically.
        """
        # Default no-op
        return

    def close(self) -> None:
        """
        Optional hook to gracefully close connections / release resources.
        """
        # Default no-op
        return

    # ---- Blue/Green alias management (optional) ----

    def promote_collection(self, new_collection_name: str) -> None:
        """
        Atomically (or near-atomically) move the alias to `new_collection_name`.
        Backends that don't support aliases can override with a best-effort swap,
        or raise NotImplementedError if not supported.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support alias promotion.")

    # ---- Write paths ----

    @abc.abstractmethod
    def upsert(self, docs: Iterable[StoredDoc], collection: Optional[str] = None) -> None:
        """
        Insert or update a batch of StoredDoc objects into the vector store.
        Implementations should:
          - Embed `doc.text` if `doc.vector` is None and embeddings are available.
          - Store `doc.metadata` (payload) for filtering and citations.
          - Use `doc.id` as the primary key for idempotent upserts.
        """

    def delete_by_ids(self, ids: Sequence[Union[str, int]], collection: Optional[str] = None) -> int:
        """
        Delete points by id. Return the count deleted (if known; else 0 or -1).
        Override if the backend supports it; default raises NotImplementedError.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support delete_by_ids.")

    def delete_by_filter(self, filters: FilterDict, collection: Optional[str] = None) -> int:
        """
        Delete points that match filters. Return the count deleted (if known; else 0 or -1).
        Override if the backend supports it; default raises NotImplementedError.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support delete_by_filter.")

    # ---- Read paths ----

    @abc.abstractmethod
    def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[FilterDict] = None,
        collection: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Vector similarity search.
        Args:
            query:    user query (will be embedded inside the store) or a special syntax if supported.
            k:        number of hits to return.
            filters:  payload filters (equality-based dict). Backends can support richer expressions.
            collection: override the target collection (defaults to the active alias).
        Returns:
            List[SearchResult]: top-k results with scores.
        """

    def msearch(
        self,
        queries: Sequence[str],
        k: int = 10,
        filters: Optional[FilterDict] = None,
        collection: Optional[str] = None,
    ) -> List[List[SearchResult]]:
        """
        Optional batched search for efficiency. Default is a naive loop over `search`.
        Override in backends that support batch query execution.
        """
        return [self.search(q, k=k, filters=filters, collection=collection) for q in queries]

    def count(self, filters: Optional[FilterDict] = None, collection: Optional[str] = None) -> int:
        """
        Return estimated number of points matching filters (or total count if filters is None).
        If not supported, return 0 or raise NotImplementedError.
        """
        return 0  # default conservative implementation


# ----------------------------
# Convenience (optional) – conversion helpers
# ----------------------------

def docs_to_storeddocs(
    docs: Iterable[Any],
    *,
    default_source_key: str = "file_name",
    default_page_key: str = "page_number",
    default_id_from: Tuple[str, str] = ("source", "page_number"),
) -> List[StoredDoc]:
    """
    Convert generic 'Document-like' objects to StoredDoc.
    The function assumes each `doc` has:
        - `page_content: str`
        - `metadata: dict` with at least 'source' or 'file_name' and optional 'page_number'
    This is intentionally loose so it can work with or without LangChain's Document class.

    Args:
        docs: iterable of objects with .page_content and .metadata
        default_source_key: fallback metadata key for source if 'source' missing
        default_page_key:   fallback metadata key for page if 'page_number' missing
        default_id_from:    tuple of (source_key, page_key) used to build a stable hash id
    """
    out: List[StoredDoc] = []
    for d in docs:
        text = normalize_text(getattr(d, "page_content", "") or "")
        meta = dict(getattr(d, "metadata", {}) or {})

        source = meta.get("source") or meta.get(default_source_key) or "unknown"
        page = int(meta.get("page_number", meta.get(default_page_key, 0)) or 0)

        # Compute stable id if not present
        _id = meta.get("hash_id") or sha256_id(text, str(source), page)

        out.append(StoredDoc(id=_id, text=text, metadata=meta))
    return out


def rrf_fuse(
    result_lists: Sequence[Sequence[SearchResult]],
    k: int = 10,
    k_rrf: int = 60,
) -> List[SearchResult]:
    """
    Reciprocal Rank Fusion (RRF) of multiple ranked lists.
    Args:
        result_lists: list of ranked lists (e.g., [vector_hits, bm25_hits_as_SearchResult])
        k:            number of fused results to return
        k_rrf:        RRF constant; higher reduces influence of tail ranks
    Returns:
        fused, deduplicated list of SearchResult (scores combined).
    """
    scores: Dict[str, float] = {}
    payload: Dict[str, Tuple[str, Metadata]] = {}  # id -> (text, metadata)

    for lst in result_lists:
        for rank, r in enumerate(lst, start=1):
            scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (k_rrf + rank)
            # keep first-seen payload
            if r.id not in payload:
                payload[r.id] = (r.text, r.metadata)

    fused = [
        SearchResult(id=_id, text=payload[_id][0], metadata=payload[_id][1], score=score)
        for _id, score in scores.items()
    ]
    fused.sort(key=lambda x: x.score, reverse=True)
    return fused[:k]
