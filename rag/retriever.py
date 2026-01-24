
# -*- coding: utf-8 -*-
"""
rag/retriever.py

Hybrid retrieval for EnterpriseRAG (Qdrant vector DB + BM25 sparse),
with role-aware reranking, deduplication, and technical/functional interleaving.

Key features
------------
- Uses your pluggable VectorStore interface (e.g., QdrantStore) for vector search
- Uses LangChain BM25 for sparse search over an in-memory corpus
- Reciprocal Rank Fusion (RRF) to combine vector + BM25 results
- Optional metadata filters (applied to both sides)
- Role-aware soft rerank (developer/ba/new_joiner)
- Deduplication across retrievers
- Interleave technical and functional chunks to improve grounding

Typical usage
-------------
    from rag.retriever import HybridRetriever
    from vector_store.qdrant_store import QdrantStore
    from rag.chunker import Chunker

    # 1) Build / attach vector store (Qdrant embedded)
    vstore = QdrantStore(collection_alias="enterprise_corpus", embeddings=emb, path="qdrant_data", vector_size=384)

    # 2) Keep a rolling BM25 over all chunks you've ingested
    retr = HybridRetriever(vstore=vstore)

    # Whenever you add new chunks:
    retr.refresh_sparse(new_chunks)  # BM25 incremental (we append into the internal list)

    # 3) Query
    docs = retr.search(
        "What are the join keys for Calendar Master?",
        role="developer",
        k=10,
        filters={"doc_type":"erd"}  # optional equality filter
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# LangChain
try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    class Document:  # minimal shim
        def __init__(self, page_content: str, metadata: dict):
            self.page_content = page_content
            self.metadata = metadata

from langchain_community.retrievers import BM25Retriever

# Vector store contracts + helpers
from vector_store.base import (
    SearchResult,
    VectorStoreBase,
    rrf_fuse,
    normalize_text,
    sha256_id,
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _doc_id_for_rrf(doc: Document) -> str:
    """
    Stable ID for a Document to be used in fusion/dedup.
    Prefer 'hash_id' if present; else compute from text+source+page.
    """
    meta = doc.metadata or {}
    if "hash_id" in meta and meta["hash_id"]:
        return str(meta["hash_id"])
    source = meta.get("source") or meta.get("file_name") or "unknown"
    page = int(meta.get("page_number", 0) or 0)
    return sha256_id(doc.page_content, str(source), page)


def _doc_to_search_result(doc: Document, pseudo_score: float) -> SearchResult:
    """
    Wrap a BM25 Document as a SearchResult with a pseudo score (e.g., 1/rank).
    """
    return SearchResult(
        id=_doc_id_for_rrf(doc),
        text=doc.page_content,
        metadata=dict(doc.metadata or {}),
        score=float(pseudo_score),
    )


def _result_to_document(res: SearchResult) -> Document:
    """
    Convert a vector SearchResult back to a LangChain Document.
    Text and metadata are carried over.
    """
    return Document(page_content=res.text, metadata=dict(res.metadata or {}))


def _apply_metadata_filters(docs: Iterable[Document], filters: Optional[Dict[str, object]]) -> List[Document]:
    """
    Equality-only filtering on Document.metadata.
    """
    if not filters:
        return list(docs)

    def ok(md: dict) -> bool:
        return all(md.get(k) == v for k, v in filters.items())

    return [d for d in docs if ok(d.metadata or {})]


def _deduplicate_docs(docs: List[Document]) -> List[Document]:
    """
    Deduplicate final Documents by (source, head of content).
    (This complements RRF's id-level dedupe.)
    """
    seen = set()
    out: List[Document] = []
    for d in docs:
        key = ( (d.metadata or {}).get("source"), (d.page_content or "")[:80] )
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _role_soft_rerank(docs: List[Document], role: str, role_doc_type_priority: Dict[str, List[str]], max_docs: int) -> List[Document]:
    """
    Soft re-ranking by role-preferred doc_type, then truncate.
    """
    preferred = set((role_doc_type_priority.get(role.lower()) or []))
    def score(d: Document) -> int:
        return 1 if (d.metadata or {}).get("doc_type") in preferred else 0
    ranked = sorted(docs, key=score, reverse=True)
    return ranked[:max_docs]


def _interleave_tech_functional(docs: List[Document]) -> List[Document]:
    """
    Interleave technical-like and functional-like documents to give the LLM both perspectives.
    """
    technical_like = {"technical", "schema", "erd", "api", "reference"}
    functional_like = {"functional", "product", "analytics", "reporting", "user_guide"}

    tech = [d for d in docs if (d.metadata or {}).get("doc_type") in technical_like]
    func = [d for d in docs if (d.metadata or {}).get("doc_type") in functional_like]

    if not tech or not func:
        return docs

    out: List[Document] = []
    for i in range(max(len(tech), len(func))):
        if i < len(func): out.append(func[i])
        if i < len(tech): out.append(tech[i])
    # Keep any remaining that didn't fit the first max loop (unlikely here)
    if len(out) < len(docs):
        out.extend(docs[len(out):])
    return out


# -----------------------------------------------------------------------------
# Hybrid Retriever
# -----------------------------------------------------------------------------

@dataclass
class HybridRetrieverConfig:
    """
    Configuration knobs for hybrid retrieval.
    """
    fetch_k_vector: int = 20    # over-fetch for fusion
    fetch_k_sparse: int = 20
    return_k: int = 10          # final k after postprocessing
    rrf_k: int = 60             # RRF constant
    enable_interleave: bool = True


class HybridRetriever:
    """
    Hybrid retriever combining a real vector DB (Qdrant via VectorStoreBase) and BM25.

    Responsibilities:
    - Maintain an in-memory BM25 over "all chunks seen so far"
    - Call the vector store for semantic results (with optional filters)
    - Fuse (RRF), then apply dedupe, role-aware soft rerank, and interleave
    """

    def __init__(
        self,
        vstore: VectorStoreBase,
        *,
        role_doc_type_priority: Optional[Dict[str, List[str]]] = None,
        config: Optional[HybridRetrieverConfig] = None,
    ) -> None:
        self.vstore = vstore
        self.role_doc_type_priority = role_doc_type_priority or {
            "developer": ["technical", "schema", "erd", "reference", "api"],
            "ba": ["functional", "product", "analytics", "reporting"],
            "new_joiner": ["product", "onboarding", "user_guide", "how_to"],
        }
        self.config = config or HybridRetrieverConfig()

        # Sparse store components
        self._sparse_docs: List[Document] = []
        self._bm25: Optional[BM25Retriever] = None

    # -------------------------------------------------------------------------
    # Sparse maintenance (call this whenever you add new chunks)
    # -------------------------------------------------------------------------

    def refresh_sparse(self, new_docs: List[Document]) -> None:
        """
        Add new documents to the in-memory BM25 corpus and rebuild the retriever.
        This can be called incrementally after each ingestion batch.
        """
        if not new_docs:
            return
        self._sparse_docs.extend(new_docs)
        self._bm25 = BM25Retriever.from_documents(self._sparse_docs)
        # we intentionally do not set .k here; we'll ask for top-k in the method using fetch_k_sparse

    # -------------------------------------------------------------------------
    # Core search API
    # -------------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        role: str = "developer",
        k: Optional[int] = None,
        filters: Optional[Dict[str, object]] = None,
    ) -> List[Document]:
        """
        Perform a hybrid search (vector + BM25) and return post-processed Documents.

        Args:
            query:   user query string
            role:    "developer" | "ba" | "new_joiner" (affects soft rerank)
            k:       final number of docs to return (defaults to config.return_k)
            filters: equality filters applied on metadata for both vector and sparse sides
                     (for BM25 we post-filter after retrieval)
        """
        cfg = self.config
        ret_k = int(k or cfg.return_k)
        fetch_vec = max(ret_k, cfg.fetch_k_vector)
        fetch_spa = max(ret_k, cfg.fetch_k_sparse)

        # --- Vector side (native filters supported) ---
        vec_hits: List[SearchResult] = self.vstore.search(query, k=fetch_vec, filters=filters)

        # --- Sparse side (BM25) ---
        bm_docs: List[Document] = []
        if self._bm25 is not None:
            # langchain BM25 doesn't accept filters at query time; we filter after retrieval
            bm_docs = self._bm25.get_relevant_documents(query)
            # keep only top fetch_spa
            bm_docs = bm_docs[:fetch_spa]
            # apply metadata filters
            bm_docs = _apply_metadata_filters(bm_docs, filters)

        # Convert BM25 docs to SearchResult with pseudo scores based on rank (1/(rank+1))
        bm_hits: List[SearchResult] = [
            _doc_to_search_result(d, pseudo_score=1.0 / (i + 1)) for i, d in enumerate(bm_docs)
        ]

        # --- Fuse with RRF ---
        fused: List[SearchResult] = rrf_fuse([vec_hits, bm_hits], k=max(ret_k * 3, 20), k_rrf=cfg.rrf_k)

        # --- Convert back to Document objects ---
        fused_docs: List[Document] = [_result_to_document(r) for r in fused]

        # --- Postprocessing: dedupe, role-aware rerank, interleave ---
        fused_docs = _deduplicate_docs(fused_docs)
        fused_docs = _role_soft_rerank(fused_docs, role=role, role_doc_type_priority=self.role_doc_type_priority, max_docs=max(ret_k * 2, 20))
        if self.config.enable_interleave:
            fused_docs = _interleave_tech_functional(fused_docs)

        # Final truncate
        return fused_docs[:ret_k]
