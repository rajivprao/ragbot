
# -*- coding: utf-8 -*-
"""
rag/enterprise_rag.py

EnterpriseRAG orchestrator:
- Vector DB: Qdrant (embedded) via vector_store.qdrant_store.QdrantStore
- Extractors: PDF (text + OCR + ERD vector parsing), DOCX, Web
- Chunking: role-aware chunker
- Retrieval: Hybrid (Qdrant vector + BM25) with role-aware rerank/interleave
- LLM: OpenRouter-compatible client
- Incremental upserts + blue/green rebuild (Qdrant alias promotion)

Typical usage:
    from rag.enterprise_rag import EnterpriseRAG
    rag = EnterpriseRAG()
    rag.ingest_path("DATA/DOCS", corpus_id="enterprise_docs")
    rag.add_web_url("https://example.com/docs/calendar", corpus_id="web_docs")
    ans = rag.ask("What are the join keys for Calendar Master?", role="developer")
"""

from __future__ import annotations

import os
import time
from typing import Dict, Iterable, List, Optional


# LangChain Document
try:
    from langchain_core.documents import Document
except Exception:
    class Document:  # minimal shim
        def __init__(self, page_content: str, metadata: dict):
            self.page_content = page_content
            self.metadata = metadata

# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vector store (Qdrant embedded)
from vector_store.base import (
    StoredDoc,
    VectorStoreBase,
    docs_to_storeddocs,
    normalize_text,
    now_iso,
)
from vector_store.qdrant_store import QdrantStore

# Extractors
from extractors.pdf_extractor import PdfExtractor
from extractors.docx_extractor import DocxExtractor
from extractors.web_extractor import WebExtractor

# Chunker
from rag.chunker import Chunker

# Retriever
from rag.retriever import HybridRetriever, HybridRetrieverConfig

# Prompts + LLM
from rag.prompts import build_prompts
from rag.llm_client import LLMClient


class EnterpriseRAG:
    """
    High-level facade that wires together:
    - Qdrant vector DB
    - BM25 hybrid retrieval
    - OCR + vector parsing (PDFs)
    - DOCX/Web ingestion
    - OpenRouter LLM calls
    """

    def __init__(
        self,
        *,
        # --- Vector DB (Qdrant embedded) ---
        collection_alias: str = "enterprise_corpus",
        qdrant_path: str = "qdrant_data",

        # --- Embeddings ---
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        distance: str = "cosine",

        # --- Chunking ---
        std_chunk_size: int = 600,
        std_overlap: int = 60,
        tech_chunk_size: int = 1200,
        tech_overlap: int = 120,
        min_text_len: int = 20,

        # --- Retrieval ---
        fetch_k_vector: int = 20,
        fetch_k_sparse: int = 20,
        return_k: int = 10,
        rrf_k: int = 60,
        enable_interleave: bool = True,

        # --- LLM ---
        openrouter_api_key: Optional[str] = None,
        openrouter_model: str = "mistralai/devstral-2512:free",
        openrouter_base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        app_name: Optional[str] = "EnterpriseRAG",

        # --- Extractors toggles ---
        enable_ocr: bool = True,
        ocr_dpi: int = 200,
        enable_vector_parse: bool = True,

        # --- Web ---
        user_agent: str = "EnterpriseRAGBot/1.0",
        web_timeout: int = 30,
        obey_robots: bool = True,
        sleep_seconds: float = 0.5,
        use_trafilatura: bool = True,
        web_min_text_len: int = 100,

    ) -> None:
        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # Vector DB
        self.vstore: VectorStoreBase = QdrantStore(
            collection_alias=collection_alias,
            embeddings=self.embeddings,
            path=qdrant_path,
            vector_size=embedding_dim,
            distance=distance,
        )

        # Chunker
        self.chunker = Chunker(
            std_chunk_size=std_chunk_size,
            std_overlap=std_overlap,
            tech_chunk_size=tech_chunk_size,
            tech_overlap=tech_overlap,
            min_text_len=min_text_len,
        )

        # Retriever (hybrid)
        self.retriever = HybridRetriever(
            vstore=self.vstore,
            config=HybridRetrieverConfig(
                fetch_k_vector=fetch_k_vector,
                fetch_k_sparse=fetch_k_sparse,
                return_k=return_k,
                rrf_k=rrf_k,
                enable_interleave=enable_interleave,
            ),
        )

        # Extractors
        self.pdfx = PdfExtractor(
            enable_ocr=enable_ocr,
            ocr_dpi=ocr_dpi,
            enable_vector_parse=enable_vector_parse,
            min_text_len=min_text_len,
        )
        self.docxx = DocxExtractor(
            min_text_len=min_text_len,
            doc_type_infer=True,
            join_tables=True,
            per_file_document=True,
        )
        self.webx = WebExtractor(
            user_agent=user_agent,
            timeout=web_timeout,
            obey_robots=obey_robots,
            sleep_seconds=sleep_seconds,
            use_trafilatura=use_trafilatura,
            min_text_len=web_min_text_len,
            doc_type_infer=True,
        )

        # LLM client (OpenRouter)
        if openrouter_api_key is None:
            # Allow late binding via set_api_key(), but warn
            print("[INFO] No OpenRouter API key passed; set via set_api_key() before ask().")
        self.llm = LLMClient(
            api_key=openrouter_api_key or "",
            model=openrouter_model,
            base_url=openrouter_base_url,
            app_name=app_name,
        )

        self.gemini_llm = LLMClient(api_key=openrouter_api_key)

    # -------------------------------------------------------------------------
    # Configuration helpers
    # -------------------------------------------------------------------------

    def set_api_key(self, api_key: str) -> None:
        """Set/replace OpenRouter API key at runtime."""
        self.llm.api_key = api_key

    # -------------------------------------------------------------------------
    # Ingestion (incremental upsert)
    # -------------------------------------------------------------------------

    def ingest_path(self, path: str, *, corpus_id: str = "default", recursive: bool = True, max_pdf_pages: Optional[int] = None) -> int:
        """
        Ingest PDFs and DOCX under a path (recursively by default).
        Returns number of chunks upserted to the vector DB.
        """
        docs: List[Document] = []
        # PDFs
        docs.extend(self.pdfx.load_path(path, recursive=recursive, max_pages_per_file=max_pdf_pages))
        # DOCX
        docs.extend(self.docxx.load_path(path, recursive=recursive))

        if not docs:
            print(f"[INFO] No PDFs/DOCX found under: {path}")
            return 0

        # Chunk
        chunks = self.chunker.split(docs)
        if not chunks:
            print("[INFO] No chunks produced after chunking.")
            return 0

        # Upsert to vector DB
        upserted = self._upsert_chunks(chunks, corpus_id=corpus_id)
        # Refresh BM25
        self.retriever.refresh_sparse(chunks)
        print(f"[OK] Ingested path={path} chunks={upserted}")
        return upserted

    def add_web_url(self, url: str, *, corpus_id: str = "web") -> int:
        docs = self.webx.load_url(url)
        return self._ingest_docs(docs, corpus_id)

    def add_web_urls(self, urls: Iterable[str], *, corpus_id: str = "web") -> int:
        docs = self.webx.load_urls(urls)
        return self._ingest_docs(docs, corpus_id)

    def crawl_site(
        self,
        start_url: str,
        *,
        max_pages: int = 30,
        same_domain: bool = True,
        allow_patterns: Optional[List[str]] = None,
        deny_patterns: Optional[List[str]] = None,
        corpus_id: str = "web",
    ) -> int:
        docs = self.webx.crawl_site(
            start_url,
            max_pages=max_pages,
            same_domain=same_domain,
            allow_patterns=allow_patterns,
            deny_patterns=deny_patterns,
        )
        return self._ingest_docs(docs, corpus_id)

    def _ingest_docs(self, docs: List[Document], corpus_id: str) -> int:
        if not docs:
            return 0
        chunks = self.chunker.split(docs)
        if not chunks:
            return 0
        upserted = self._upsert_chunks(chunks, corpus_id=corpus_id)
        self.retriever.refresh_sparse(chunks)
        return upserted

    def _upsert_chunks(self, chunks: List[Document], *, corpus_id: str) -> int:
        """
        Convert chunk Documents to StoredDoc and upsert into the vector DB.
        """
        # Ensure mandatory metadata are present
        for c in chunks:
            md = c.metadata or {}
            md.setdefault("created_at", now_iso())
            md.setdefault("corpus_id", corpus_id)
            c.metadata = md

        batch: List[StoredDoc] = docs_to_storeddocs(chunks)
        if not batch:
            return 0
        self.vstore.upsert(batch)
        return len(batch)

    # -------------------------------------------------------------------------
    # Blue/Green rebuild (Qdrant alias promotion)
    # -------------------------------------------------------------------------

    def rebuild_index_from_path(self, path: str, *, corpus_id: str = "default", new_collection_name: Optional[str] = None, recursive: bool = True, max_pdf_pages: Optional[int] = None) -> None:
        """
        Build a fresh collection from the given path, then atomically promote alias to it.
        """
        fresh_name = new_collection_name or f"{self.vstore.collection_alias}_{int(time.time())}"
        self.vstore.ensure_collection(collection=fresh_name)

        # Build docs
        docs: List[Document] = []
        docs.extend(self.pdfx.load_path(path, recursive=recursive, max_pages_per_file=max_pdf_pages))
        docs.extend(self.docxx.load_path(path, recursive=recursive))

        chunks = self.chunker.split(docs)
        batch: List[StoredDoc] = docs_to_storeddocs(chunks)
        # Upsert into the NEW collection
        self.vstore.upsert(batch, collection=fresh_name)
        # Promote alias => new collection
        self.vstore.promote_collection(fresh_name)
        print(f"[OK] Rebuild complete; alias '{self.vstore.collection_alias}' promoted to '{fresh_name}'.")

        # Local retriever refresh to use the newly built chunks in BM25 memory
        self.retriever.refresh_sparse(chunks)

    # -------------------------------------------------------------------------
    # Question answering
    # -------------------------------------------------------------------------

    def ask(self, query: str, *, role: str = "developer", k: Optional[int] = None, filters: Optional[Dict[str, object]] = None, temperature: float = 0.2) -> str:
        """
        Run hybrid retrieval, assemble prompts, and call the LLM.
        """
        # Retrieve
        docs = self.retriever.search(query, role=role, k=k, filters=filters)
        # Format context
        context = self._format_context(docs)
        # Build prompts
        prompts = build_prompts(query=query, role=role, context=context, include_legend=True)
        # Call LLM
        if not self.llm.api_key:
            raise RuntimeError("OpenRouter API key is not set. Use set_api_key() or pass in constructor.")
        #answer = self.llm.chat(prompts["system"], prompts["user"], temperature=temperature)
        answer = self.gemini_llm.gemini_chat(prompts["system"], prompts["user"], temperature=temperature)
        return answer

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _format_context(self, docs: List[Document]) -> str:
        if not docs:
            return "No relevant documentation found."
        blocks = []
        for i, d in enumerate(docs, start=1):
            md = d.metadata or {}
            file_name = md.get("file_name", "?")
            page_no = md.get("page_number", "?")
            doc_type = md.get("doc_type", "General")
            kind = md.get("chunk_kind", "text")
            blocks.append(
                f"--- DOCUMENT CHUNK {i} ---\n"
                f"SOURCE: {file_name} (Page {page_no})\n"
                f"CATEGORY: {doc_type} / {kind}\n"
                f"CONTENT: {d.page_content}\n"
            )
        return "\n".join(blocks)
