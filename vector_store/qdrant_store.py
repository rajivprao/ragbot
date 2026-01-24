
# -*- coding: utf-8 -*-
"""
vector_store/qdrant_store.py

Qdrant (Embedded) backend implementation for VectorStoreBase.

Features
--------
- Embedded Qdrant: runs on-disk without Docker/Cloud (QdrantClient(path="..."))
- Collection management with a stable "collection_alias" for blue/green swaps
- Upsert with idempotent IDs, accepts precomputed vectors or embeds text on the fly
- Equality-based metadata filtering (payload filters)
- Search (top-k) returning normalized SearchResult objects
- Delete by IDs or by filter, and count()
- Optional alias promotion (swap to a new concrete collection name atomically)

Requirements
------------
pip install qdrant-client

If you are using HuggingFaceEmbeddings (MiniLM-L6-v2), set vector_size=384.

Usage
-----
from vector_store.base import StoredDoc, SearchResult, VectorStoreBase
from vector_store.qdrant_store import QdrantStore

vdb = QdrantStore(
    collection_alias="enterprise_corpus",
    embeddings=your_embeddings,          # must implement .embed_query & .embed_documents
    path="qdrant_data",                  # embedded storage path
    vector_size=384,                     # dimension for your embedding model
    distance="cosine",                   # or "dot" / "euclid"
)
vdb.ensure_collection()                  # creates alias/collection if needed
vdb.upsert([...])                        # insert/update
hits = vdb.search("find things", k=10, filters={"doc_type":"erd"})
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    PointStruct,
    CreateAlias,
    DeleteAlias,
    AliasOperations,
)

from vector_store.base import (
    StoredDoc,
    SearchResult,
    VectorStoreBase,
    VectorStoreError,
)

from qdrant_client.models import Filter, FieldCondition, MatchValue
# ----------------------------
# Helpers
# ----------------------------

def _to_distance(distance: Union[str, Distance]) -> Distance:
    if isinstance(distance, Distance):
        return distance
    s = (distance or "").strip().lower()
    if s in ("cos", "cosine"):
        return Distance.COSINE
    if s in ("dot", "ip", "inner", "inner_product"):
        return Distance.DOT
    if s in ("l2", "euclid", "euclidean"):
        return Distance.EUCLID
    raise ValueError(f"Unsupported distance: {distance}")


def _build_filter(filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
    """
    Build a Qdrant Filter with simple equality conditions:
      { "key1": value1, "key2": value2, ... }
    """
    if not filters:
        return None
    must = []
    for k, v in filters.items():
        must.append(FieldCondition(key=str(k), match=MatchValue(value=v)))
    return Filter(must=must)


# ----------------------------
# QdrantStore
# ----------------------------

class QdrantStore(VectorStoreBase):
    """
    Qdrant-backed vector store (Embedded mode by default).

    Constructor Parameters
    ----------------------
    collection_alias : str
        Stable alias used by the app (e.g., "enterprise_corpus").
        You can later promote a new concrete collection to this alias.
    embeddings : Any
        Object with .embed_query(text) -> List[float] and
        .embed_documents(texts) -> List[List[float]].
        If None, you must provide precomputed vectors in StoredDoc.vector.
    path : str
        Local directory for embedded Qdrant storage (default: "qdrant_data").
    vector_size : int
        Dimension of your embedding vectors (e.g., 384 for MiniLM-L6-v2).
    distance : str | Distance
        "cosine" (default), "dot", or "euclid".
    """

    def __init__(
        self,
        collection_alias: str,
        embeddings: Any = None,
        *,
        path: str = "qdrant_data",
        vector_size: int = 384,
        distance: Union[str, Distance] = "cosine",
        **kwargs: Any,
    ) -> None:
        super().__init__(collection_alias=collection_alias, embeddings=embeddings, **kwargs)

        self.client = QdrantClient(path=path)
        self.vector_size = int(vector_size)
        self.distance = _to_distance(distance)

        # In embedded mode, we can use the alias name also as the initial concrete collection name
        # unless you plan a blue/green swap. ensure_collection() will create if missing.
        self.ensure_collection()

    # ---- Lifecycle / collections ----

    def ensure_collection(self, collection: Optional[str] = None, **kwargs: Any) -> None:
        """
        Ensure a concrete collection exists and the alias points to it (or to an existing one).
        If alias doesn't exist, we create a collection named = alias, and use it directly.
        """
        target = collection or self.collection_alias

        # If a collection with 'target' exists, do nothing; else create it
        existing = {c.name for c in self.client.get_collections().collections}
        if target not in existing:
            self.client.create_collection(
                collection_name=target,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance,
                ),
            )
        # If you plan to use alias indirection later (blue/green), you can keep alias==collection for now.
        # You can promote later via promote_collection(new_concrete_name).

    def load(self) -> None:
        # Embedded client re-opens as needed; nothing special required.
        return

    def save(self) -> None:
        # Qdrant persists automatically; nothing to do.
        return

    def close(self) -> None:
        # Nothing required for embedded client.
        return

    # ---- Blue/Green alias management ----

    def promote_collection(self, new_collection_name: str) -> None:
        """
        Atomically map the alias to `new_collection_name`.
        Note: This assumes `new_collection_name` already exists and is fully built.
        """
        ops = AliasOperations(
            actions=[
                # Remove alias if already mapped (safe if it doesn't exist)
                DeleteAlias(alias_name=self.collection_alias),
                # Create alias -> new collection
                CreateAlias(alias_name=self.collection_alias, collection_name=new_collection_name),
            ]
        )
        self.client.update_aliases(operations=ops)

    # ---- Write paths ----

    def upsert(self, docs: Iterable[StoredDoc], collection: Optional[str] = None) -> None:
        """
        Insert/update points. If vectors are not provided, compute embeddings using self.embeddings.
        """
        target = collection or self.collection_alias
        docs_list = list(docs)
        if not docs_list:
            return

        # Prepare vectors (batch embedding when needed)
        texts_to_embed: List[str] = []
        idx_map: List[int] = []  # indices of docs that need embedding
        vectors: List[Optional[List[float]]] = [None] * len(docs_list)

        for i, d in enumerate(docs_list):
            if d.vector is not None:
                vectors[i] = list(d.vector)
            else:
                texts_to_embed.append(d.text)
                idx_map.append(i)

        if texts_to_embed:
            if self.embeddings is None:
                raise VectorStoreError(
                    "Embeddings are required to upsert docs without precomputed vectors."
                )
            # embed_documents expects List[str] -> List[List[float]]
            embeds = self.embeddings.embed_documents(texts_to_embed)
            if not embeds or any(e is None for e in embeds):
                raise VectorStoreError("Embedding failed for one or more documents.")
            if any(len(e) != self.vector_size for e in embeds):
                raise VectorStoreError(
                    f"Embedding size mismatch. Expected {self.vector_size}, got {[len(e) for e in embeds][:3]}..."
                )
            for j, vec in enumerate(embeds):
                vectors[idx_map[j]] = vec

        # Build Qdrant points
        points: List[PointStruct] = []
        for i, d in enumerate(docs_list):
            vec = vectors[i]
            if vec is None:
                raise VectorStoreError("Internal error: vector is None after embedding.")
            payload = dict(d.metadata or {})
            # Store text in payload for retrieval/citations
            payload.setdefault("text", d.text)

            # Qdrant supports ids as str or int; we keep them as str for stability
            points.append(PointStruct(id=str(d.id), vector=vec, payload=payload))

        # Upsert in one go (Qdrant handles batch size internally; for very large batches split manually)
        self.client.upsert(collection_name=target, points=points)

    def delete_by_ids(self, ids: Sequence[Union[str, int]], collection: Optional[str] = None) -> int:
        target = collection or self.collection_alias
        if not ids:
            return 0
        # Qdrant delete accepts a list of points
        self.client.delete(collection_name=target, points_selector=list(ids))
        return len(ids)

    def delete_by_filter(self, filters: Dict[str, Any], collection: Optional[str] = None) -> int:
        target = collection or self.collection_alias
        qf = _build_filter(filters)
        if not qf:
            return 0
        self.client.delete(collection_name=target, points_selector=qf)
        # Qdrant doesn't return deleted count synchronously; return -1 to indicate unknown
        return -1

    # ---- Read paths ----

    def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Vector similarity search with optional payload filters.
        """
        target = collection or self.collection_alias

        if self.embeddings is None:
            raise VectorStoreError("Embeddings are required to search with a text query.")
        qvec = self.embeddings.embed_query(query)
        if qvec is None or len(qvec) != self.vector_size:
            raise VectorStoreError(
                f"Query embedding size mismatch. Expected {self.vector_size}, got {len(qvec) if qvec else None}."
            )

        qf = _build_filter(filters)

        """res = self.client.search(
            collection_name=target,
            query_vector=qvec,
            limit=int(k),
            query_filter=qf,
            with_payload=True,
            with_vectors=False,
        )

        res = self.client.query_points( 
            collection_name=target, 
            query=qvec, 
            limit=int(k), 
            query_filter=qf
        )"""

        res,_ = self.client.scroll( 
            collection_name=target, 
            with_vectors=qvec, 
            limit=int(k), 
            scroll_filter=qf
        )

        out: List[SearchResult] = []
        for r in res:
            payload = r.payload or {}
            text = payload.get("text", "")
            # Normalize score to float (Qdrant returns 'score' with backend meaning)
            score = float(r.score) if hasattr(r, "score") else 0.0
            out.append(SearchResult(id=str(r.id), text=text, metadata=payload, score=score))
        return out

    def count(self, filters: Optional[Dict[str, Any]] = None, collection: Optional[str] = None) -> int:
        target = collection or self.collection_alias
        qf = _build_filter(filters)
        try:
            # exact=False is faster (approximate). Use exact=True when you need precise count.
            cnt = self.client.count(collection_name=target, count_filter=qf, exact=False)
            # qdrant-client returns an object with 'count' attribute
            return int(getattr(cnt, "count", 0))
        except Exception:
            return 0
