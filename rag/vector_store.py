# rag/vector_store.py

from qdrant_client import QdrantClient
from qdrant_client.http import models
import hashlib

class VectorStore:
    def __init__(self, collection_name="rag_collection", host="localhost", port=6333):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

        # Create collection if not exists
        if not self.client.get_collections().collections or \
           self.collection_name not in [c.name for c in self.client.get_collections().collections]:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)  # adjust size to your embedding model
            )

    def _doc_id(self, text: str) -> str:
        """Generate a stable hash ID for a document chunk."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def exists(self, text: str) -> bool:
        """Check if a document chunk already exists in the vector DB."""
        doc_id = self._doc_id(text)
        result = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[doc_id]
        )
        return len(result) > 0

    def insert(self, text: str, embedding: list, metadata: dict = None):
        """Insert a new document chunk if not already present."""
        doc_id = self._doc_id(text)
        if not self.exists(text):
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload={"text": text, **(metadata or {})}
                    )
                ]
            )

    def search(self, query_embedding: list, top_k: int = 5):
        """Search for nearest neighbors."""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        return [
            {"id": r.id, "score": r.score, "text": r.payload.get("text")}
            for r in results
        ]
