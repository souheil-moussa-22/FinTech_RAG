"""ChromaDB wrapper service for persistent vector indexing and search."""

from pathlib import Path
from threading import Lock

from app.models.document import DocumentChunk, RetrievedChunk


class VectorStoreService:
    """Manages Chroma collection lifecycle, inserts, searches, and deletes."""

    def __init__(self, persist_directory: Path, collection_name: str) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._collection = None
        self._lock = Lock()

    def _get_collection(self):
        """Create or reuse a persistent Chroma collection."""
        if self._collection is None:
            with self._lock:
                if self._collection is None:
                    try:
                        import chromadb
                    except ImportError as exc:
                        raise RuntimeError("chromadb is not installed. Install requirements.txt dependencies.") from exc

                    client = chromadb.PersistentClient(path=str(self.persist_directory))
                    self._collection = client.get_or_create_collection(name=self.collection_name)
        return self._collection

    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Insert chunks and vectors into ChromaDB."""
        if not chunks:
            return

        collection = self._get_collection()
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "document_hash": chunk.document_id,
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                }
                for chunk in chunks
            ],
        )

    def document_exists(self, document_hash: str) -> bool:
        """Return whether a document hash is already indexed in collection metadata."""
        results = self._get_collection().get(where={"document_hash": document_hash}, limit=1)
        ids = results.get("ids", [])
        return bool(ids)

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Perform top-k semantic retrieval using cosine-distance-like ranking."""
        collection = self._get_collection()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            if not metadata:
                continue
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=str(metadata.get("document_id", "")),
                    document_name=str(metadata.get("document_name", "unknown")),
                    page_number=int(metadata.get("page_number", 0)),
                    text=document,
                    distance=float(distance),
                )
            )
        return retrieved

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks tied to one document id."""
        self._get_collection().delete(where={"document_id": document_id})
