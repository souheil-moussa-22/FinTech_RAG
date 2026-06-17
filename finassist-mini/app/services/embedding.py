"""Embedding service wrapper around sentence-transformers."""

from threading import Lock


class EmbeddingService:
    """Loads one embedding model once and exposes encode helpers."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._lock = Lock()

    def _load_model(self):
        """Load sentence-transformers model lazily on first use."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError as exc:
                        raise RuntimeError(
                            "sentence-transformers is not installed. Install requirements.txt dependencies."
                        ) from exc
                    self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Generate one embedding vector for a single text."""
        vector = self._load_model().encode(text)
        return vector.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts in one call."""
        vectors = self._load_model().encode(texts)
        return vectors.tolist()
