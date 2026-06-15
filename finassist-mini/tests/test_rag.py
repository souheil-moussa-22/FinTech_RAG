"""Focused unit tests for RAG orchestration behavior."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.document import RetrievedChunk
from app.services.rag import RAGService


class FakeEmbeddingService:
    """Simple embedding stub used for deterministic tests."""

    def embed_text(self, text: str):
        return [0.1, 0.2]


class FakeVectorStoreService:
    """Simple retrieval stub returning preconfigured chunks."""

    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query_embedding, top_k: int):
        return self.chunks[:top_k]


class FakeLLMService:
    """Simple LLM stub that captures prompt input."""

    def __init__(self):
        self.last_prompt = ""

    def generate_answer(self, prompt: str):
        self.last_prompt = prompt
        return "Stub answer"


class RAGServiceTests(unittest.TestCase):
    """Validate prompt grounding and source extraction."""

    def test_rag_returns_answer_and_unique_sources(self):
        chunks = [
            RetrievedChunk(
                chunk_id="1",
                document_id="doc-a",
                document_name="fees.pdf",
                page_number=2,
                text="Transfer fee is 5 USD.",
                distance=0.1,
            ),
            RetrievedChunk(
                chunk_id="2",
                document_id="doc-a",
                document_name="fees.pdf",
                page_number=2,
                text="Duplicate page context.",
                distance=0.2,
            ),
        ]

        llm = FakeLLMService()
        service = RAGService(
            embedding_service=FakeEmbeddingService(),
            vector_store_service=FakeVectorStoreService(chunks),
            llm_service=llm,
            top_k=4,
        )

        answer, sources = service.answer_question("What are transfer fees?")

        self.assertEqual(answer, "Stub answer")
        self.assertEqual(sources, [{"document": "fees.pdf", "page": 2}])
        self.assertIn("Transfer fee is 5 USD.", llm.last_prompt)


if __name__ == "__main__":
    unittest.main()
