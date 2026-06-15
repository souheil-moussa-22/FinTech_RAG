"""Orchestrates the complete Retrieval-Augmented Generation (RAG) flow."""

from app.models.document import RetrievedChunk
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.vector_store import VectorStoreService


class RAGService:
    """Runs embedding, retrieval, prompt-building, and generation as one workflow."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService,
        llm_service: LLMService,
        top_k: int = 4,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.llm_service = llm_service
        self.top_k = top_k

    def answer_question(self, question: str) -> tuple[str, list[dict[str, int | str]]]:
        """Execute full RAG pipeline and return answer plus source citations."""
        query_embedding = self.embedding_service.embed_text(question)
        retrieved_chunks = self.vector_store_service.search(query_embedding=query_embedding, top_k=self.top_k)
        prompt = self._build_prompt(question=question, retrieved_chunks=retrieved_chunks)
        answer = self.llm_service.generate_answer(prompt)
        return answer, self._build_sources(retrieved_chunks)

    def _build_prompt(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        """Compose a grounded prompt that restricts model output to retrieved context."""
        if not retrieved_chunks:
            context = "No relevant context was found in the uploaded documents."
        else:
            context = "\n\n".join(
                [
                    (
                        f"[Document: {chunk.document_name} | Page: {chunk.page_number}]\n"
                        f"{chunk.text}"
                    )
                    for chunk in retrieved_chunks
                ]
            )

        return (
            "You are FinAssist, a financial assistant. "
            "Answer strictly from the provided context. "
            "If the answer is not in context, say you do not have enough information from uploaded documents.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    def _build_sources(self, retrieved_chunks: list[RetrievedChunk]) -> list[dict[str, int | str]]:
        """Build unique source references for API output."""
        seen: set[tuple[str, int]] = set()
        sources: list[dict[str, int | str]] = []
        for chunk in retrieved_chunks:
            key = (chunk.document_name, chunk.page_number)
            if key in seen:
                continue
            seen.add(key)
            sources.append({"document": chunk.document_name, "page": chunk.page_number})
        return sources
