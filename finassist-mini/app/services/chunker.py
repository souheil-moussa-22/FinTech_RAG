"""Text chunking service for building retrieval-ready document chunks."""

from app.models.document import DocumentChunk, ParsedPage


class ChunkerService:
    """Splits parsed pages into fixed-size overlapping word chunks."""

    def __init__(self, chunk_size_words: int = 500, chunk_overlap_words: int = 50) -> None:
        """Configure chunk size and overlap behavior.

        Overlap helps preserve context between adjacent chunks so facts near a chunk
        boundary remain retrievable even when a sentence spans two chunks.
        """
        if chunk_overlap_words >= chunk_size_words:
            raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words

    def chunk_pages(
        self,
        document_id: str,
        document_name: str,
        pages: list[ParsedPage],
    ) -> list[DocumentChunk]:
        """Convert parsed pages into overlapping chunks.

        Args:
            document_id: Internal document identifier.
            document_name: Human-readable source name.
            pages: Parsed page data from PDF parser.

        Returns:
            A list of DocumentChunk instances.
        """
        chunks: list[DocumentChunk] = []
        step = self.chunk_size_words - self.chunk_overlap_words

        for parsed_page in pages:
            words = parsed_page.text.split()
            if not words:
                continue

            for chunk_index, start_idx in enumerate(range(0, len(words), step)):
                chunk_words = words[start_idx : start_idx + self.chunk_size_words]
                if not chunk_words:
                    continue
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document_id}_p{parsed_page.page_number}_c{chunk_index}",
                        document_id=document_id,
                        document_name=document_name,
                        page_number=parsed_page.page_number,
                        text=" ".join(chunk_words),
                    )
                )

        return chunks
