"""Domain models used across parsing, chunking, retrieval, and API layers."""

from dataclasses import dataclass


@dataclass(slots=True)
class ParsedPage:
    """Represents extracted text from one PDF page.

    Attributes:
        page_number: One-based page index in the source PDF.
        text: Extracted textual content for the page.
    """

    page_number: int
    text: str


@dataclass(slots=True)
class DocumentChunk:
    """Represents one chunk prepared for vector indexing.

    Attributes:
        chunk_id: Stable chunk identifier.
        document_id: Internal document identifier.
        document_name: User-facing document name.
        page_number: Source page number.
        text: Chunk content.
    """

    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    text: str


@dataclass(slots=True)
class RetrievedChunk:
    """Represents one retrieved chunk returned by semantic search."""

    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    text: str
    distance: float
