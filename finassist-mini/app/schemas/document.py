"""Pydantic schemas for document management endpoints."""

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    """Response returned after a successful PDF upload and indexing."""

    id: str
    filename: str
    chunks_indexed: int


class DocumentInfo(BaseModel):
    """Metadata returned by document listing endpoint."""

    id: str
    filename: str
