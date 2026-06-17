"""Pydantic schemas for document management endpoints."""

from typing import Literal

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    """Response returned after a successful PDF upload and indexing."""

    job_id: str
    status: Literal["processing", "completed", "failed"]
    id: str
    filename: str
    document_hash: str
    chunks_indexed: int | None = None
    detail: str | None = None


class DocumentInfo(BaseModel):
    """Metadata returned by document listing endpoint."""

    id: str
    filename: str


class IngestionJobStatusResponse(BaseModel):
    """Response payload for upload background job status."""

    job_id: str
    status: Literal["processing", "completed", "failed"]
    document_id: str
    filename: str
    document_hash: str
    chunks_indexed: int
    detail: str | None = None
    created_at: float
    updated_at: float
