"""API routes for document upload, listing, deletion, and ingestion job status."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from time import perf_counter

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile, status

from app.schemas.document import DocumentInfo, DocumentUploadResponse, IngestionJobStatusResponse
from app.services.observability import current_memory_mb, timed_stage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


def _list_document_files(documents_directory: Path) -> list[Path]:
    """Return all stored PDF files sorted by filename."""
    return sorted(documents_directory.glob("*.pdf"), key=lambda item: item.name)


def _extract_document_id(file_name: str) -> str:
    """Extract internal id from stored filename pattern '<id>__<name>.pdf'."""
    return file_name.split("__", maxsplit=1)[0]


def _delete_document_files(documents_directory: Path, document_id: str) -> None:
    for file_path in _list_document_files(documents_directory):
        if _extract_document_id(file_path.name) == document_id:
            file_path.unlink(missing_ok=True)


def _run_ingestion_job(request: Request, job_id: str, document_id: str, filename: str, output_path: Path) -> None:
    parser = request.app.state.parser_service
    chunker = request.app.state.chunker_service
    embedding = request.app.state.embedding_service
    vector_store = request.app.state.vector_store_service
    jobs = request.app.state.ingestion_job_service
    ingestion_gate = request.app.state.ingestion_semaphore

    ingestion_gate.acquire()
    started = perf_counter()
    try:
        try:
            with timed_stage() as parse_stage:
                pages = parser.parse_pdf(output_path)
            if not pages:
                _delete_document_files(request.app.state.settings.docs_directory, document_id)
                jobs.mark_failed(job_id, "The PDF does not contain extractable text.")
                return

            with timed_stage() as chunk_stage:
                chunks = chunker.chunk_pages(document_id=document_id, document_name=filename, pages=pages)
            with timed_stage() as embed_stage:
                embeddings = embedding.embed_texts([chunk.text for chunk in chunks])
            with timed_stage() as upsert_stage:
                vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings)
            jobs.mark_completed(job_id=job_id, chunks_indexed=len(chunks))
            logger.info(
                "upload_pipeline_metrics filename=%s pages=%d chunks=%d embeddings=%d parse_s=%.3f chunk_s=%.3f embed_s=%.3f upsert_s=%.3f total_s=%.3f py_mem_mb=%.2f",
                filename,
                len(pages),
                len(chunks),
                len(embeddings),
                parse_stage[0],
                chunk_stage[0],
                embed_stage[0],
                upsert_stage[0],
                perf_counter() - started,
                current_memory_mb(),
            )
        except Exception as exc:
            _delete_document_files(request.app.state.settings.docs_directory, document_id)
            request.app.state.vector_store_service.delete_document(document_id=document_id)
            jobs.mark_failed(job_id, "Failed to process uploaded document.")
            logger.exception("Ingestion failed for job_id=%s filename=%s: %s", job_id, filename, exc)
    finally:
        ingestion_gate.release()


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> DocumentUploadResponse:
    """Upload a PDF, persist it by streaming, and process ingestion in the background."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are accepted.")

    semaphore = request.app.state.upload_semaphore
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=0.05)
    except TimeoutError as exc:
        raise HTTPException(status_code=429, detail="Upload service is busy. Please retry shortly.") from exc

    settings = request.app.state.settings
    safe_filename = Path(file.filename).name
    hasher = hashlib.sha256()
    max_upload_size_bytes = settings.max_upload_size_mb * 1024 * 1024
    bytes_written = 0
    tmp_output_path = settings.docs_directory / f"tmp__{safe_filename}"

    try:
        with tmp_output_path.open("wb") as handle:
            while True:
                chunk = await file.read(settings.upload_write_chunk_bytes)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_upload_size_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds {settings.max_upload_size_mb} MB limit.",
                    )
                hasher.update(chunk)
                handle.write(chunk)
        document_hash = hasher.hexdigest()
        document_id = document_hash
        output_path = settings.docs_directory / f"{document_id}__{safe_filename}"

        if request.app.state.vector_store_service.document_exists(document_hash):
            tmp_output_path.unlink(missing_ok=True)
            logger.info(
                "upload_deduplicated filename=%s bytes=%d document_hash=%s py_mem_mb=%.2f",
                safe_filename,
                bytes_written,
                document_hash,
                current_memory_mb(),
            )
            job = request.app.state.ingestion_job_service.create_job(document_id, safe_filename, document_hash)
            request.app.state.ingestion_job_service.mark_completed(job.job_id, 0)
            return DocumentUploadResponse(
                job_id=job.job_id,
                status="completed",
                id=document_id,
                filename=safe_filename,
                document_hash=document_hash,
                chunks_indexed=0,
                detail="Document already indexed. Reused existing embeddings.",
            )

        _delete_document_files(settings.docs_directory, document_id)
        tmp_output_path.replace(output_path)
        job = request.app.state.ingestion_job_service.create_job(document_id, safe_filename, document_hash)
        background_tasks.add_task(_run_ingestion_job, request, job.job_id, document_id, safe_filename, output_path)
        logger.info(
            "upload_request_metrics filename=%s bytes=%d document_hash=%s status=processing py_mem_mb=%.2f",
            safe_filename,
            bytes_written,
            document_hash,
            current_memory_mb(),
        )
        return DocumentUploadResponse(
            job_id=job.job_id,
            status=job.status,
            id=document_id,
            filename=safe_filename,
            document_hash=document_hash,
            chunks_indexed=None,
        )
    finally:
        semaphore.release()
        await file.close()
        if tmp_output_path.exists():
            tmp_output_path.unlink(missing_ok=True)


@router.get("/jobs/{job_id}", response_model=IngestionJobStatusResponse)
def get_ingestion_job_status(job_id: str, request: Request) -> IngestionJobStatusResponse:
    """Fetch ingestion job status for polling upload progress and result."""
    job = request.app.state.ingestion_job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion job not found.")
    return IngestionJobStatusResponse(**job)


@router.get("", response_model=list[DocumentInfo])
def list_documents(request: Request) -> list[DocumentInfo]:
    """List uploaded documents currently stored in data/documents."""
    documents = []
    for file_path in _list_document_files(request.app.state.settings.docs_directory):
        file_name = file_path.name
        document_id = _extract_document_id(file_name)
        if document_id == "tmp":
            continue
        documents.append(
            DocumentInfo(
                id=document_id,
                filename=file_name.split("__", maxsplit=1)[1] if "__" in file_name else file_name,
            )
        )
    return documents


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, request: Request) -> None:
    """Delete stored PDF and remove its chunks from ChromaDB collection."""
    docs_directory = request.app.state.settings.docs_directory
    target_files = [path for path in _list_document_files(docs_directory) if _extract_document_id(path.name) == document_id]
    if not target_files:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    for file_path in target_files:
        file_path.unlink(missing_ok=True)

    request.app.state.vector_store_service.delete_document(document_id=document_id)
