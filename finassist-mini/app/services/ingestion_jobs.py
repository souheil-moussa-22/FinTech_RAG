"""In-memory ingestion job tracking for background upload processing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from threading import Lock
from time import time
from typing import Literal
from uuid import uuid4


JobState = Literal["processing", "completed", "failed"]


@dataclass(slots=True)
class IngestionJob:
    """Represents lifecycle state and outcome for one ingestion task."""

    job_id: str
    status: JobState
    document_id: str
    filename: str
    document_hash: str
    chunks_indexed: int = 0
    detail: str | None = None
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)


class IngestionJobService:
    """Thread-safe tracker for ingestion jobs returned by upload API."""

    def __init__(self) -> None:
        self._jobs: dict[str, IngestionJob] = {}
        self._lock = Lock()

    def create_job(self, document_id: str, filename: str, document_hash: str) -> IngestionJob:
        job = IngestionJob(
            job_id=str(uuid4()),
            status="processing",
            document_id=document_id,
            filename=filename,
            document_hash=document_hash,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def mark_completed(self, job_id: str, chunks_indexed: int) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = "completed"
            job.chunks_indexed = chunks_indexed
            job.updated_at = time()

    def mark_failed(self, job_id: str, detail: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = "failed"
            job.detail = detail
            job.updated_at = time()

    def get_job(self, job_id: str) -> dict[str, str | int | float | None] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return asdict(job)
