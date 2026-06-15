"""API routes for document upload, listing, and deletion."""

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.schemas.document import DocumentInfo, DocumentUploadResponse


router = APIRouter(prefix="/documents", tags=["documents"])


def _list_document_files(documents_directory: Path) -> list[Path]:
    """Return all stored PDF files sorted by filename."""
    return sorted(documents_directory.glob("*.pdf"), key=lambda item: item.name)


def _extract_document_id(file_name: str) -> str:
    """Extract internal id from stored filename pattern '<id>__<name>.pdf'."""
    return file_name.split("__", maxsplit=1)[0]


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(request: Request, file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Upload a PDF, parse it, chunk it, embed it, and index it in ChromaDB."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are accepted.")

    document_id = str(uuid4())
    safe_filename = Path(file.filename).name
    stored_filename = f"{document_id}__{safe_filename}"
    output_path = request.app.state.settings.docs_directory / stored_filename

    file_bytes = await file.read()
    output_path.write_bytes(file_bytes)

    parser = request.app.state.parser_service
    chunker = request.app.state.chunker_service
    embedding = request.app.state.embedding_service
    vector_store = request.app.state.vector_store_service

    pages = parser.parse_pdf(output_path)
    if not pages:
        output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The PDF does not contain extractable text.")

    chunks = chunker.chunk_pages(document_id=document_id, document_name=safe_filename, pages=pages)
    embeddings = embedding.embed_texts([chunk.text for chunk in chunks])
    vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings)

    return DocumentUploadResponse(id=document_id, filename=safe_filename, chunks_indexed=len(chunks))


@router.get("", response_model=list[DocumentInfo])
def list_documents(request: Request) -> list[DocumentInfo]:
    """List uploaded documents currently stored in data/documents."""
    documents = []
    for file_path in _list_document_files(request.app.state.settings.docs_directory):
        file_name = file_path.name
        documents.append(
            DocumentInfo(
                id=_extract_document_id(file_name),
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
