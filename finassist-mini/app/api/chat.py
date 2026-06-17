"""API route for asking questions against indexed document knowledge."""

import asyncio

from fastapi import APIRouter, HTTPException, Request

from app.schemas.chat import ChatRequest, ChatResponse, ChatSource


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    """Answer one question via the RAG pipeline and return answer with citations."""
    semaphore = request.app.state.chat_semaphore
    wait_seconds = request.app.state.settings.admission_wait_seconds
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=wait_seconds)
    except TimeoutError as exc:
        raise HTTPException(status_code=429, detail="Chat service is busy. Please retry shortly.") from exc

    rag_service = request.app.state.rag_service
    try:
        answer, sources = rag_service.answer_question(payload.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        semaphore.release()

    return ChatResponse(answer=answer, sources=[ChatSource(**source) for source in sources])
