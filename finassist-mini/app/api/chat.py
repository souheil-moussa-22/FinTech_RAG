"""API route for asking questions against indexed document knowledge."""

from fastapi import APIRouter, HTTPException, Request

from app.schemas.chat import ChatRequest, ChatResponse, ChatSource


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    """Answer one question via the RAG pipeline and return answer with citations."""
    rag_service = request.app.state.rag_service
    try:
        answer, sources = rag_service.answer_question(payload.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ChatResponse(answer=answer, sources=[ChatSource(**source) for source in sources])
