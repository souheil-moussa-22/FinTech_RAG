"""Pydantic schemas for chat input/output validation."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Input payload for asking the assistant a question."""

    question: str = Field(..., min_length=1, description="User question about uploaded documents")


class ChatSource(BaseModel):
    """One citation entry returned alongside the answer."""

    document: str
    page: int


class ChatResponse(BaseModel):
    """Output payload for the chat endpoint."""

    answer: str
    sources: list[ChatSource]
