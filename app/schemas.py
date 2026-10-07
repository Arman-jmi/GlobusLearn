"""Pydantic request/response models for the API."""
from pydantic import BaseModel, Field


class SessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    chapter: str
    page: int


class ChatResponse(BaseModel):
    answer: str
    source: str | None = None  # e.g. "Chapter 1 – Chemical Reactions and Equations"
    sources: list[Source] = []
    in_scope: bool = True
    cache_hit: bool = False
    latency_ms: int = 0


class ImageChatResponse(ChatResponse):
    question: str | None = None  # the question read from the image
    extraction_failed: bool = False  # True when no usable question could be read (answer holds a friendly message)


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class StoredMessage(BaseModel):
    id: int
    role: str
    content: str
    message_type: str = "text"  # text | image | solution
    source: str | None = None
    sources: list[Source] = []
    in_scope: bool | None = None
    cache_hit: bool | None = None
    latency_ms: int | None = None
    has_image: bool = False
    created_at: str


class ConversationDetail(ConversationSummary):
    messages: list[StoredMessage]
