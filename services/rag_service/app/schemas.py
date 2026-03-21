from typing import Literal

from pydantic import BaseModel, Field


class ChatHistoryMessage(BaseModel):
    role: Literal["assistant", "system", "user"]
    content: str


class ChatRequest(BaseModel):
    doc_id: str
    query: str
    history: list[ChatHistoryMessage] = Field(default_factory=list)


class UploadResponse(BaseModel):
    doc_id: str
    chunks: int
    source_id: str