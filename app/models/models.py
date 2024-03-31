"""Pydantic models."""

from pydantic import BaseModel, Field


class MessageData(BaseModel):
    """Datamodel for messages."""

    query: str = Field(..., description="Query message.")


class RAGResponse(BaseModel):
    """Datamodel for RAG response."""

    question: str = Field(..., description="User's question/query.")
    answer: str = Field(..., description="Prediction answer.")


class GitHubPayload(BaseModel):
    """Datamodel for GitHub entity"""

    owner: str = Field(..., description="Repository's owner name.")
    repo: str = Field(..., description="Repository name.")
