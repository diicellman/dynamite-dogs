"""Endpoints."""

from fastapi import APIRouter

from app.engine.index import (
    get_doc_sub_question_query_engine,
    get_social_media_sub_question_query_engine,
)
from app.engine.load import build_index
from app.models.models import (
    MessageData,
    RAGResponse,
    GitHubPayload,
)

rag_router = APIRouter()


@rag_router.post("/create-indexes", tags=["Load"])
def create_index(payload: GitHubPayload):
    return build_index(github_payload=payload)


@rag_router.post("/doc-sub-question-query", response_model=RAGResponse, tags=["RAG"])
async def doc_sub_question_query(message_data: MessageData):
    query_engine = get_doc_sub_question_query_engine(
        stream_response=message_data.stream_response
    )

    response = await query_engine.aquery(message_data.query)

    return RAGResponse(question=message_data.query, answer=response.response)


@rag_router.post(
    "/social-media-sub-question-query", response_model=RAGResponse, tags=["RAG"]
)
async def social_media_sub_question_query(message_data: MessageData):
    query_engine = get_social_media_sub_question_query_engine(
        stream_response=message_data.stream_response
    )

    response = await query_engine.aquery(message_data.query)

    return RAGResponse(question=message_data.query, answer=response.response)
