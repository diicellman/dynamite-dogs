"""Index and engines generation functions."""

import logging
import os

from dotenv import load_dotenv
import lancedb
from pydantic import Field

from app.engine.prompts import (
    CODE_RAG_CHAT_HAIKU_PROMPT,
    CODE_SUB_QUESTION_TEXT_QA_HAIKU_PROMPT,
    SOCIAL_MEDIA_SUB_QUESTION_TEXT_QA_HAIKU_PROMPT,
)

from llama_index.core import Settings, VectorStoreIndex

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine


# global default
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "your_key")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "localhost")
DATA_DIR = "data"


Settings.embed_model = OllamaEmbedding(
    model_name="mxbai-embed-large:latest", base_url=OLLAMA_BASE_URL
)
Settings.llm = Anthropic(
    model="claude-3-haiku-20240307", api_key=ANTHROPIC_API_KEY, max_tokens=2048
)


def get_doc_sub_question_query_engine(
    streaming: bool = Field(
        default=False, description="Param for frontend, by default is False"
    )
):
    """Builds a sub question query engine for documentation"""
    logger = logging.getLogger("uvicorn")
    logger.info("Connecting to LanceDB...")
    lance_db_uri = f"{DATA_DIR}/lancedb"
    db = lancedb.connect(lance_db_uri)
    github_table_names = db.table_names()
    logger.info("Connected.")

    colbert_reranker = ColbertRerank(
        top_n=5,
        model="colbert-ir/colbertv2.0",
        tokenizer="colbert-ir/colbertv2.0",
        keep_retrieval_score=False,
    )

    logger.info("Building indexes.")
    query_tools = []
    for github_table in github_table_names:
        vector_store = LanceDBVectorStore(uri=lance_db_uri, table_name=github_table)

        index = VectorStoreIndex.from_vector_store(vector_store)

        query_engine = index.as_query_engine(
            similarity_top_k=10,
            text_qa_template=CODE_RAG_CHAT_HAIKU_PROMPT,
            streaming=streaming,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window"),
                colbert_reranker,
            ],
        )

        github_summary = (
            f"Useful for any questions related to {github_table} github repository."
        )
        query_tool = QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name=f"tool_{github_table}",
                description=github_summary,
            ),
        )
        query_tools.append(query_tool)

    doc_sub_query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_tools,
        llm=Settings.llm,
        verbose=True,
        use_async=True,
    )
    logger.info("Success.")

    doc_sub_query_engine.update_prompts(
        {
            "response_synthesizer:text_qa_template": CODE_SUB_QUESTION_TEXT_QA_HAIKU_PROMPT
        }
    )

    return doc_sub_query_engine


def get_social_media_sub_question_query_engine(
    streaming: bool = Field(
        default=False, description="Param for frontend, by default is False"
    )
):
    """Builds a sub question query engine for social media posts"""
    logger = logging.getLogger("uvicorn")
    logger.info("Connecting to LanceDB...")
    lance_db_uri = f"{DATA_DIR}/lancedb"
    db = lancedb.connect(lance_db_uri)
    github_table_names = db.table_names()
    logger.info("Connected.")

    colbert_reranker = ColbertRerank(
        top_n=5,
        model="colbert-ir/colbertv2.0",
        tokenizer="colbert-ir/colbertv2.0",
        keep_retrieval_score=False,
    )

    logger.info("Building indexes.")
    query_tools = []
    for github_table in github_table_names:
        vector_store = LanceDBVectorStore(uri=lance_db_uri, table_name=github_table)

        index = VectorStoreIndex.from_vector_store(vector_store)

        query_engine = index.as_query_engine(
            similarity_top_k=10,
            text_qa_template=CODE_RAG_CHAT_HAIKU_PROMPT,
            streaming=streaming,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window"),
                colbert_reranker,
            ],
        )

        github_summary = (
            f"Useful for any questions related to {github_table} github repository."
        )
        query_tool = QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name=f"tool_{github_table}",
                description=github_summary,
            ),
        )
        query_tools.append(query_tool)

    social_media_sub_query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_tools,
        llm=Settings.llm,
        verbose=True,
        use_async=True,
    )
    logger.info("Success.")

    social_media_sub_query_engine.update_prompts(
        {
            "response_synthesizer:text_qa_template": SOCIAL_MEDIA_SUB_QUESTION_TEXT_QA_HAIKU_PROMPT
        }
    )

    return social_media_sub_query_engine
