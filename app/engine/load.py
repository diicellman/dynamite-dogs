"""Load data functions."""

import logging
import os
from typing import Dict

from dotenv import load_dotenv

from app.models.models import GitHubPayload

from llama_index.core import Settings, StorageContext, VectorStoreIndex

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.readers.github import GithubClient, GithubRepositoryReader
from llama_index.vector_stores.lancedb import LanceDBVectorStore


# global settings
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "your_token")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "localhost")
DATA_DIR = "data"


Settings.embed_model = OllamaEmbedding(
    model_name="mxbai-embed-large:latest", base_url=OLLAMA_BASE_URL
)


def build_index(github_payload: GitHubPayload) -> Dict:
    """Builds repo index in LanceDB"""
    logger = logging.getLogger("uvicorn")
    github_client = GithubClient(GITHUB_TOKEN)
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=5,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    logger.info("Loading data from %s.", github_payload.repo)
    loader = GithubRepositoryReader(
        github_client=github_client,
        owner=github_payload.owner,
        repo=github_payload.repo,
    )
    documents = loader.load_data(branch="main")

    nodes = node_parser.get_nodes_from_documents(documents)

    logger.info("Creating new index.")

    github_table_name = github_payload.repo
    github_vector_store = LanceDBVectorStore(
        uri=f"{DATA_DIR}/lancedb", table_name=github_table_name
    )
    github_storage_context = StorageContext.from_defaults(
        vector_store=github_vector_store
    )

    github_index = VectorStoreIndex(
        nodes,
        storage_context=github_storage_context,
        show_progress=True,
        embed_model=Settings.embed_model,
    )

    logger.info(
        "Successfully created embeddings for %s in the LanceDB.",
        github_payload.repo,
    )

    return {"message": "success!"}
