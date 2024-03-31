from dotenv import load_dotenv

load_dotenv()

import logging
import os

import uvicorn
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers.rag import rag_router
from app.gradio_ui.ui import demo_dynamite_dogs

from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from langfuse.llama_index import LlamaIndexCallbackHandler


LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "your_key")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "your_key")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "hostname")

do_not_instrument = os.getenv("INSTRUMENT_LLAMA", "true") == "false"
if not do_not_instrument:
    logger = logging.getLogger("uvicorn")

    # instrument()

    langfuse_callback_handler = LlamaIndexCallbackHandler(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST,
    )
    Settings.callback_manager = CallbackManager([langfuse_callback_handler])
    logger.info("Tracing is ON.")


app = FastAPI(title="Dynamite Dogs 🐕")

environment = os.getenv("ENVIRONMENT", "dev")  # Default to 'development' if not set

if environment == "dev":
    logger = logging.getLogger("uvicorn")
    logger.warning("Running in development mode - allowing CORS for all origins")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(rag_router, prefix="/api/rag")
app = gr.mount_gradio_app(app, demo_dynamite_dogs, path="/gradio")


if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=8000, reload=True)
