"""Gradio Chat ui component."""

import gradio as gr
from typing import List

from app.engine.index import (
    get_doc_sub_question_query_engine,
    get_social_media_sub_question_query_engine,
)


async def doc_query(message: str, history: List[list]):
    sub_query_engine = get_doc_sub_question_query_engine()

    bot_message = await sub_query_engine.aquery(message)

    return bot_message.response


async def social_media_query(message: str, history: List[list]):
    sub_query_engine = get_social_media_sub_question_query_engine()

    bot_message = await sub_query_engine.aquery(message)

    return bot_message.response


doc_chat = gr.ChatInterface(
    doc_query,
    chatbot=gr.Chatbot(show_copy_button=True, layout="panel", height="55vh"),
    textbox=gr.Textbox(placeholder="Enter your message", container=False, scale=7),
    description="Useful for creating documentation related to a GitHub repository.",
    submit_btn="Submit ⌲",
    clear_btn="Clear 🗑️",
    undo_btn=None,
    retry_btn=None,
    fill_height=True,
)


social_media_chat = gr.ChatInterface(
    social_media_query,
    chatbot=gr.Chatbot(show_copy_button=True, layout="panel", height="55vh"),
    textbox=gr.Textbox(placeholder="Enter your message", container=False, scale=7),
    description="Useful for creating social media posts related to a GitHub repository.",
    submit_btn="Submit ⌲",
    clear_btn="Clear 🗑️",
    undo_btn=None,
    retry_btn=None,
    fill_height=True,
)


demo_chat_ui = gr.TabbedInterface(
    [doc_chat, social_media_chat],
    ["Docs Chat 💬", "Social media 🌐"],
)
