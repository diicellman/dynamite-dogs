import gradio as gr

from app.gradio_ui.chat_ui import demo_chat_ui
from app.gradio_ui.load_ui import load_ui

demo_dynamite_dogs = gr.TabbedInterface(
    [demo_chat_ui, load_ui], ["Chat", "Loading"], title="Dynamite Dogs 🐕"
)
