"""Gradio Load ui component."""

import gradio as gr
import re

from app.engine.load import build_index
from app.models.models import GitHubPayload


def parse_github_url(url: str):
    pattern = r"https://github.com/([^/]+)/([^/]+)"
    match = re.search(pattern, url)
    if match:
        owner = match.group(1)
        repo_name = match.group(2)
        return owner, repo_name
    else:
        return None, None


def load_github_repository(github_url: str):
    owner, repo = parse_github_url(github_url)
    github_payload = GitHubPayload(owner=owner, repo=repo)
    try:
        build_index(github_payload=github_payload)
        return f"Successfully created embeddings for {owner}/{repo} in the LanceDB."
        gr.Info("Success!")
    except Exception as e:
        return "Something went wrong."
        gr.Warning("Something went wrong.")


load_ui = gr.Interface(
    fn=load_github_repository,
    inputs=[
        gr.Textbox(
            label="Link to GitHub repository.",
            placeholder="https://github.com/owner/repo_name",
        ),
    ],
    outputs=gr.Textbox(label="Loading status"),
    title="Loading 📥",
    allow_flagging="never",
    submit_btn="Upload to vector store ⏎",
    clear_btn="Clear 🗑️",
)
