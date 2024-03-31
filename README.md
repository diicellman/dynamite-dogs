# 🐶 Dynamite-Dogs

Dynamite-Dogs is a groundbreaking application designed for developers and technology enthusiasts. This innovative tool transforms your GitHub repository into a conversational agent, allowing for interactive, chat-based exploration of your codebase. Additionally, Dynamite-Dogs features a unique capability to generate engaging social media posts to share your repository across platforms like LinkedIn and Twitter, enhancing visibility and outreach.

**What sets Dynamite-Dogs apart is its cost-effectiveness.** By leveraging advanced prompt engineering techniques and few-shot learning examples, we've built a solution that is not only powerful but also the most affordable on the market. Utilizing the Anthropic Claude 3 Haiku model, Dynamite-Dogs offers responses at approximately $0.01 each, with an impressive 200k tokens context window, ensuring no compromise on quality.

## Features 🐾

- 🦴 **Dynamic Documentation:** Engage with your repository through a chat interface to extract information, clarify doubts, and navigate your codebase interactively.
- 🐕 **Social Media Post Generation:** Easily create ready-to-share posts about your repository, tailor-made for various social media platforms.
- 🐾 **User-Friendly Interface:** Enjoy a seamless experience with a straightforward setup and intuitive navigation.
- 🐩 **Customizable Experience:** Adjust the prompts for documentation and social media posts to better match the specifics of your project.

## User Interface 🎨

Dynamite-Dogs boasts a comprehensive and easy-to-navigate user interface, divided into three main components:

- **Main Interface:** Start here to choose between chatting with your repository or generating a social media post.
- **Repository Loader:** Input the GitHub repository URL you're interested in for processing and analysis.
- **Chat Interface:** Interact dynamically with your repository, asking questions and receiving information as if you're talking to a fellow developer.
- **Social Media Post Generator:** Seamlessly generate customized posts for social media, choosing the platform and content focus with ease.

## Installation

### Prerequisites

- Docker and Docker-Compose
- Git (optional, for cloning the repository)
- Ollama,  follow the [readme](https://github.com/ollama/ollama) to set up and run a local Ollama instance.
- Langfuse, follow this [guide](https://langfuse.com/docs/get-started) to set up Langfuse. 

### Clone the Repository

First, clone the repository to your local machine (skip this step if you have the project files already).

```bash
git clone https://github.com/diicellman/dynamite-dogs.git
cd dynamite-dogs
```
### Getting Started with Local Development

First, setup the environment:

```bash
poetry config virtualenvs.in-project true
poetry install
poetry shell
```
Specify your environment variables in an .env file in root directory.
Example .env file:
```yml
ANTHROPIC_API_KEY = <your_api_key>
ENVIRONMENT= <your_environment_value>
INSTRUMENT_LLAMA= <true or false>
OLLAMA_BASE_URL= <your_ollama_instance_endpoint>
LANGFUSE_PUBLIC_KEY = <your_pb_key>
LANGFUSE_SECRET_KEY = <your_sk_key>
LANGFUSE_HOST = <your_host>
GITHUB_TOKEN = <your_github_token>
```

Second, run this command to pull embeddings model
```bash
ollama pull mxbai-embed-large
```

Then run this command to start the FastAPI server:
```bash
python main.py
```

1. Backend docs can be viewed using the [OpenAPI](http://0.0.0.0:8000/docs).
2. Frontend can be viewed using [Gradio](http://0.0.0.0:8000/gradio)

## Contributing:
Your contributions are what make the open-source community an incredible platform for learning, inspiring, and creating. Every contribution is deeply appreciated. If you have suggestions to improve Dynamite-Dogs, please fork the repo and submit a pull request, or simply open an issue with the tag "enhancement".

## License:
Distributed under the MIT License. For more information, refer to the LICENSE file.