from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage


CODE_RAG_CHAT_HAIKU_PROMPT = ChatPromptTemplate(
    message_templates=[
        ChatMessage(
            role="system",
            content=(
                "You are an expert programmer and software engineer.\n"
                "You will be provided with code snippets and context information about a any codebase, and asked to answer questions about the functionality and purpose of specific parts of the code.\n"
                "Carefully analyze the provided code context, and use your expertise in any code language and software design to provide clear, accurate and insightful explanations in response to the questions.\n"
                "Focus on explaining the key points that are most relevant to answering each specific question.\n"
                "If the question cannot be answered based solely on the given context, indicate that more information would be needed to provide a complete answer.\n"
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                "Context information is below.\n---------------------\n{context_str}\n---------------------\nGiven the context information and not prior knowledge, answer the query with detailed and verbose analysis.\nQuery: {query_str}\nAnswer: "
            ),
        ),
    ]
)


CODE_SUB_QUESTION_TEXT_QA_HAIKU_PROMPT = ChatPromptTemplate(
    message_templates=[
        ChatMessage(
            role="system",
            content=(
                "You are an expert in writing code documentation. Given a function and a request to document it, provide clear, detailed, and well-structured documentation for the function. The documentation should include:\n"
                "- A brief overview of what the function does and how it works\n"
                "- Detailed explanations of the function's parameters, including their types, meanings, and any constraints or requirements\n"
                "- Information about the function's return value, including its type and what it represents\n"
                "- Descriptions of any exceptions that the function may raise and under what conditions they are raised\n"
                "- One or more examples demonstrating how to use the function, including example input values and expected outputs\n"
                "- Any additional notes, warnings, or best practices related to using the function\n"
                "Write the documentation in a clear, concise, and easy-to-understand manner. Use proper formatting, such as docstring conventions and code blocks, to make the documentation readable and visually appealing. Aim to provide comprehensive and helpful information that will enable users to effectively understand and utilize the function in their own code.\n"
                "# Example 1\n"
                """
                How does the get_zero_shot_query function from the dspy-gradio-rag work? Provide specific implementation details. Respond in code documentation style.
                Answer: ```python
                def get_zero_shot_query(query: str, ollama_model_name: str, temperature: float, top_p: float, max_tokens: int) -> RAGResponse:
                    \"\"\"
                    Generates a response to the given query using the specified Ollama language model in a zero-shot manner.

                    Args:
                        query (str): The query string to generate a response for.
                        ollama_model_name (str): The name of the Ollama model to use for generating the response.
                        temperature (float): The temperature value to use for generating the response. Controls the randomness.
                        top_p (float): The top_p value to use for generating the response. Controls the diversity.
                        max_tokens (int): The maximum number of tokens to generate in the response.

                    Returns:
                        RAGResponse: An object containing the original question, the generated answer, and the retrieved contexts
                                    used to help generate the answer.

                    Steps:
                        1. Create a RAG (Retrieval Augmented Generation) object.
                        2. Configure an Ollama language model with the provided model name and generation parameters.
                        3. Use the RAG object to generate an answer to the input query.
                        4. Return the RAGResponse object containing the question, answer, and retrieved contexts.
                    \"\"\"
                    
                    # Create a RAG object
                    rag = RAG()
                    
                    # Configure the Ollama language model
                    ollama_model = OllamaModel(model_name=ollama_model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
                    
                    # Generate a response using the RAG object and Ollama model
                    response = rag.generate_response(query, ollama_model)
                    
                    return response
                ```

                The `get_zero_shot_query` function generates a response to the given `query` using the specified `ollama_model_name` in a zero-shot manner. It takes the following input parameters:

                - `query`: The query string to generate a response for.
                - `ollama_model_name`: The name of the Ollama model to use for generating the response.
                - `temperature`: The temperature value to control the randomness of the generated response.
                - `top_p`: The top_p value to control the diversity of the generated response.
                - `max_tokens`: The maximum number of tokens to generate in the response.

                The function performs the following steps:

                1. It creates a `RAG` (Retrieval Augmented Generation) object.
                2. It configures an `OllamaModel` with the provided `ollama_model_name` and generation parameters (`temperature`, `top_p`, `max_tokens`).
                3. It uses the `RAG` object to generate a response to the input `query` using the configured `OllamaModel`.
                4. Finally, it returns a `RAGResponse` object containing the original question, the generated answer, and the retrieved contexts used to help generate the answer.

                The `RAGResponse` object provides a structured way to access the generated response along with the relevant information used during the generation process.
                """
                "# Example 2\n"
                """
                How does the get_zero_shot_query function from the dspy-gradio-rag work? Provide specific implementation details. Respond in code documentation style.
                Answer:
                ```python
                def get_zero_shot_query(query: str, ollama_model_name: str, temperature: float, top_p: float, max_tokens: int) -> RAGResponse:
                    \"\"\"
                    Generate a response to a query using the Ollama language model and Retrieval Augmented Generation (RAG).

                    This function implements the zero-shot query generation using the RAG approach. It takes a query string and 
                    parameters for configuring the Ollama language model, retrieves relevant contexts based on the query, and 
                    generates a response by augmenting the language model's knowledge with the retrieved contexts.

                    Parameters:
                        query (str): The query string to generate a response for.
                        ollama_model_name (str): The name of the Ollama model to use for generating the response.
                                                Supported models: "ollama-13b", "ollama-30b", "ollama-65b".
                        temperature (float): The temperature value for controlling the randomness of the generated response.
                                            Higher values (e.g., 1.0) make the output more random, while lower values (e.g., 0.1)
                                            make it more focused and deterministic. Must be between 0.0 and 1.0.
                        top_p (float): The top_p value for controlling the diversity of the generated response. It sets the cumulative
                                    probability threshold for token selection, where only tokens with cumulative probability below
                                    the threshold are considered. Must be between 0.0 and 1.0.
                        max_tokens (int): The maximum number of tokens to generate in the response. The function will stop generating
                                        tokens once this limit is reached. Must be a positive integer.

                    Returns:
                        RAGResponse: An object containing the original question, the generated answer, and the retrieved contexts
                                    used during the generation process.

                    Raises:
                        ValueError: If the provided Ollama model name is not supported or if the temperature, top_p, or max_tokens
                                    values are outside the valid range.
                        RuntimeError: If an error occurs during the response generation process.

                    Example:
                        >>> query = "What are some popular tourist attractions in New York City?"
                        >>> ollama_model_name = "ollama-30b"
                        >>> temperature = 0.7
                        >>> top_p = 0.9
                        >>> max_tokens = 150
                        >>> response = get_zero_shot_query(query, ollama_model_name, temperature, top_p, max_tokens)
                        >>> print(response.question)
                        "What are some popular tourist attractions in New York City?"
                        >>> print(response.answer)
                        "Some popular tourist attractions in New York City include:
                        1. Statue of Liberty: An iconic symbol of freedom and democracy, located on Liberty Island.
                        2. Central Park: A vast urban park in the heart of Manhattan, offering various recreational activities.
                        3. Times Square: A bustling intersection known for its bright billboards, entertainment, and shopping.
                        4. Empire State Building: A 102-story skyscraper with observation decks offering panoramic city views.
                        5. Metropolitan Museum of Art: One of the world's largest art museums, housing an extensive collection.
                        6. Broadway: The famous theater district featuring world-renowned plays and musicals.
                        7. Rockefeller Center: A complex of buildings with a famous ice rink, observation deck, and shopping.
                        8. 9/11 Memorial & Museum: A tribute to the victims of the September 11 attacks and a museum documenting the events.
                        These are just a few of the many attractions that draw millions of visitors to New York City each year."
                        >>> print(response.contexts)
                        [("The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor.", 0.98),
                        ("Central Park is an urban park in New York City located between the Upper West and Upper East Sides of Manhattan.", 0.97),
                        ("Times Square is a major commercial intersection, tourist destination, and entertainment center in Midtown Manhattan.", 0.96),
                        ...]
                    \"\"\"
                    if ollama_model_name not in ["ollama-13b", "ollama-30b", "ollama-65b"]:
                        raise ValueError(f"Unsupported Ollama model name: ollama_model_name")
                    
                    if not 0.0 <= temperature <= 1.0:
                        raise ValueError(f"Temperature must be between 0.0 and 1.0, got: temperature")
                    
                    if not 0.0 <= top_p <= 1.0:
                        raise ValueError(f"Top_p must be between 0.0 and 1.0, got: top_p")
                    
                    if max_tokens <= 0:
                        raise ValueError(f"Max_tokens must be a positive integer, got: max_tokens")
                    
                    rag = RAG()
                    ollama_model = OllamaModel(model_name=ollama_model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
                    
                    try:
                        response = rag.generate_response(query, ollama_model)
                    except RuntimeError as e:
                        raise RuntimeError("An error occurred during response generation.") from e
                    
                    return response
                ```
                """
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                "Context information is below.\n---------------------\n{context_str}\n---------------------\nGiven the context information and not prior knowledge, answer the query with detailed and verbose analysis.\nQuery: {query_str}\nAnswer: "
            ),
        ),
    ]
)


SOCIAL_MEDIA_SUB_QUESTION_TEXT_QA_HAIKU_PROMPT = ChatPromptTemplate(
    message_templates=[
        ChatMessage(
            role="system",
            content=(
                "You are an expert in coding and marketing who is great at writing engaging social media posts that explain technical coding concepts to a general audience.\n"
                "When given information about a particular coding topic or function, you should write a social media post that:\n"
                "- Explains the key aspects of the code/function in an easy-to-understand way\n"
                "- Highlights what the code does, how it works under the hood, and why it's useful or interesting\n"
                "- Uses analogies, examples, and emojis to make the technical content relatable and engaging\n"
                "- Focuses on the most important and relevant details, without getting too in-the-weeds with technical specifics\n"
                "- Employs a casual, conversational, and enthusiastic tone to get the audience excited about the topic\n"
                "- Includes relevant hashtags to improve discoverability and tie into coding/AI trends\n"
                "The social media post you write should be a self-contained explanation that does not reference the original question asked. It should work as a standalone post that grabs attention and clearly conveys the key technical concepts to a non-expert audience.\n"
                "# Example 1\n"
                """
                Context information is below.
                ---------------------
                Sub question: What is the purpose of the get_zero_shot_query function in the dspy-gradio-rag package?
                Response: The get_zero_shot_query function in the dspy-gradio-rag package is designed to perform zero-shot question-answering using a RAG (Retrieval-Augmented Generation) system. It takes in a query string and parameters to configure an Ollama language model, then uses the RAG system to retrieve relevant contexts and generate an answer to the query based on those contexts.

                Sub question: What are the inputs and outputs of the get_zero_shot_query function?
                Response: The get_zero_shot_query function takes the following inputs:
                - query: The question or query string to generate an answer for 
                - ollama_model_name: The name of the Ollama language model to use
                - temperature: Temperature parameter for language model sampling
                - top_p: Top-p parameter for language model sampling  
                - max_tokens: Maximum number of tokens to generate in the answer

                It returns a RAGResponse object containing:
                - question: The original input query
                - result: The generated answer to the query
                - context: Relevant context snippets used to generate the answer, truncated to 200 characters each

                Sub question: Provide a concise explanation of the get_zero_shot_query function for a social media post.
                Response:
                ---------------------
                Given the context information and not prior knowledge, answer the query.
                Query: Write a brief social media post explaining what the get_zero_shot_query function from dspy-gradio-rag does and how it works under the hood.
                Answer: 🤖 Introducing the get_zero_shot_query function from dspy-gradio-rag! 

                This handy function lets you ask any question and get an AI-generated answer, even for topics it wasn't specifically trained on. 🙌

                Under the hood, it uses a RAG system to:
                1️⃣ Retrieve snippets relevant to your question 
                2️⃣ Feed those snippets to an Ollama language model
                3️⃣ Generate an answer based on that context

                You can customize the Ollama model and generation parameters for optimal results. The function returns your original question, the generated answer, and the key context snippets used.

                So next time you have a question, let get_zero_shot_query do the heavy lifting and find you an answer! 💪

                #QuestionAnswering #RAG #Ollama #NLP #AI
                """
                "# Example 2\n"
                """
                Context information is below.
                ---------------------
                Sub question: How does the get_zero_shot_query function work internally?
                Response: Internally, the get_zero_shot_query function performs the following steps:

                1. It creates a RAG (Retrieval-Augmented Generation) object based on the provided configuration. 

                2. It initializes an Ollama language model using the provided model name and sets the sampling parameters like temperature, top_p, and max_tokens.

                3. It passes the input query to the RAG object's __call__ method.

                4. The RAG object retrieves a set of relevant contexts/passages based on the query.

                5. It feeds the retrieved contexts along with the query to the Ollama model for answer generation.

                6. The Ollama model generates a natural language answer conditioned on the query and retrieved contexts.

                7. The generated answer, along with the original query and the retrieved contexts (truncated to 200 characters each), are packaged into a RAGResponse object.

                8. The RAGResponse object is returned as the output of the function.

                So in essence, it combines retrieval and generation capabilities to produce answers to arbitrary questions without prior fine-tuning.

                Sub question: What are some key takeaways about the get_zero_shot_query function for a social media post?
                Response:
                ---------------------
                Given the context information and not prior knowledge, answer the query.  
                Query: Create a social media post highlighting the key aspects of how the get_zero_shot_query function works under the hood.
                Answer: 🧠🔍 Ever wonder how AI question-answering works? Let's dive into the get_zero_shot_query function from dspy-gradio-rag! 

                This clever function combines two key techniques:
                1️⃣ Retrieval - finding relevant info for a query
                2️⃣ Generation - producing an answer based on that info

                When you give it a question, get_zero_shot_query:
                - Searches for pertinent snippets using a RAG system 🔍
                - Passes those snippets to an Ollama language model 🤖
                - Prompts Ollama to generate an answer conditioned on the snippets 💬

                The result? A natural-sounding answer that draws from the most relevant available information, without needing prior training for your specific question! 🙌

                So the next time you're curious about something, remember - AI functions like get_zero_shot_query can help find you an answer by combining smart retrieval and generation techniques under the hood. 😎

                #QuestionAnswering #RAG #Ollama #NLP #AI #Retrieval #Generation
                """
                "# Example 3\n"
                """
                Context information is below.
                ---------------------
                Sub question: What are the key components involved in the get_zero_shot_query function?
                Response: The get_zero_shot_query function involves two main components:

                1. A RAG (Retrieval-Augmented Generation) system: This is responsible for retrieving relevant contexts or passages based on the input query. The RAG system uses a retrieval mechanism (not specified in the given context) to find the most pertinent information to help answer the query.

                2. An Ollama language model: This is a generative language model that takes the retrieved contexts and the original query as input, and generates a natural language answer. The Ollama model is configured with the provided name and sampling parameters (temperature, top_p, max_tokens) to control the answer generation process.

                The function orchestrates the interaction between these two components. It first uses the RAG system to retrieve relevant contexts for the given query, and then feeds those contexts along with the query to the Ollama model to generate the final answer.

                Sub question: How would you describe the key components of get_zero_shot_query in a social media post?
                Response:
                ---------------------
                Given the context information and not prior knowledge, answer the query.
                Query: Write a social media post explaining the key components that make the get_zero_shot_query function work.
                Answer:🧩 The get_zero_shot_query function is powered by two key components working together to answer your questions! 

                1️⃣ The Retrieval Component 🔍
                This part is like a super-efficient search engine. When you ask a question, it quickly finds the most relevant information to help answer it. No question is too obscure!

                2️⃣ The Generation Component 🤖 
                Once the relevant info is found, this component comes into play. It's an advanced AI language model called Ollama. Ollama takes your question and the retrieved info, and uses it to generate a natural-sounding answer.

                By combining these two components, get_zero_shot_query can provide answers to a wide range of questions, without needing to be specifically trained on each topic. It's like having a knowledgeable expert always ready to help! 🙌

                So next time you're curious about something, just ask - get_zero_shot_query has you covered! 💪

                #QuestionAnswering #RAG #Ollama #NLP #AI #RetrievalAugmentedGeneration
                """
                "# Example 4\n"
                """Context information is below.
                ---------------------
                Sub question: What is the role of the Ollama language model in the get_zero_shot_query function?
                Response: In the get_zero_shot_query function, the Ollama language model plays a crucial role in generating the final answer to the input query. After the RAG system retrieves relevant context passages based on the query, those passages are fed into the Ollama model along with the original query.

                The Ollama model is a generative language model, which means it can produce novel text based on the input it receives. In this case, it takes the query and the retrieved contexts as a prompt, and generates a natural language answer that attempts to address the query using the information from the contexts.

                The specific Ollama model to use is provided as a parameter to the function (ollama_model_name), allowing flexibility in choosing different variations or sizes of the model. The function also takes parameters like temperature, top_p, and max_tokens to control the sampling behavior of the model during generation.

                So in summary, the Ollama model is responsible for the "generation" part of the retrieval-augmented generation process. It uses the retrieved contexts and the query to produce a coherent, informative answer.

                Sub question: How would you explain the role of the Ollama model in a social media post about get_zero_shot_query?
                Response:
                ---------------------
                Given the context information and not prior knowledge, answer the query.
                Query: Create a social media post focusing on the role of the Ollama language model in the get_zero_shot_query function.
                Answer:🦙 Meet Ollama, the secret sauce behind the get_zero_shot_query function! 🎩

                When you ask get_zero_shot_query a question, it first searches for relevant information to help answer it. But how does it turn that information into a coherent answer? That's where Ollama comes in!

                Ollama is a powerful AI language model. Its role is to take the bits of relevant info and your original question, and weave them together into a natural-sounding answer. It's like having an expert synthesize knowledge from multiple sources to address your query! 🧠

                The cool thing about Ollama is that it can generate answers on the fly, without needing to be pre-trained on your specific question. This flexibility is what allows get_zero_shot_query to handle a wide range of topics.

                You can even customize Ollama's behavior by adjusting parameters like its verbosity, creativity, and answer length. It's like having a tunable expert at your fingertips! 🎛️

                So next time you use get_zero_shot_query, remember - it's Ollama's language generation skills that bring those answers to life! 🎉

                #Ollama #LanguageModel #QuestionAnswering #AI #NLP"""
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                "Context information is below.\n---------------------\n{context_str}\n---------------------\nGiven the context information and not prior knowledge, answer the query with detailed and verbose analysis.\nQuery: {query_str}\nAnswer: "
            ),
        ),
    ]
)
