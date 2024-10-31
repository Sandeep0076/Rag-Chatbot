import logging
from typing import List

import openai

from rtl_rag_chatbot_api.common.base_handler import BaseRAGHandler


class AzureChatbot(BaseRAGHandler):
    """Handles interactions with Azure OpenAI models for RAG applications."""

    def __init__(self, configs, gcs_handler):
        super().__init__(configs, gcs_handler)
        self.configs = configs
        self.model_config = None
        self.llm_client = None
        self.embedding_client = None

    def initialize(
        self,
        model_choice: str,
        file_id: str = None,
        embedding_type: str = None,
        collection_name: str = None,
    ):
        """Initialize the Azure model with specific configurations."""
        self.model_choice = model_choice
        self.file_id = file_id
        self.embedding_type = embedding_type
        self.collection_name = collection_name or f"rag_collection_{file_id}"

        # Get model configuration based on model_choice
        self.model_config = self.configs.azure_llm.models.get(model_choice)
        if not self.model_config:
            raise ValueError(f"Configuration for model {model_choice} not found")

        self.initialize_azure_clients()

    def initialize_azure_clients(self):
        """Initialize Azure OpenAI and embedding clients."""
        # Initialize Azure OpenAI client for chat completions
        self.llm_client = openai.AzureOpenAI(
            api_key=self.model_config.api_key,
            azure_endpoint=self.model_config.endpoint,
            api_version=self.model_config.api_version,
        )

        # Initialize Azure Text Analytics client for embeddings
        self.embedding_client = openai.AzureOpenAI(
            api_key=self.configs.azure_embedding.azure_embedding_api_key,
            azure_endpoint=self.configs.azure_embedding.azure_embedding_endpoint,
            api_version=self.configs.azure_embedding.azure_embedding_api_version,
        )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using Azure's embedding model."""
        try:
            batch_embeddings = []
            for text in texts:
                response = self.embedding_client.embeddings.create(
                    model=self.configs.azure_embedding.azure_embedding_deployment,
                    input=text,
                )
                batch_embeddings.append(response.data[0].embedding)
            return batch_embeddings
        except Exception as e:
            logging.error(f"Error getting Azure embeddings: {str(e)}")
            raise

    def get_answer(self, query: str) -> str:
        """Generate an answer to a query using relevant context."""
        try:
            relevant_chunks = self.query_chroma(query, self.file_id, n_results=3)
            if not relevant_chunks:
                return (
                    "I couldn't find any relevant information to answer your question."
                )

            context = "\n".join(relevant_chunks)
            messages = [
                {
                    "role": "system",
                    "content": self.configs.chatbot.system_prompt_rag_llm,
                },
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"},
            ]

            response = self.llm_client.chat.completions.create(
                model=self.model_config.deployment,
                messages=messages,
                temperature=self.configs.llm_hyperparams.temperature,
                max_tokens=self.configs.llm_hyperparams.max_tokens,
                top_p=self.configs.llm_hyperparams.top_p,
                frequency_penalty=self.configs.llm_hyperparams.frequency_penalty,
                presence_penalty=self.configs.llm_hyperparams.presence_penalty,
                stop=self.configs.llm_hyperparams.stop,
            )

            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in get_answer: {str(e)}")
            raise

    def get_n_nearest_neighbours(self, query: str, n_neighbours: int = 3) -> List[str]:
        """Get nearest neighbors for a query."""
        return self.query_chroma(query, self.file_id, n_results=n_neighbours)


def get_azure_non_rag_response(
    configs, query: str, model_choice: str = "gpt_4o_mini"
) -> str:
    """
    Retrieves a response from Azure OpenAI without using Retrieval-Augmented Generation (RAG).

    This function initializes a Chatbot instance with the given configurations and model choice,
    then uses it to generate a response to the provided query.

    Args:
        configs (Config): Configuration object containing necessary settings for the Chatbot.
        query (str): The user's input query or prompt.
        model_choice (str, optional): The specific Azure OpenAI model to use. Defaults to "gpt_4o_mini".

    Returns:
        str: The generated response from the Azure OpenAI model.

    Raises:
        Exception: If there's an error in getting the response, with the error message included.
    """
    try:
        # Initialize Azure OpenAI client for direct completion
        llm_client = openai.AzureOpenAI(
            api_key=configs.azure_llm.models[model_choice].api_key,
            azure_endpoint=configs.azure_llm.models[model_choice].endpoint,
            api_version=configs.azure_llm.models[model_choice].api_version,
        )

        # Create completion with system prompt and user query
        messages = [
            {
                "role": "system",
                "content": configs.chatbot.system_prompt_plain_llm,
            },
            {"role": "user", "content": query},
        ]

        completion = llm_client.chat.completions.create(
            model=configs.azure_llm.models[model_choice].deployment,
            messages=messages,
            temperature=configs.llm_hyperparams.temperature,
            max_tokens=configs.llm_hyperparams.max_tokens,
            top_p=configs.llm_hyperparams.top_p,
            frequency_penalty=configs.llm_hyperparams.frequency_penalty,
            presence_penalty=configs.llm_hyperparams.presence_penalty,
            stop=configs.llm_hyperparams.stop,
        )

        return completion.choices[0].message.content

    except Exception as e:
        logging.error(f"Error in get_azure_non_rag_response: {str(e)}")
        raise Exception(f"Failed to get response: {str(e)}")
