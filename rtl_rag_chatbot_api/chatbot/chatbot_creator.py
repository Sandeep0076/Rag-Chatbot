import logging
from typing import List

import openai

from rtl_rag_chatbot_api.common.base_handler import BaseRAGHandler


class AzureChatbot(BaseRAGHandler):
    """Handles interactions with Azure OpenAI models for RAG applications."""

    def __init__(
        self,
        configs,
        gcs_handler,
        model_choice: str,
        file_id: str = None,
        file_ids: List[str] = None,
        all_file_infos: dict = None,  # Passed from app.py, contains file_info for each file_id
        collection_name_prefix: str = "rag_collection_",
        user_id: str = None,
    ):
        super().__init__(configs, gcs_handler)
        self.configs = configs
        self.model_choice = model_choice
        self.user_id = user_id
        self.all_file_infos = all_file_infos if all_file_infos else {}
        self._collection_name_prefix = collection_name_prefix  # e.g. "RAG_CHATBOT_"
        self.active_file_ids: List[str] = []
        self.is_multi_file = False

        if file_ids and len(file_ids) > 0:
            self.is_multi_file = True
            self.active_file_ids = sorted(
                list(set(file_ids))
            )  # Ensure unique and sorted
            # For BaseRAGHandler compatibility if its methods are ever called directly in multi-mode
            # These singular attributes don't really apply, so set to None or first file for safety.
            self.file_id = None
            self.collection_name = None
            self.embedding_type = (
                None  # In multi-file, embedding type is always 'azure' for this class
            )
            logging.info(
                f"AzureChatbot initialized for multi-file: {self.active_file_ids}"
            )
        elif file_id:
            self.is_multi_file = False
            self.active_file_ids = [file_id]
            self.file_id = file_id  # For BaseRAGHandler compatibility
            # collection_name for single file is typically just prefix + file_id
            self.collection_name = f"{self._collection_name_prefix}{self.file_id}"
            self.embedding_type = "azure"  # Default/assumed for AzureChatbot
            logging.info(f"AzureChatbot initialized for single-file: {self.file_id}")
        else:
            raise ValueError(
                "AzureChatbot requires either file_id or file_ids for initialization."
            )

        self.model_config = self.configs.azure_llm.models.get(model_choice)
        if not self.model_config:
            raise ValueError(f"Configuration for model {model_choice} not found")

        self._initialize_azure_clients()

    def _initialize_azure_clients(self):
        """Initialize Azure OpenAI and embedding clients."""
        self.llm_client = openai.AzureOpenAI(
            api_key=self.model_config.api_key,
            azure_endpoint=self.model_config.endpoint,
            api_version=self.model_config.api_version,
        )
        self.embedding_client = openai.AzureOpenAI(
            api_key=self.configs.azure_embedding.azure_embedding_api_key,
            azure_endpoint=self.configs.azure_embedding.azure_embedding_endpoint,
            api_version=self.configs.azure_embedding.azure_embedding_api_version,
        )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using Azure's embedding model."""
        try:
            batch_embeddings = []
            for (
                text_item
            ) in texts:  # Ensure we handle if a list of lists is accidentally passed
                current_text = (
                    text_item if isinstance(text_item, str) else " ".join(text_item)
                )
                if not current_text.strip():  # Handle empty strings
                    # OpenAI API errors on empty strings, return zero vector or skip
                    # For simplicity, let's assume a zero vector of the expected dimension (e.g., 1536 for ada-002)
                    # This part might need adjustment based on actual embedding dimension
                    logging.warning(
                        "Empty string encountered in get_embeddings, returning zero vector."
                    )
                    batch_embeddings.append([0.0] * 1536)  # Assuming ada-002 dimension
                    continue
                response = self.embedding_client.embeddings.create(
                    model=self.configs.azure_embedding.azure_embedding_deployment,
                    input=current_text,
                )
                batch_embeddings.append(response.data[0].embedding)
            return batch_embeddings
        except Exception as e:
            logging.error(f"Error getting Azure embeddings: {str(e)}")
            # It might be useful to log the problematic texts if possible, carefully
            # logging.error(f"Problematic texts (first 100 chars): {[t[:100] for t in texts]}")
            raise

    def get_answer(self, query: str) -> str:
        try:
            logging.info(
                f"AzureChatbot ({self.model_choice}) get_answer for file(s): {self.active_file_ids}"
            )
            all_relevant_docs = []
            embedding_model_for_rag = "azure"  # AzureChatbot uses Azure embeddings

            for f_id in self.active_file_ids:
                # Construct collection name specific to this file_id
                current_collection_name = f"{self._collection_name_prefix}{f_id}"
                logging.info(
                    f"Querying ChromaDB for file: {f_id}, collection: {current_collection_name}"
                )
                try:
                    # Use the chroma_manager from BaseRAGHandler
                    # We need to ensure query_chroma can be called with a specific collection
                    # if it's not using self.collection_name. For now, let's assume
                    # direct ChromaManager usage for flexibility here.
                    query_embedding = self.get_embeddings([query])[0]
                    chroma_collection = self.chroma_manager.get_collection(
                        file_id=f_id,  # Used for path generation by chroma_manager
                        embedding_type=embedding_model_for_rag,  # 'azure' or 'google'
                        collection_name=current_collection_name,
                        user_id=self.user_id,  # Pass for potential session tracking, not filtering
                        is_embedding=False,  # We are querying
                    )
                    results = chroma_collection.query(
                        query_embeddings=[query_embedding], n_results=3
                    )  # n_results can be configured
                    docs_from_file = (
                        results["documents"][0] if results["documents"] else []
                    )

                    if self.is_multi_file:
                        all_relevant_docs.extend(
                            [f"[Source: {f_id}] {doc}" for doc in docs_from_file]
                        )
                    else:
                        all_relevant_docs.extend(docs_from_file)
                    logging.info(f"Retrieved {len(docs_from_file)} docs from {f_id}")
                except Exception as e:
                    logging.error(f"Error querying ChromaDB for file {f_id}: {str(e)}")
                    # Optionally, continue to next file or raise error
                    # For now, we log and continue, so chat can proceed with partial context if one DB fails

            if not all_relevant_docs:
                logging.warning(
                    f"No relevant documents found for query: '{query}' in files: {self.active_file_ids}"
                )
                # Fallback: use a generic response or attempt non-RAG if desired.
                # For now, we'll proceed, and the LLM will answer based on its general knowledge if context is empty.

            context_str = "\n".join(all_relevant_docs)

            system_message = self.configs.chatbot.system_prompt_rag_llm
            messages = [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": f"Context:\n{context_str}\n\nQuestion: {query}",
                },
            ]
            max_response_tokens = (
                self.model_config.max_tokens
                if hasattr(self.model_config, "max_tokens")
                and self.model_config.max_tokens is not None
                else 3000
            )  # Use model specific or default

            logging.info(
                f"Sending to LLM. Deployment: {self.model_config.deployment}. Files: {self.active_file_ids}"
            )
            # logging.debug(f"Messages for LLM: {messages}") # Be careful logging full context

            if "o3-mini" in self.model_config.deployment.lower():
                logging.info("Using o3 mini specific parameters for LLM call")
                response = self.llm_client.chat.completions.create(
                    model=self.model_config.deployment,
                    messages=messages,
                    # max_completion_tokens=50000, # This was likely a typo, should be max_tokens
                    max_tokens=max_response_tokens,
                    stop=None,
                    stream=False,
                )
            else:
                response = self.llm_client.chat.completions.create(
                    model=self.model_config.deployment,
                    messages=messages,
                    temperature=self.configs.llm_hyperparams.temperature,  # Use configured temperature
                    max_tokens=max_response_tokens,
                )

            final_answer = response.choices[0].message.content
            logging.info(f"Received answer from LLM for files {self.active_file_ids}")
            return final_answer

        except Exception as e:
            logging.error(
                f"Error in get_answer for files {self.active_file_ids}: {str(e)}",
                exc_info=True,
            )
            # Provide a more generic error to the user via the API
            raise Exception(
                f"Failed to get answer due to an internal error. Details: {str(e)}"
            )

    def get_n_nearest_neighbours(self, query: str, n_neighbours: int = 3) -> List[str]:
        """Get nearest neighbors for a query."""
        return self.query_chroma(query, self.file_id, n_results=n_neighbours)


def get_azure_non_rag_response(
    configs, query: str, model_choice: str = "gpt_4o_mini"
) -> str:
    """
    Retrieves a response from Azure OpenAI without using Retrieval-Augmented Generation (RAG).

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
        logging.info(f"Using Azure OpenAI model: {model_choice}")

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

        # Check if the model is o3 mini which requires different parameters
        logging.info(
            f"Non-RAG model deployment name: {configs.azure_llm.models[model_choice].deployment}"
        )
        # Check for o3-mini in the deployment name
        if "o3-mini" in configs.azure_llm.models[model_choice].deployment.lower():
            logging.info("Using o3 mini specific parameters for non-RAG response")
            response = llm_client.chat.completions.create(
                model=configs.azure_llm.models[model_choice].deployment,
                messages=messages,
                max_completion_tokens=configs.llm_hyperparams.max_tokens,
                stop=configs.llm_hyperparams.stop,
                stream=False,
            )
        else:
            response = llm_client.chat.completions.create(
                model=configs.azure_llm.models[model_choice].deployment,
                messages=messages,
                temperature=configs.llm_hyperparams.temperature,
                max_tokens=configs.llm_hyperparams.max_tokens,
                top_p=configs.llm_hyperparams.top_p,
                frequency_penalty=configs.llm_hyperparams.frequency_penalty,
                presence_penalty=configs.llm_hyperparams.presence_penalty,
                stop=configs.llm_hyperparams.stop,
            )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logging.error(f"Error in get_azure_non_rag_response: {str(e)}")
        raise
