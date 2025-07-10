import concurrent.futures
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
        chroma_manager=None,  # Accept an optional ChromaDBManager instance
    ):
        # Pass the existing chroma_manager to the parent, or let it create one
        super().__init__(configs, gcs_handler, chroma_manager)
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

    def _query_single_file(self, f_id: str, query_embedding):
        """Retrieve top docs from a single file collection (helper for threading)."""
        try:
            current_collection_name = f"{self._collection_name_prefix}{f_id}"
            logging.info(
                f"[Thread] Querying ChromaDB for file: {f_id}, collection: {current_collection_name}"
            )
            chroma_collection = self.chroma_manager.get_collection(
                file_id=f_id,
                embedding_type="azure",
                collection_name=current_collection_name,
                user_id=self.user_id,
                is_embedding=False,
            )
            results = chroma_collection.query(
                query_embeddings=[query_embedding], n_results=3
            )
            docs_from_file = results["documents"][0] if results["documents"] else []
            if self.is_multi_file:
                original_filename = self.all_file_infos.get(f_id, {}).get(
                    "original_filename", f_id
                )
                return [
                    f"[Source: {original_filename}] {doc}" for doc in docs_from_file
                ]
            return docs_from_file
        except Exception as e:
            logging.error(f"Error querying ChromaDB for file {f_id}: {str(e)}")
            return []

    def get_answer(self, query: str) -> str:
        try:
            logging.info(
                f"AzureChatbot ({self.model_choice}) get_answer for file(s): {self.active_file_ids}"
            )
            all_relevant_docs = []
            # embedding_model_for_rag = "azure"  # AzureChatbot uses Azure embeddings

            # Compute the embedding **once** for the whole request to avoid
            # redundant Azure OpenAI /embeddings calls when querying multiple
            # files. This removes N-1 identical HTTP requests and speeds up
            # multi-file chat dramatically.
            query_embedding = self.get_embeddings([query])[0]

            # Parallelise Chroma queries for multiple files
            if self.is_multi_file and len(self.active_file_ids) > 1:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(4, len(self.active_file_ids))
                ) as executor:
                    futures = [
                        executor.submit(self._query_single_file, f_id, query_embedding)
                        for f_id in self.active_file_ids
                    ]
                    for fut in concurrent.futures.as_completed(futures):
                        all_relevant_docs.extend(fut.result())
            else:
                # Single file or only one file -> sequential
                for f_id in self.active_file_ids:
                    all_relevant_docs.extend(
                        self._query_single_file(f_id, query_embedding)
                    )

            if not all_relevant_docs:
                logging.warning(
                    f"No relevant documents found for query: '{query}' in files: {self.active_file_ids}"
                )
                # Fallback: use a generic response or attempt non-RAG if desired.
                # For now, we'll proceed, and the LLM will answer based on its general knowledge if context is empty.

            # Prepare additional context about available files for both single and multi-file mode
            files_context = ""
            if self.all_file_infos:
                if self.is_multi_file:
                    # For multi-file, show list with filenames and URLs if available
                    file_details = []
                    for file_id in self.active_file_ids:
                        if file_id in self.all_file_infos:
                            file_info = self.all_file_infos.get(file_id, {})
                            original_filename = file_info.get(
                                "original_filename", f"Unknown filename (ID: {file_id})"
                            )

                            # For URL files, include the actual URL in the context
                            if "url" in file_info:
                                url = file_info.get("url")
                                file_details.append(
                                    f"- {original_filename} (URL: {url})"
                                )
                            else:
                                file_details.append(f"- {original_filename}")

                    if file_details:
                        files_context = (
                            "Available documents:\n" + "\n".join(file_details) + "\n\n"
                        )
                        logging.info(
                            f"Added multi-file context with {len(file_details)} files (including URLs where applicable)"
                        )
                else:
                    # For single file, also build a clean context string
                    file_id = self.active_file_ids[0]
                    if file_id in self.all_file_infos:
                        file_info = self.all_file_infos.get(file_id, {})
                        original_filename = file_info.get(
                            "original_filename", f"Unknown filename (ID: {file_id})"
                        )

                        # Start with the filename
                        file_context_detail = f"- {original_filename}"

                        # If it's a URL, add it
                        if "url" in file_info:
                            url = file_info.get("url")
                            file_context_detail += f" (URL: {url})"

                        files_context = (
                            "File Information:\n" + file_context_detail + "\n\n"
                        )
                        logging.info(
                            f"Added clean file context for single file: {file_id}"
                        )

            context_str = "\n".join(all_relevant_docs)

            system_message = self.configs.chatbot.system_prompt_rag_llm
            messages = [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": f"{files_context}Context:\n{context_str}\n\nQuestion: {query}",
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

            # Check for any o3 or o4 models which have different parameter requirements
            deployment_lower = self.model_config.deployment.lower()
            if "o4" in deployment_lower:
                # O4 models: Minimal parameters only
                logging.info(
                    f"Using O4-specific parameters for model: {self.model_config.deployment}"
                )
                response = self.llm_client.chat.completions.create(
                    model=self.model_config.deployment,
                    messages=messages,
                    max_completion_tokens=max_response_tokens,
                )
            elif "o3" in deployment_lower:
                # O3 models: Use minimal parameters for simplicity
                logging.info(
                    f"Using O3-specific parameters for model: {self.model_config.deployment}"
                )
                response = self.llm_client.chat.completions.create(
                    model=self.model_config.deployment,
                    messages=messages,
                    max_tokens=max_response_tokens,
                )
            else:
                # Regular OpenAI models
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
    configs, query: str, model_choice: str = "gpt_4o_mini", max_tokens: int = None
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

        # Use provided max_tokens or fall back to config default
        effective_max_tokens = (
            max_tokens if max_tokens is not None else configs.llm_hyperparams.max_tokens
        )

        # Check if the model is any o3 or o4 variant which requires different parameters
        logging.info(
            f"Non-RAG model deployment name: {configs.azure_llm.models[model_choice].deployment}"
        )
        deployment_lower = configs.azure_llm.models[model_choice].deployment.lower()

        if "o4" in deployment_lower:
            # O4 models: Minimal parameters only
            logging.info(
                f"Using O4-specific parameters for non-RAG response with model: "
                f"{configs.azure_llm.models[model_choice].deployment}"
            )
            response = llm_client.chat.completions.create(
                model=configs.azure_llm.models[model_choice].deployment,
                messages=messages,
                max_completion_tokens=effective_max_tokens,
            )
        elif "o3" in deployment_lower:
            # O3 models: Use minimal parameters for simplicity
            logging.info(
                f"Using O3-specific parameters for non-RAG response with model: "
                f"{configs.azure_llm.models[model_choice].deployment}"
            )
            response = llm_client.chat.completions.create(
                model=configs.azure_llm.models[model_choice].deployment,
                messages=messages,
                max_tokens=effective_max_tokens,
            )
        else:
            # Regular OpenAI models
            response = llm_client.chat.completions.create(
                model=configs.azure_llm.models[model_choice].deployment,
                messages=messages,
                temperature=configs.llm_hyperparams.temperature,
                max_tokens=effective_max_tokens,
                top_p=configs.llm_hyperparams.top_p,
                frequency_penalty=configs.llm_hyperparams.frequency_penalty,
                presence_penalty=configs.llm_hyperparams.presence_penalty,
                stop=configs.llm_hyperparams.stop,
            )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logging.error(f"Error in get_azure_non_rag_response: {str(e)}")
        raise
