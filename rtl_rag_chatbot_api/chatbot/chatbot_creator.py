import concurrent.futures
import logging
from typing import List

import openai

from rtl_rag_chatbot_api.common.base_handler import BaseRAGHandler
from rtl_rag_chatbot_api.common.errors import ModelInitializationError


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
            raise ModelInitializationError(
                "AzureChatbot requires either file_id or file_ids for initialization.",
                details={"file_id": file_id, "file_ids": file_ids},
            )

        self.model_config = self.configs.azure_llm.models.get(model_choice)
        if not self.model_config:
            raise ModelInitializationError(
                f"Configuration for model {model_choice} not found",
                details={"model_choice": model_choice},
            )

        self._initialize_azure_clients()

    def _initialize_azure_clients(self):
        """Initialize Azure OpenAI and embedding clients."""
        self.llm_client = openai.AzureOpenAI(
            api_key=self.model_config.api_key,
            azure_endpoint=self.model_config.endpoint,
            api_version=self.model_config.api_version,
        )

        # Initialize both embedding clients
        # Legacy ada-002 client
        self.embedding_client_ada002 = openai.AzureOpenAI(
            api_key=self.configs.azure_embedding.azure_embedding_api_key,
            azure_endpoint=self.configs.azure_embedding.azure_embedding_endpoint,
            api_version=self.configs.azure_embedding.azure_embedding_api_version,
        )

        # New 03 client
        self.embedding_client_03 = openai.AzureOpenAI(
            api_key=self.configs.azure_embedding_3_large.azure_embedding_3_large_api_key,
            azure_endpoint=self.configs.azure_embedding_3_large.azure_embedding_3_large_endpoint,
            api_version=self.configs.azure_embedding_3_large.azure_embedding_3_large_api_version,
        )

        # Default to 03 for new embeddings
        self.embedding_client = self.embedding_client_03

    def _get_embedding_config_for_file(self, file_id: str = None):
        """Get the appropriate embedding client and deployment based on file's embedding_type"""

        # TODO ZL: this has to go to BaseRAGHandler or a common utility, since this class inherits from it
        # and the functionality is same as in gemini_handler.py

        # Default to new embedding for new files
        default_client = self.embedding_client_03
        default_deployment = (
            self.configs.azure_embedding_3_large.azure_embedding_3_large_deployment
        )

        if not file_id:
            return default_client, default_deployment

        import logging

        # Check file_info to determine embedding type
        try:
            # First check if we have it in all_file_infos
            if (
                hasattr(self, "all_file_infos")
                and self.all_file_infos
                and file_id in self.all_file_infos
            ):
                file_info_data = self.all_file_infos[file_id]

                # Use embedding_type from all_file_infos (enriched in chat endpoint)
                embedding_type = file_info_data.get("embedding_type")

                if not embedding_type:
                    # Default to configurable embedding for files without info
                    embedding_type = file_info_data.get(
                        "embedding_type", self.configs.chatbot.default_embedding_type
                    )

                # For tabular files (no embedding_type), explicitly use configurable default embedding
                if file_info_data.get("is_tabular", False):
                    embedding_type = "azure-3-large"
                    logging.info(
                        f"Tabular file {file_id} detected, using text-embedding-3-large"
                    )
            else:
                # Check local file_info.json
                import json
                import os

                local_info_path = os.path.join("./chroma_db", file_id, "file_info.json")
                if os.path.exists(local_info_path):
                    with open(local_info_path, "r") as f:
                        file_info_data = json.load(f)
                        embedding_type = file_info_data.get(
                            "embedding_type",
                            self.configs.chatbot.default_embedding_type,
                        )
                        # For tabular files (no embedding_type), explicitly use configurable default embedding
                        if file_info_data.get("is_tabular", False):
                            embedding_type = self.configs.chatbot.default_embedding_type
                            logging.info(
                                f"Tabular file {file_id} detected, using {embedding_type}"
                            )
                else:
                    # Default to configurable embedding for files without info
                    embedding_type = self.configs.chatbot.default_embedding_type

            # Return appropriate client and deployment based on embedding_type
            logging.info(f"File {file_id} has embedding_type: '{embedding_type}'")

            if embedding_type == "azure":
                logging.info(f"Using legacy ada-002 embeddings for file {file_id}")
                return (
                    self.embedding_client_ada002,
                    self.configs.azure_embedding.azure_embedding_deployment,
                )
            else:  # "azure-3-large" or any other value (including tabular files) defaults to new embedding
                logging.info(
                    f"Using text-embedding-3-large for file {file_id} (embedding_type: '{embedding_type}')"
                )
                return (
                    self.embedding_client_03,
                    self.configs.azure_embedding_3_large.azure_embedding_3_large_deployment,
                )
        except Exception as e:
            logging.warning(
                f"Error getting embedding config for file {file_id}: {e}, using default"
            )
            return default_client, default_deployment

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using Azure's embedding model based on file embedding_type."""
        try:
            # For query-time, check the first active file to determine embedding type
            # In multi-file scenarios, all files should use the same embedding type for compatibility
            file_id_for_config = None
            if hasattr(self, "active_file_ids") and self.active_file_ids:
                file_id_for_config = self.active_file_ids[0]
            elif hasattr(self, "file_id") and self.file_id:
                file_id_for_config = self.file_id

            # Get appropriate client and deployment
            embedding_client, deployment = self._get_embedding_config_for_file(
                file_id_for_config
            )

            batch_embeddings = []
            for (
                text_item
            ) in texts:  # Ensure we handle if a list of lists is accidentally passed
                current_text = (
                    text_item if isinstance(text_item, str) else " ".join(text_item)
                )
                if not current_text.strip():  # Handle empty strings
                    # OpenAI API errors on empty strings, return zero vector or skip
                    # Default dimension for both models (ada-002: 1536, text-embedding-3-large: 1536)
                    logging.warning(
                        "Empty string encountered in get_embeddings, returning zero vector."
                    )
                    dims_by_deployment = {
                        "text-embedding-ada-002": 1536,
                        "text-embedding-3-large": 3072,
                    }
                    zero_dim = dims_by_deployment.get(deployment, 3072)
                    batch_embeddings.append([0.0] * zero_dim)
                    continue
                response = embedding_client.embeddings.create(
                    model=deployment,
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

            # Check for O3, O4, or GPT-5 models which use max_completion_tokens
            deployment_lower = self.model_config.deployment.lower()
            use_max_completion = any(
                term in deployment_lower for term in ["o3", "o4", "gpt-5", "gpt_5"]
            )

            # Resolve optional env-configurable advanced params
            configured_reasoning_effort = (
                self.configs.llm_hyperparams.reasoning_effort
                if hasattr(self.configs, "llm_hyperparams")
                else None
            )
            configured_verbosity = (
                self.configs.llm_hyperparams.verbosity
                if hasattr(self.configs, "llm_hyperparams")
                else None
            )

            # Build params similar to working examples/test-gpt-5.py
            completion_params = {
                "model": self.model_config.deployment,
                "messages": messages,
            }

            if use_max_completion:
                completion_params["max_completion_tokens"] = max_response_tokens
                completion_params[
                    "presence_penalty"
                ] = self.configs.llm_hyperparams.presence_penalty
                # 'reasoning_effort' and 'verbosity' are only supported by GPT-5 family, not by O4/O3
                if any(term in deployment_lower for term in ["gpt-5", "gpt_5"]):
                    completion_params["reasoning_effort"] = (
                        configured_reasoning_effort or "minimal"
                    )
                    completion_params["verbosity"] = configured_verbosity or "medium"
                # Do NOT include 'stop' for GPT-5/O-series models
            else:
                # Regular OpenAI models
                completion_params[
                    "temperature"
                ] = self.configs.llm_hyperparams.temperature
                completion_params["max_tokens"] = max_response_tokens
                completion_params["top_p"] = self.configs.llm_hyperparams.top_p
                completion_params[
                    "frequency_penalty"
                ] = self.configs.llm_hyperparams.frequency_penalty
                completion_params[
                    "presence_penalty"
                ] = self.configs.llm_hyperparams.presence_penalty
                if self.configs.llm_hyperparams.stop is not None:
                    completion_params["stop"] = self.configs.llm_hyperparams.stop

            response = self.llm_client.chat.completions.create(**completion_params)

            final_answer = response.choices[0].message.content
            logging.info(f"Received answer from LLM for files {self.active_file_ids}")
            return final_answer

        except Exception as e:
            logging.error(
                f"Error in get_answer for files {self.active_file_ids}: {str(e)}",
                exc_info=True,
            )
            # Provide a more generic error to the user via the API
            from rtl_rag_chatbot_api.common.errors import BaseAppError, ErrorRegistry

            raise BaseAppError(
                ErrorRegistry.ERROR_LLM_GENERATION_FAILED,
                f"Failed to get answer due to an internal error. Details: {str(e)}",
                details={"file_id": self.file_id},
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
                "content": (
                    "You are a helpful assistant that provides accurate and relevant "
                    "information based on the given data."
                ),
            },
            {"role": "user", "content": query},
        ]

        # Use provided max_tokens or fall back to config default
        effective_max_tokens = (
            max_tokens if max_tokens is not None else configs.llm_hyperparams.max_tokens
        )

        # Check if the model is O3, O4, or GPT-5 variant which requires max_completion_tokens
        logging.info(
            f"Non-RAG model deployment name: {configs.azure_llm.models[model_choice].deployment}"
        )
        deployment_lower = configs.azure_llm.models[model_choice].deployment.lower()
        use_max_completion = any(
            term in deployment_lower for term in ["o3", "o4", "gpt-5", "gpt_5"]
        )

        # Resolve optional env-configurable advanced params
        configured_reasoning_effort = (
            configs.llm_hyperparams.reasoning_effort
            if hasattr(configs, "llm_hyperparams")
            else None
        )
        configured_verbosity = (
            configs.llm_hyperparams.verbosity
            if hasattr(configs, "llm_hyperparams")
            else None
        )

        # Build params similar to working examples/test-gpt-5.py
        completion_params = {
            "model": configs.azure_llm.models[model_choice].deployment,
            "messages": messages,
        }

        if use_max_completion:
            completion_params["max_completion_tokens"] = effective_max_tokens
            completion_params[
                "presence_penalty"
            ] = configs.llm_hyperparams.presence_penalty
            # 'reasoning_effort' and 'verbosity' are only supported by GPT-5 family, not by O4/O3
            if any(term in deployment_lower for term in ["gpt-5", "gpt_5"]):
                completion_params["reasoning_effort"] = (
                    configured_reasoning_effort or "minimal"
                )
                completion_params["verbosity"] = configured_verbosity or "medium"
            # Do NOT include 'stop' for GPT-5/O-series models
        else:
            # Regular models: Use all standard parameters
            completion_params["temperature"] = configs.llm_hyperparams.temperature
            completion_params["max_tokens"] = effective_max_tokens
            completion_params["top_p"] = configs.llm_hyperparams.top_p
            completion_params[
                "frequency_penalty"
            ] = configs.llm_hyperparams.frequency_penalty
            completion_params[
                "presence_penalty"
            ] = configs.llm_hyperparams.presence_penalty
            if configs.llm_hyperparams.stop is not None:
                completion_params["stop"] = configs.llm_hyperparams.stop

        response = llm_client.chat.completions.create(**completion_params)

        return response.choices[0].message.content.strip()

    except Exception as e:
        logging.error(f"Error in get_azure_non_rag_response: {str(e)}")
        raise
