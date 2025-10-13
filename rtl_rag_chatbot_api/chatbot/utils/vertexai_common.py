import logging
import time
from abc import abstractmethod
from typing import List

import vertexai
from openai import AzureOpenAI

from rtl_rag_chatbot_api.common.base_handler import BaseRAGHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VertexAIRAGHandler(BaseRAGHandler):
    """
    Unified base class for Vertex AI model handlers (Gemini, Anthropic).
    Contains all shared RAG logic using Azure embeddings.
    """

    def __init__(
        self,
        configs,
        gcs_handler,
        model_choice: str = None,
        file_id: str = None,
        file_ids: List[str] = None,
        all_file_infos: dict = None,
        collection_name_prefix: str = "rag_collection_",
        user_id: str = None,
        temperature: float = 0.8,
        vertex_project: str = None,
        vertex_location: str = None,
    ):
        super().__init__(configs, gcs_handler)

        # Initialize Vertex AI with project/location
        vertexai.init(project=vertex_project, location=vertex_location)
        logger.info(
            "Initialized Vertex AI with project: %s, location: %s",
            vertex_project,
            vertex_location,
        )

        # Initialize Azure OpenAI clients for unified embeddings
        # Legacy ada-002 client (for older collections)
        self.azure_client_ada002 = AzureOpenAI(
            api_key=configs.azure_embedding.azure_embedding_api_key,
            azure_endpoint=configs.azure_embedding.azure_embedding_endpoint,
            api_version=configs.azure_embedding.azure_embedding_api_version,
        )
        # New 03-large client (default for new collections)
        self.azure_client_03_large = AzureOpenAI(
            api_key=configs.azure_embedding_3_large.azure_embedding_3_large_api_key,
            azure_endpoint=configs.azure_embedding_3_large.azure_embedding_3_large_endpoint,
            api_version=configs.azure_embedding_3_large.azure_embedding_3_large_api_version,
        )
        self.azure_embedding_deployment_03_large = (
            configs.azure_embedding_3_large.azure_embedding_3_large_deployment
        )
        self.azure_embedding_deployment_ada002 = (
            configs.azure_embedding.azure_embedding_deployment
        )

        # Common state
        self.configs = configs
        self.model_choice = model_choice
        self.user_id = user_id
        self.all_file_infos = all_file_infos if all_file_infos else {}
        self.temperature = temperature
        self._collection_name_prefix = collection_name_prefix
        self.active_file_ids: List[str] = []
        self.is_multi_file = False

        # Handle multi-file or single-file initialization
        if file_ids and len(file_ids) > 0:
            self.is_multi_file = True
            self.active_file_ids = sorted(list(set(file_ids)))
            self.file_id = None
            self.collection_name = None
            self.embedding_type = None
            logger.info(
                "VertexAI handler initialized for multi-file: %s", self.active_file_ids
            )
        elif file_id:
            self.is_multi_file = False
            self.active_file_ids = [file_id]
            self.file_id = file_id
            self.collection_name = f"{self._collection_name_prefix}{self.file_id}"
            self.embedding_type = "azure"
            logger.info(
                "VertexAI handler initialized for single-file: %s", self.file_id
            )

    def initialize(
        self,
        model: str,
        file_id: str = None,
        embedding_type: str = None,
        collection_name: str = None,
        user_id: str = None,
    ):
        """Initialize with specific configurations (compatibility with ModelHandler)."""
        self._initialize_model(model)
        self.file_id = file_id
        self.embedding_type = embedding_type
        self.collection_name = collection_name or f"rag_collection_{file_id}"
        self.user_id = user_id
        if file_id and not self.active_file_ids:
            self.active_file_ids = [file_id]
            self.is_multi_file = False

    @abstractmethod
    def _initialize_model(self, model_choice: str):
        """Initialize the specific model (Gemini/Anthropic). Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _call_model(self, prompt: str) -> str:
        """Call the specific model with the prompt. Must be implemented by subclasses."""
        pass

    def _get_embedding_config_for_file(self, file_id: str = None):
        """Select correct Azure embedding client/deployment based on file's embedding_type.

        Falls back to 03-large for new/tabular files; uses ada-002 for legacy files
        marked as 'azure'.
        """
        # Defaults
        default_client = self.azure_client_03_large
        default_deployment = self.azure_embedding_deployment_03_large

        if not file_id:
            return default_client, default_deployment

        try:
            # Prefer in-memory all_file_infos
            if self.all_file_infos and file_id in self.all_file_infos:
                file_info_data = self.all_file_infos[file_id]
                embedding_type = file_info_data.get("embedding_type", "azure-03-large")
                if file_info_data.get("is_tabular", False):
                    embedding_type = "azure-03-large"
                    logging.info(
                        "Tabular file %s detected, using text-embedding-3-large",
                        file_id,
                    )
            else:
                # Fallback to local file_info.json if available
                import json
                import os

                local_info_path = os.path.join("./chroma_db", file_id, "file_info.json")
                if os.path.exists(local_info_path):
                    with open(local_info_path, "r") as f:
                        file_info_data = json.load(f)
                        embedding_type = file_info_data.get(
                            "embedding_type", "azure-03-large"
                        )
                        if file_info_data.get("is_tabular", False):
                            embedding_type = "azure-03-large"
                            logging.info(
                                "Tabular file %s detected, using text-embedding-3-large",
                                file_id,
                            )
                else:
                    embedding_type = "azure-03-large"

            logging.info("File %s has embedding_type: '%s'", file_id, embedding_type)

            if embedding_type == "azure":
                logging.info("Using legacy ada-002 embeddings for file %s", file_id)
                return self.azure_client_ada002, self.azure_embedding_deployment_ada002
            else:
                logging.info(
                    "Using text-embedding-3-large for file %s (embedding_type: '%s')",
                    file_id,
                    embedding_type,
                )
                return default_client, default_deployment
        except Exception as e:
            logging.warning(
                "Error getting embedding config for file %s: %s, using default",
                file_id,
                str(e),
            )
            return default_client, default_deployment

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get Azure embeddings, selecting the correct deployment per file."""
        try:
            logger.info(
                "Getting Azure embeddings for %d texts (unified embedding approach)",
                len(texts),
            )

            # Determine file context (first active file or singular file_id)
            file_id_for_config = None
            if self.active_file_ids:
                file_id_for_config = self.active_file_ids[0]
            elif getattr(self, "file_id", None):
                file_id_for_config = self.file_id

            azure_client, azure_deployment = self._get_embedding_config_for_file(
                file_id_for_config
            )

            all_embeddings = []
            batch_size = min(len(texts), getattr(self, "BATCH_SIZE", 5))

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                max_retries = 3
                retry_delay = 1
                attempt = 0
                while attempt < max_retries:
                    try:
                        response = azure_client.embeddings.create(
                            input=batch, model=azure_deployment
                        )
                        batch_embeddings = [item.embedding for item in response.data]
                        all_embeddings.extend(batch_embeddings)
                        break
                    except Exception as e:
                        attempt += 1
                        if attempt >= max_retries:
                            logger.error(
                                "Failed to get Azure embeddings after %d attempts: %s",
                                max_retries,
                                str(e),
                            )
                            raise
                        logger.warning(
                            "Azure Embedding API error (attempt %d/%d): %s. Retrying in %ds...",
                            attempt,
                            max_retries,
                            str(e),
                            retry_delay,
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2

            return all_embeddings
        except Exception as e:
            logging.error("Error getting embeddings: %s", str(e))
            raise

    def _collect_relevant_docs(
        self, question: str, query_embedding: List[float], n_results_per_file: int = 3
    ) -> List[str]:
        """Collect relevant documents from ChromaDB."""
        all_relevant_docs: List[str] = []

        if self.is_multi_file:
            for f_id in self.active_file_ids:
                try:
                    current_collection_name = f"{self._collection_name_prefix}{f_id}"
                    logger.info(
                        "Querying ChromaDB for file: %s, collection: %s",
                        f_id,
                        current_collection_name,
                    )
                    chroma_collection = self.chroma_manager.get_collection(
                        file_id=f_id,
                        embedding_type="azure",
                        collection_name=current_collection_name,
                        user_id=self.user_id,
                        is_embedding=False,
                    )
                    results = chroma_collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_results_per_file,
                    )
                    docs_from_file = (
                        results["documents"][0] if results["documents"] else []
                    )
                    file_info = self.all_file_infos.get(f_id, {})
                    original_filename = file_info.get("original_filename", "Document")
                    docs_with_source = [
                        f"[Source: {original_filename}] {doc}" for doc in docs_from_file
                    ]
                    all_relevant_docs.extend(docs_with_source)
                    logger.info("Retrieved %d docs from %s", len(docs_from_file), f_id)
                except Exception as e:
                    logger.error(
                        "Error querying ChromaDB for file %s: %s", f_id, str(e)
                    )
        else:
            relevant_docs = self.query_chroma(question, self.file_id)
            all_relevant_docs = relevant_docs
            logger.info("Retrieved %d docs from %s", len(relevant_docs), self.file_id)

        return all_relevant_docs

    def _build_files_context(self) -> str:
        """Build files context string."""
        files_context = ""
        if self.all_file_infos:
            if self.is_multi_file:
                file_details = []
                for file_id in self.active_file_ids:
                    if file_id in self.all_file_infos:
                        file_info = self.all_file_infos.get(file_id, {})
                        original_filename = file_info.get(
                            "original_filename", "Unknown document"
                        )
                        if "url" in file_info:
                            url = file_info.get("url")
                            file_details.append(f"- {original_filename} (URL: {url})")
                        else:
                            file_details.append(f"- {original_filename}")
                if file_details:
                    files_context = (
                        "Available documents:\n" + "\n".join(file_details) + "\n\n"
                    )
            else:
                file_id = self.active_file_ids[0]
                if file_id in self.all_file_infos:
                    file_info = self.all_file_infos.get(file_id, {})
                    original_filename = file_info.get(
                        "original_filename", f"Unknown filename (ID: {file_id})"
                    )
                    file_context_detail = f"- {original_filename}"
                    if "url" in file_info:
                        url = file_info.get("url")
                        file_context_detail += f" (URL: {url})"
                    files_context = "File Information:\n" + file_context_detail + "\n\n"
        return files_context

    def _build_rag_prompt(
        self, files_context: str, context_str: str, question: str
    ) -> str:
        """Build the final RAG prompt."""
        return f"""{self.configs.chatbot.system_prompt_rag_llm}
            Elaborate and give detailed answer based on the context provided.

            {files_context}Context:
            {context_str}

            Question: {question}

            Answer:"""

    def get_answer(self, query: str) -> str:
        """Generate an answer using RAG (unified implementation)."""
        try:
            logger.info(
                "VertexAI handler (%s) get_answer for file(s): %s",
                self.model_choice,
                self.active_file_ids,
            )

            # Compute query embedding once
            query_embedding = self.get_embeddings([query])[0]

            # Collect relevant documents
            all_relevant_docs = self._collect_relevant_docs(query, query_embedding, 3)

            if not all_relevant_docs:
                logger.warning(
                    "No relevant documents found for query: '%s' in files: %s",
                    query,
                    self.active_file_ids,
                )

            # Build context and prompt
            files_context = self._build_files_context()
            max_docs = max(5, len(self.active_file_ids) * 3)
            context_str = "\n".join(all_relevant_docs[:max_docs])
            prompt = self._build_rag_prompt(files_context, context_str, query)

            # Call the specific model implementation
            response = self._call_model(prompt)
            logger.info("Received answer from model for files %s", self.active_file_ids)
            return response.strip()

        except Exception as e:
            logger.error(
                "Error in VertexAI handler get_answer for files %s: %s",
                self.active_file_ids,
                str(e),
                exc_info=True,
            )
            return f"An error occurred while processing your question: {str(e)}"
