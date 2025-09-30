import logging
import time
from typing import List

# Import Azure OpenAI for embeddings
import vertexai
from openai import AzureOpenAI
from vertexai.generative_models import GenerativeModel
from vertexai.preview.generative_models import (
    GenerationConfig,
    HarmBlockThreshold,
    HarmCategory,
)
from vertexai.preview.language_models import TextEmbeddingModel

from rtl_rag_chatbot_api.common.base_handler import BaseRAGHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiSafetyFilterError(Exception):
    """Custom exception for Gemini safety filter blocks."""

    def __init__(
        self, message: str, safety_ratings: dict = None, finish_reason: str = None
    ):
        super().__init__(message)
        self.safety_ratings = safety_ratings or {}
        self.finish_reason = finish_reason


def _handle_gemini_safety_response(response, context: str = ""):
    """
    Handle Gemini responses that might be blocked by safety filters.

    Args:
        response: The Gemini response object
        context: Additional context for logging (e.g., "streaming", "non-rag")

    Returns:
        str: The response text if available

    Raises:
        GeminiSafetyFilterError: If response is blocked by safety filters
    """
    try:
        # Try to get the text from the response
        if hasattr(response, "text") and response.text:
            return response.text

        # If no text, check if it's blocked by safety filters
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]

            # Check if candidate has no parts (safety filter block)
            if not hasattr(candidate, "content") or not candidate.content.parts:
                safety_info = {}
                finish_reason = getattr(candidate, "finish_reason", "UNKNOWN")

                # Extract safety ratings if available
                if hasattr(candidate, "safety_ratings"):
                    safety_info = {}
                    for rating in candidate.safety_ratings:
                        severity = "UNKNOWN"
                        if hasattr(rating, "severity"):
                            severity = getattr(rating, "severity", {}).name
                        safety_info[rating.category.name] = {
                            "probability": rating.probability.name,
                            "severity": severity,
                        }

                # Create a user-friendly error message
                error_msg = "Response was blocked by Google's safety filters. "
                if finish_reason == "SAFETY":
                    error_msg += "The content was flagged as potentially harmful. "
                elif finish_reason == "MAX_TOKENS":
                    error_msg += "The response was truncated due to length limits. "
                else:
                    error_msg += f"Reason: {finish_reason}. "

                error_msg += (
                    "Please try rephrasing your question or asking something different."
                )

                logger.warning(
                    f"Gemini {context} response blocked by safety filters: {error_msg}"
                )
                logger.info(f"Safety ratings: {safety_info}")
                logger.info(f"Finish reason: {finish_reason}")

                raise GeminiSafetyFilterError(error_msg, safety_info, finish_reason)

        # If we get here, there's some other issue
        error_msg = "No response content available from Gemini model."
        logger.warning(f"Gemini {context} response has no content: {error_msg}")
        raise GeminiSafetyFilterError(error_msg)

    except GeminiSafetyFilterError:
        # Re-raise our custom exception
        raise
    except Exception as e:
        # Handle other potential errors
        if (
            "safety filters" in str(e).lower()
            or "candidate content has no parts" in str(e).lower()
        ):
            error_msg = (
                "Response was blocked by Google's safety filters. "
                "Please try rephrasing your question or asking something different."
            )
            logger.warning(f"Gemini {context} safety filter error: {str(e)}")
            raise GeminiSafetyFilterError(error_msg)
        else:
            # For other errors, let them bubble up
            raise


class GeminiHandler(BaseRAGHandler):
    """
    Handles interactions with Google's Gemini AI models for RAG applications.
    Inherits common functionality from BaseRAGHandler.
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
    ):
        super().__init__(configs, gcs_handler)
        # Initialize Gemini
        vertexai.init(project=configs.gemini.project, location=configs.gemini.location)
        logger.info(
            f"Initialized Gemini with project: {configs.gemini.project}, location: {configs.gemini.location}"
        )

        # Initialize both Gemini and Azure embedding models
        # Keep Gemini embedding model for backward compatibility
        self.gemini_embedding_model = TextEmbeddingModel.from_pretrained(
            "textembedding-gecko@latest"
        )

        # Initialize Azure OpenAI clients for embeddings
        # This is critical for our unified embedding approach

        # Legacy ada-002 client
        self.azure_client_ada002 = AzureOpenAI(
            api_key=configs.azure_embedding.azure_embedding_api_key,
            azure_endpoint=configs.azure_embedding.azure_embedding_endpoint,
            api_version=configs.azure_embedding.azure_embedding_api_version,
        )

        # New 03 client
        self.azure_client_03 = AzureOpenAI(
            api_key=configs.azure_embedding_3_large.azure_embedding_3_large_api_key,
            azure_endpoint=configs.azure_embedding_3_large.azure_embedding_3_large_endpoint,
            api_version=configs.azure_embedding_3_large.azure_embedding_3_large_api_version,
        )

        # Default to 03 for new embeddings
        self.azure_client = self.azure_client_03
        self.azure_embedding_deployment = (
            configs.azure_embedding_3_large.azure_embedding_3_large_deployment
        )

        self.generative_model = None
        self.MAX_TOKENS_PER_REQUEST = 15000

        # Add multi-file support similar to AzureChatbot
        self.configs = configs
        self.model_choice = model_choice
        self.user_id = user_id
        self.all_file_infos = all_file_infos if all_file_infos else {}
        self.temperature = temperature

        # Flag to control which embedding system to use
        # With our unified approach, we always use Azure embeddings
        self.use_azure_embeddings = True
        self._collection_name_prefix = collection_name_prefix
        self.active_file_ids: List[str] = []
        self.is_multi_file = False

        # Initialize the model if model_choice is provided
        if model_choice:
            # Map model choice to actual model name
            self._initialize_gemini_model(model_choice, temperature)

        # Handle multi-file or single file mode
        if file_ids and len(file_ids) > 0:
            self.is_multi_file = True
            self.active_file_ids = sorted(
                list(set(file_ids))
            )  # Ensure unique and sorted
            # For BaseRAGHandler compatibility if its methods are ever called directly in multi-mode
            self.file_id = None
            self.collection_name = None
            self.embedding_type = (
                None  # In multi-file, embedding type is always 'google' for this class
            )
            logger.info(
                f"GeminiHandler initialized for multi-file: {self.active_file_ids}"
            )
        elif file_id:
            self.is_multi_file = False
            self.active_file_ids = [file_id]
            self.file_id = file_id  # For BaseRAGHandler compatibility
            self.collection_name = f"{self._collection_name_prefix}{self.file_id}"
            self.embedding_type = (
                "azure"  # Using Azure embeddings as default for all handlers
            )
            logger.info(f"GeminiHandler initialized for single-file: {self.file_id}")
        # If neither file_id nor file_ids are provided, do nothing - will be set during initialize()

    def _initialize_gemini_model(self, model_choice: str, temperature: float = 0.8):
        """Initialize the Gemini model with the specified model choice and temperature."""
        # Map model choice to actual model name - only 2.5 models
        model_mapping = {
            "gemini-2.5-flash": self.configs.gemini.model_flash,
            "gemini-2.5-pro": self.configs.gemini.model_pro,
        }

        actual_model = model_mapping.get(model_choice)
        if not actual_model:
            raise ValueError(
                f"Invalid model choice: {model_choice}. "
                f"Available models: {list(model_mapping.keys())}"
            )

        # Use VertexAI approach for all Gemini models (including 2.5)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=1,
            top_k=40,
            max_output_tokens=4096,
        )

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }

        self.generative_model = GenerativeModel(
            model_name=actual_model,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

    def initialize(
        self,
        model: str,
        file_id: str = None,
        embedding_type: str = None,
        collection_name: str = None,
        user_id: str = None,
    ):
        """Initialize the Gemini model with specific configurations."""
        # Initialize the model
        self._initialize_gemini_model(model)

        # Set the file and collection parameters
        self.file_id = file_id
        self.embedding_type = embedding_type
        self.collection_name = collection_name or f"rag_collection_{file_id}"
        self.user_id = user_id

        # For backwards compatibility, also set active_file_ids if not already set
        if file_id and not self.active_file_ids:
            self.active_file_ids = [file_id]
            self.is_multi_file = False

    def _get_embedding_config_for_file(self, file_id: str = None):
        """Get the appropriate embedding client and deployment based on file's embedding_type"""

        # TODO ZL: this has to go to BaseRAGHandler or a common utility, since this class inherits from it
        # and the functionality is same as in gemini_handler.py

        # Default to new embedding for new files
        default_client = self.azure_client_03
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
                    self.azure_client_ada002,
                    self.configs.azure_embedding.azure_embedding_deployment,
                )
            else:  # "azure-3-large" or any other value (including tabular files) defaults to new embedding
                logging.info(
                    f"Using text-embedding-3-large for file {file_id} (embedding_type: '{embedding_type}')"
                )
                return (
                    self.azure_client_03,
                    self.configs.azure_embedding_3_large.azure_embedding_3_large_deployment,
                )
        except Exception as e:
            logging.warning(
                f"Error getting embedding config for file {file_id}: {e}, using default"
            )
            return default_client, default_deployment

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts, using Azure embeddings for unified approach.

        This method has been updated to use Azure embeddings to match the stored embeddings.
        Since we're using a unified embedding approach where all models use Azure embeddings,
        we need to make sure our query embeddings also use Azure's embedding model
        to match the dimensionality (1536 vs 768 for Gemini).

        Args:
            texts: List of text strings to get embeddings for

        Returns:
            List of embedding vectors
        """
        try:
            # Use Azure embeddings (to match dimensionality with stored embeddings)
            if self.use_azure_embeddings:
                # For query-time, check the first active file to determine embedding type
                file_id_for_config = None
                if hasattr(self, "active_file_ids") and self.active_file_ids:
                    file_id_for_config = self.active_file_ids[0]
                elif hasattr(self, "file_id") and self.file_id:
                    file_id_for_config = self.file_id

                # Get appropriate client and deployment
                azure_client, azure_deployment = self._get_embedding_config_for_file(
                    file_id_for_config
                )

                logger.info(
                    f"Getting Azure embeddings for {len(texts)} texts (for unified embedding approach)"
                )
                all_embeddings = []
                batch_size = min(len(texts), self.BATCH_SIZE)

                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    # Implement retry logic with exponential backoff
                    max_retries = 3
                    retry_delay = 1  # Initial delay in seconds
                    attempt = 0

                    while attempt < max_retries:
                        try:
                            # Use Azure OpenAI embedding API with appropriate client and deployment
                            response = azure_client.embeddings.create(
                                input=batch, model=azure_deployment
                            )

                            # Extract embeddings from response
                            batch_embeddings = [
                                item.embedding for item in response.data
                            ]
                            all_embeddings.extend(batch_embeddings)
                            break  # Success, exit retry loop
                        except Exception as e:
                            attempt += 1
                            if attempt >= max_retries:
                                logger.error(
                                    f"Failed to get Azure embeddings after {max_retries} attempts: {str(e)}"
                                )
                                raise  # Re-raise the exception after all retries failed

                            # Log the retry attempt
                            logger.warning(
                                f"Azure Embedding API error (attempt {attempt}/{max_retries}):"
                                f" {str(e)}. Retrying in {retry_delay}s..."
                            )

                            # Sleep with exponential backoff
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff

                return all_embeddings

            # Fallback to Gemini embeddings (not used with unified approach, but keeping for backward compatibility)
            else:
                logger.info(f"Getting Gemini embeddings for {len(texts)} texts")
                all_embeddings = []
                batch_size = min(len(texts), self.BATCH_SIZE)

                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    # Implement retry logic with exponential backoff
                    max_retries = 3
                    retry_delay = 1  # Initial delay in seconds
                    attempt = 0

                    while attempt < max_retries:
                        try:
                            embeddings = self.gemini_embedding_model.get_embeddings(
                                batch
                            )
                            all_embeddings.extend(
                                [embedding.values for embedding in embeddings]
                            )
                            break  # Success, exit retry loop
                        except Exception as e:
                            attempt += 1
                            if attempt >= max_retries:
                                logger.error(
                                    f"Failed to get Gemini embeddings after {max_retries} attempts: {str(e)}"
                                )
                                raise  # Re-raise the exception after all retries failed

                            # Log the retry attempt
                            logger.warning(
                                f"Gemini Embedding API error (attempt {attempt}/{max_retries}):"
                                f" {str(e)}. Retrying in {retry_delay}s..."
                            )

                            # Sleep with exponential backoff
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff

                return all_embeddings
        except Exception as e:
            logging.error(f"Error getting embeddings: {str(e)}")
            raise

    def get_gemini_response_stream(self, prompt: str) -> str:
        """Stream responses from Gemini model and concatenate them."""
        try:
            # Ensure model is initialized
            if self.generative_model is None:
                if not self.model_choice:
                    raise ValueError(
                        "Model choice not set. Cannot initialize Gemini model."
                    )
                self._initialize_gemini_model(self.model_choice, self.temperature)

            # Use VertexAI approach for all Gemini models (including 2.5)
            responses = self.generative_model.generate_content(prompt, stream=True)
            full_response = ""

            for response in responses:
                try:
                    response_text = _handle_gemini_safety_response(
                        response, "streaming"
                    )
                    full_response += response_text
                except GeminiSafetyFilterError as e:
                    # For streaming, if any part is blocked, return what we have plus the error
                    if full_response:
                        return (
                            f"{full_response}\n\n[Note: Response was partially blocked "
                            f"by safety filters: {str(e)}]"
                        )
                    else:
                        # If nothing was generated, re-raise the exception
                        raise

            return full_response

        except GeminiSafetyFilterError:
            # Re-raise safety filter errors to be handled by the calling code
            raise
        except Exception as e:
            logging.error(f"Error in Gemini response streaming: {str(e)}")
            # Check if it's a safety filter error in disguise
            if (
                "safety filters" in str(e).lower()
                or "candidate content has no parts" in str(e).lower()
            ):
                raise GeminiSafetyFilterError(
                    "Response was blocked by Google's safety filters. "
                    "Please try rephrasing your question or asking something different."
                )
            return f"Error generating response: {str(e)}"

    def get_answer(self, query: str) -> str:
        """Generate an answer to a query using relevant context."""
        try:
            logger.info(
                f"GeminiHandler ({self.model_choice}) get_answer for file(s): {self.active_file_ids}"
            )

            all_relevant_docs = []
            # IMPORTANT: Use Azure embeddings for all models including Gemini
            # With our unified embedding approach, all models use Azure embeddings
            embedding_model_for_rag = "azure"  # Use Azure embeddings for RAG

            # Compute embedding only once per request to avoid making the same
            # Azure embedding call for each file.
            query_embedding = self.get_embeddings([query])[0]

            # Handle multi-file or single-file mode appropriately
            if self.is_multi_file:
                # Query each file and collect relevant documents
                for f_id in self.active_file_ids:
                    # Construct collection name specific to this file_id
                    try:
                        current_collection_name = (
                            f"{self._collection_name_prefix}{f_id}"
                        )
                        logger.info(
                            f"Querying ChromaDB for file: {f_id}, collection: {current_collection_name}"
                        )
                        chroma_collection = self.chroma_manager.get_collection(
                            file_id=f_id,
                            embedding_type=embedding_model_for_rag,
                            collection_name=current_collection_name,
                            user_id=self.user_id,
                            is_embedding=False,
                        )

                        # Query the collection
                        results = chroma_collection.query(
                            query_embeddings=[query_embedding],
                            n_results=3,  # Get top 3 results per file
                        )

                        # Extract documents from results
                        docs_from_file = (
                            results["documents"][0] if results["documents"] else []
                        )

                        # Add source filename (instead of file ID) to each document
                        file_info = self.all_file_infos.get(f_id, {})
                        original_filename = file_info.get(
                            "original_filename", "Document"
                        )
                        docs_with_source = [
                            f"[Source: {original_filename}] {doc}"
                            for doc in docs_from_file
                        ]
                        all_relevant_docs.extend(docs_with_source)

                        logger.info(f"Retrieved {len(docs_from_file)} docs from {f_id}")
                    except Exception as e:
                        logger.error(
                            f"Error querying ChromaDB for file {f_id}: {str(e)}"
                        )
                        # Continue to next file if there's an error with one
            else:
                # Single-file mode - use the existing query_chroma method
                relevant_docs = self.query_chroma(query, self.file_id)
                all_relevant_docs = relevant_docs
                logger.info(f"Retrieved {len(relevant_docs)} docs from {self.file_id}")

            # Check if we found any relevant documents
            if not all_relevant_docs:
                logger.warning(
                    f"No relevant documents found for query: '{query}' in files: {self.active_file_ids}"
                )
                # No early return - continue with empty context like the AzureChatbot implementation

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
                                "original_filename", "Unknown document"
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
                        logger.info(
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
                        logger.info(
                            f"Added clean file context for single file: {file_id}"
                        )

            # Construct the prompt with context
            # Dynamic limit based on number of files to ensure all files get representation
            max_docs = max(5, len(self.active_file_ids) * 3)  # At least 3 docs per file
            context_str = "\n".join(all_relevant_docs[:max_docs])

            prompt = f"""{self.configs.chatbot.system_prompt_rag_llm}
            Elaborate and give detailed answer based on the context provided.

            {files_context}Context:
            {context_str}

            Question: {query}

            Answer:"""

            # Get streaming response and return it
            logger.info(f"Sending to Gemini model. Files: {self.active_file_ids}")
            response = self.get_gemini_response_stream(prompt)
            logger.info(f"Received answer from Gemini for files {self.active_file_ids}")
            return response.strip()

        except Exception as e:
            logger.error(
                f"Error in GeminiHandler get_answer for files {self.active_file_ids}: {str(e)}",
                exc_info=True,
            )
            # Check if it's a safety filter error
            if isinstance(e, GeminiSafetyFilterError):
                return f"I apologize, but I cannot provide a response to this question. {str(e)}"
            return f"An error occurred while processing your question: {str(e)}"


def get_gemini_non_rag_response(
    config,
    prompt: str,
    model_choice: str,
    temperature: float = 0.8,
    max_tokens: int = 4096,
) -> str:
    """
    Get a response from Gemini model without using RAG context.

    Args:
        config: Configuration object containing Gemini settings
        prompt (str): The prompt to send to the model
        model_choice (str): The specific Gemini model to use
            (gemini-2.5-flash or gemini-2.5-pro)

    Returns:
        str: The model's response

    Raises:
        GeminiSafetyFilterError: If response is blocked by safety filters
        ValueError: If model configuration is invalid
    """
    try:
        # Map model choice to actual model name - only 2.5 models
        model_mapping = {
            "gemini-2.5-flash": config.gemini.model_flash,
            "gemini-2.5-pro": config.gemini.model_pro,
        }

        model_name = model_mapping.get(model_choice)
        if not model_name:
            raise ValueError(
                f"Invalid Gemini model choice: {model_choice}. "
                f"Available models: {list(model_mapping.keys())}"
            )

        # Use VertexAI approach for all Gemini models (including 2.5)
        # Initialize Vertex AI
        vertexai.init(project=config.gemini.project, location=config.gemini.location)

        # Initialize the model
        model = GenerativeModel(model_name)

        # Configure generation parameters for more focused responses
        generation_config = GenerationConfig(
            temperature=temperature,  # Configurable temperature
            max_output_tokens=max_tokens,  # Configurable token limit for different use cases
            top_p=0.8,  # More focused sampling
            top_k=20,  # More focused token selection
            candidate_count=1,  # Single response only
        )

        # Configure safety settings
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # Add system message to enforce direct responses
        system_prompt = config.chatbot.system_prompt_plain_llm + "\nUSER QUERY:\n"

        full_prompt = system_prompt + prompt

        # Generate response
        response = model.generate_content(
            full_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        # Handle the response with safety filter checking
        answer = _handle_gemini_safety_response(response, "non-rag")
        return answer.strip()

    except GeminiSafetyFilterError:
        # Re-raise safety filter errors to be handled by the calling code
        raise
    except Exception as e:
        logging.error(f"Error in get_gemini_non_rag_response: {str(e)}", exc_info=True)
        # Check if it's a safety filter error in disguise
        if (
            "safety filters" in str(e).lower()
            or "candidate content has no parts" in str(e).lower()
        ):
            raise GeminiSafetyFilterError(
                "Response was blocked by Google's safety filters. "
                "Please try rephrasing your question or asking something different."
            )
        return f"Error generating response: {str(e)}"
