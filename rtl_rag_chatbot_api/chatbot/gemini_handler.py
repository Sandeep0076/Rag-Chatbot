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

        # Initialize Azure OpenAI client for embeddings
        # This is critical for our unified embedding approach
        self.azure_client = AzureOpenAI(
            api_key=configs.azure_embedding.azure_embedding_api_key,
            azure_endpoint=configs.azure_embedding.azure_embedding_endpoint,
            api_version=configs.azure_embedding.azure_embedding_api_version,
        )
        self.azure_embedding_deployment = (
            configs.azure_embedding.azure_embedding_deployment
        )

        self.generative_model = None
        self.MAX_TOKENS_PER_REQUEST = 15000

        # Add multi-file support similar to AzureChatbot
        self.configs = configs
        self.model_choice = model_choice
        self.user_id = user_id
        self.all_file_infos = all_file_infos if all_file_infos else {}

        # Flag to control which embedding system to use
        # With our unified approach, we always use Azure embeddings
        self.use_azure_embeddings = True
        self._collection_name_prefix = collection_name_prefix
        self.active_file_ids: List[str] = []
        self.is_multi_file = False

        # Initialize the model if model_choice is provided
        if model_choice:
            # Map model choice to actual model name
            self._initialize_gemini_model(model_choice)

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

    def _initialize_gemini_model(self, model_choice: str):
        """Initialize the Gemini model with the specified model choice."""
        # Map model choice to actual model name
        model_mapping = {
            "gemini-flash": self.configs.gemini.model_flash,
            "gemini-pro": self.configs.gemini.model_pro,
        }

        actual_model = model_mapping.get(model_choice)
        if not actual_model:
            raise ValueError(f"Invalid model choice: {model_choice}")

        generation_config = GenerationConfig(
            temperature=0.9,
            top_p=1,
            top_k=40,
            max_output_tokens=2048,
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
                            # Use Azure OpenAI embedding API
                            response = self.azure_client.embeddings.create(
                                input=batch, model=self.azure_embedding_deployment
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
            responses = self.generative_model.generate_content(prompt, stream=True)
            full_response = ""
            for response in responses:
                if response and response.text:
                    full_response += response.text
            return full_response
        except Exception as e:
            logging.error(f"Error in Gemini response streaming: {str(e)}")
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
                    # For multi-file, show list with filenames
                    file_details = []
                    for file_id in self.active_file_ids:
                        if file_id in self.all_file_infos:
                            file_info = self.all_file_infos.get(file_id, {})
                            original_filename = file_info.get(
                                "original_filename", "Unknown document"
                            )
                            file_details.append(f"- {original_filename}")

                    if file_details:
                        files_context = (
                            "Available documents:\n" + "\n".join(file_details) + "\n\n"
                        )
                        logger.info(
                            f"Added multi-file context with {len(file_details)} files"
                        )
                else:
                    # For single file, provide complete file_info.json as context
                    file_id = self.active_file_ids[0]
                    if file_id in self.all_file_infos:
                        file_info = self.all_file_infos.get(file_id, {})
                        import json

                        # Format the file info nicely
                        file_info_str = json.dumps(file_info, indent=2, default=str)
                        files_context = f"File Information:\n{file_info_str}\n\n"
                        logger.info(
                            f"Added complete file_info.json context for single file: {file_id}"
                        )

            # Construct the prompt with context
            context_str = "\n".join(all_relevant_docs[:5])  # Limit to top 5 total docs

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
            return f"An error occurred while processing your question: {str(e)}"


def get_gemini_non_rag_response(config, prompt: str, model_choice: str) -> str:
    """
    Get a response from Gemini model without using RAG context.

    Args:
        config: Configuration object containing Gemini settings
        prompt (str): The prompt to send to the model
        model_choice (str): The specific Gemini model to use (e.g., 'gemini-flash', 'gemini-pro')

    Returns:
        str: The model's response

    Raises:
        ValueError: If model configuration is invalid
    """
    try:
        # Initialize Vertex AI
        vertexai.init(project=config.gemini.project, location=config.gemini.location)

        # Map model choice to actual model name
        model_mapping = {
            "gemini-flash": config.gemini.model_flash,
            "gemini-pro": config.gemini.model_pro,
        }

        model_name = model_mapping.get(model_choice)
        if not model_name:
            raise ValueError(f"Invalid Gemini model choice: {model_choice}")

        # Initialize the model
        model = GenerativeModel(model_name)

        # Configure generation parameters for more focused responses
        generation_config = GenerationConfig(
            temperature=0.1,  # Lower temperature for more focused responses
            max_output_tokens=1024,  # Reduced token limit to discourage verbosity
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

        # Clean up response
        answer = response.text.strip()
        return answer

    except Exception as e:
        logging.error(f"Error in get_gemini_non_rag_response: {str(e)}", exc_info=True)
        return f"Error generating response: {str(e)}"
