import logging
from typing import List

import vertexai
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

    def __init__(self, configs, gcs_handler):
        super().__init__(configs, gcs_handler)
        vertexai.init(project=configs.gemini.project, location=configs.gemini.location)
        logger.info(
            f"Initialized Gemini with project: {configs.gemini.project}, location: {configs.gemini.location}"
        )
        self.embedding_model = TextEmbeddingModel.from_pretrained(
            "textembedding-gecko@latest"
        )
        self.generative_model = None
        self.MAX_TOKENS_PER_REQUEST = 15000

    def initialize(
        self,
        model: str,
        file_id: str = None,
        embedding_type: str = None,
        collection_name: str = None,
        user_id: str = None,
    ):
        """Initialize the Gemini model with specific configurations."""
        # Map model choice to actual model name
        model_mapping = {
            "gemini-flash": self.configs.gemini.model_flash,
            "gemini-pro": self.configs.gemini.model_pro,
        }

        actual_model = model_mapping.get(model)
        if not actual_model:
            raise ValueError(f"Invalid model choice: {model}")

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
        self.file_id = file_id
        self.embedding_type = embedding_type
        self.collection_name = collection_name or f"rag_collection_{file_id}"
        self.user_id = user_id

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            logger.info(f"Getting Gemini embeddings for {len(texts)} texts")
            # Process in smaller batches if needed
            all_embeddings = []
            batch_size = min(len(texts), self.BATCH_SIZE)

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                embeddings = self.embedding_model.get_embeddings(batch)
                all_embeddings.extend([embedding.values for embedding in embeddings])

            return all_embeddings
        except Exception as e:
            logging.error(f"Error getting Gemini embeddings: {str(e)}")
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
            # Get relevant documents from ChromaDB
            relevant_docs = self.query_chroma(query, self.file_id)

            if not relevant_docs:
                return (
                    "I couldn't find any relevant information to answer your question."
                )

            # Construct the prompt with context
            context = "\n".join(relevant_docs[:3])  # Use top 3 most relevant documents
            prompt = f"""{self.configs.chatbot.system_prompt_rag_llm}
            Elaborate the answer based on the context provided.
            Context:
            {context}

            Question: {query}

            Answer:"""

            # Get streaming response and return it
            response = self.get_gemini_response_stream(prompt)
            return response.strip()

        except Exception as e:
            logging.error(f"Error in Gemini get_answer: {str(e)}")
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
