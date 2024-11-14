import asyncio
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

    async def get_gemini_response_stream(self, prompt: str):
        """Stream responses from Gemini model."""
        try:
            responses = self.generative_model.generate_content(prompt, stream=True)
            for chunk in responses:
                if chunk.text:
                    yield chunk.text
                    await asyncio.sleep(0.1)
        except Exception as e:
            logging.error(f"Error in get_gemini_response_stream: {str(e)}")
            yield f"Error: {str(e)}"

    def get_answer(self, query: str) -> str:
        """Generate an answer to a query using relevant context."""
        try:
            relevant_chunks = self.query_chroma(query, self.file_id, n_results=3)
            if not relevant_chunks:
                return (
                    "I couldn't find any relevant information to answer your question."
                )

            context = "\n".join(relevant_chunks)
            prompt = f"""Based on the following context, please answer the question.
            If the answer is not in the context, say 'I don't have enough information
            to answer that question from the uploaded document. Please rephrase or ask another question.

            Context: {context}

            Question: {query}

            Answer:"""

            response = self.generative_model.generate_content(prompt)
            logger.info("Response from Google")
            return response.text
        except Exception as e:
            logging.error(f"Error in get_answer: {str(e)}")
            return f"Error generating response: {str(e)}"
