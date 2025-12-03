import logging
from typing import List

from vertexai.generative_models import GenerativeModel
from vertexai.preview.generative_models import (
    GenerationConfig,
    HarmBlockThreshold,
    HarmCategory,
)
from vertexai.preview.language_models import TextEmbeddingModel

from rtl_rag_chatbot_api.chatbot.utils.vertexai_common import VertexAIRAGHandler
from rtl_rag_chatbot_api.common.errors import ModelInitializationError

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


class GeminiHandler(VertexAIRAGHandler):
    """
    Handles interactions with Google's Gemini AI models for RAG applications.
    Inherits unified RAG logic from VertexAIRAGHandler.
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
        custom_gpt: bool = False,
        system_prompt: str = None,
    ):
        # Initialize the unified base class
        super().__init__(
            configs=configs,
            gcs_handler=gcs_handler,
            model_choice=model_choice,
            file_id=file_id,
            file_ids=file_ids,
            all_file_infos=all_file_infos,
            collection_name_prefix=collection_name_prefix,
            user_id=user_id,
            temperature=temperature,
            vertex_project=configs.gemini.project,
            vertex_location=configs.gemini.location,
            custom_gpt=custom_gpt,
            system_prompt=system_prompt,
        )

        # Gemini-specific initialization
        # Keep Gemini embedding model for backward compatibility
        self.gemini_embedding_model = TextEmbeddingModel.from_pretrained(
            "textembedding-gecko@latest"
        )
        self.generative_model = None
        self.MAX_TOKENS_PER_REQUEST = 15000

        # Initialize the model if provided
        if model_choice:
            self._initialize_model(model_choice)

    def _initialize_model(self, model_choice: str):
        """Initialize the Gemini model with the specified model choice and temperature."""
        # Map model choice to actual model name - only 2.5 models
        model_mapping = {
            "gemini-2.5-flash": self.configs.gemini.model_flash,
            "gemini-2.5-pro": self.configs.gemini.model_pro,
        }

        actual_model = model_mapping.get(model_choice)
        if not actual_model:
            raise ModelInitializationError(
                f"Invalid model choice: {model_choice}. "
                f"Available models: {list(model_mapping.keys())}",
                details={
                    "model_choice": model_choice,
                    "available_models": list(model_mapping.keys()),
                },
            )

        # Use VertexAI approach for all Gemini models (including 2.5)
        generation_config = GenerationConfig(
            temperature=self.temperature,
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
        logger.info("Initialized Gemini model: %s", actual_model)

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
            # Use Azure embeddings from base class
            return super().get_embeddings(texts)
        except Exception as e:
            logging.error(f"Error getting embeddings: {str(e)}")
            raise

    def get_gemini_response_stream(self, prompt: str) -> str:
        """Stream responses from Gemini model and concatenate them."""
        try:
            # Ensure model is initialized
            if self.generative_model is None:
                if not self.model_choice:
                    raise ModelInitializationError(
                        "Model choice not set. Cannot initialize Gemini model.",
                        details={"model_choice": self.model_choice},
                    )
                self._initialize_model(self.model_choice)

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

    def _call_model(self, prompt: str) -> str:
        """Call Gemini model with streaming support."""
        try:
            response = self.get_gemini_response_stream(prompt)
            return response
        except GeminiSafetyFilterError as e:
            return f"I apologize, but I cannot provide a response to this question. {str(e)}"
        except Exception as e:
            logger.error("Error calling Gemini model: %s", str(e))
            return f"Error generating response: {str(e)}"


def get_gemini_non_rag_response(
    config,
    prompt: str,
    model_choice: str,
    temperature: float = 0.8,
    max_tokens: int = 4096,
) -> str:
    """Get a response from Gemini model without using RAG context."""
    try:
        # Map model choice to actual model name - only 2.5 models
        model_mapping = {
            "gemini-2.5-flash": config.gemini.model_flash,
            "gemini-2.5-pro": config.gemini.model_pro,
        }

        model_name = model_mapping.get(model_choice)
        if not model_name:
            raise ModelInitializationError(
                f"Invalid Gemini model choice: {model_choice}. "
                f"Available models: {list(model_mapping.keys())}",
                details={
                    "model_choice": model_choice,
                    "available_models": list(model_mapping.keys()),
                },
            )

        # Use VertexAI approach for all Gemini models (including 2.5)
        import vertexai

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
        logging.error("Error in get_gemini_non_rag_response: %s", str(e), exc_info=True)
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
