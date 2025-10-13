import logging
from typing import List, Optional

from anthropic import AnthropicVertex

from rtl_rag_chatbot_api.chatbot.utils.vertexai_common import VertexAIRAGHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnthropicHandler(VertexAIRAGHandler):
    """
    Handles interactions with Anthropic Claude models hosted on Vertex AI for RAG applications.
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
            vertex_project=configs.anthropic.project,
            vertex_location=configs.anthropic.location,
        )

        # Anthropic-specific initialization
        self.client = AnthropicVertex(
            region=configs.anthropic.location, project_id=configs.anthropic.project
        )
        self.generative_model_name: Optional[str] = None
        self.MAX_TOKENS_PER_REQUEST = 4096

        # Initialize the model if provided
        if model_choice:
            self._initialize_model(model_choice)

    def _initialize_model(self, model_choice: str):
        """Initialize Anthropic model mapping using config."""
        model_mapping = {
            "Claude Sonnet 4": self.configs.anthropic.model_sonnet,
        }
        actual_model = model_mapping.get(model_choice)
        if not actual_model:
            raise ValueError(
                f"Invalid Anthropic model choice: {model_choice}. Available: {list(model_mapping.keys())}"
            )
        self.generative_model_name = actual_model
        logger.info("Initialized Anthropic model: %s", actual_model)

    def _call_model(self, prompt: str) -> str:
        """Call Anthropic Vertex model with the prompt."""
        try:
            message = self.client.messages.create(
                model=self.generative_model_name,
                max_tokens=self.MAX_TOKENS_PER_REQUEST,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )

            # Extract text robustly
            if hasattr(message, "content") and message.content:
                first_block = message.content[0]
                text = first_block.text if hasattr(first_block, "text") else None
                if not text and isinstance(first_block, dict):
                    text = first_block.get("text")
                if text:
                    return text
            # Fallback
            return message.model_dump_json(indent=2)
        except Exception as e:
            logger.error("Error calling Anthropic model: %s", str(e))
            return str(message) if "message" in locals() else str(e)


def get_anthropic_non_rag_response(
    config,
    prompt: str,
    model_choice: str,
    temperature: float = 0.6,
    max_tokens: int = 4096,
) -> str:
    """Get a response from Anthropic Vertex model without using RAG context."""
    try:
        model_mapping = {
            "Claude Sonnet 4": config.anthropic.model_sonnet,
        }
        model_name = model_mapping.get(model_choice)
        if not model_name:
            raise ValueError(
                f"Invalid Anthropic model choice: {model_choice}. Available: {list(model_mapping.keys())}"
            )

        # Initialize Vertex AI and client
        import vertexai

        vertexai.init(
            project=config.anthropic.project, location=config.anthropic.location
        )
        client = AnthropicVertex(
            region=config.anthropic.location, project_id=config.anthropic.project
        )

        system_prompt = config.chatbot.system_prompt_plain_llm + "\nUSER QUERY:\n"
        full_prompt = system_prompt + prompt

        message = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=temperature,
        )

        # Extract text
        if hasattr(message, "content") and message.content:
            first_block = message.content[0]
            text = first_block.text if hasattr(first_block, "text") else None
            if not text and isinstance(first_block, dict):
                text = first_block.get("text")
            if text:
                return text.strip()
        return message.model_dump_json(indent=2)

    except Exception as e:
        logging.error(
            "Error in get_anthropic_non_rag_response: %s", str(e), exc_info=True
        )
        return f"Error generating response: {str(e)}"
