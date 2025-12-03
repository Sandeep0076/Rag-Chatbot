import uuid
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from rtl_rag_chatbot_api.common.errors import BaseAppError, ErrorRegistry


class Query(BaseModel):
    text: List[str]
    file_id: Optional[str] = None  # Keep for single file, now optional
    file_ids: Optional[List[str]] = None  # For multiple files
    model_choice: str = Field(..., description="The chosen language model")
    user_id: str
    session_id: str  # Mandatory session ID for tracking and isolation
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    custom_gpt: bool = False
    system_prompt: Optional[str] = None
    model_config = {"protected_namespaces": ()}


class PreprocessRequest(BaseModel):
    file_id: str
    is_image: bool


class FileUploadResponse(BaseModel):
    message: str
    file_id: Optional[str] = None  # Optional to support multi-file mode
    file_ids: Optional[List[str]] = None  # For multiple file uploads
    original_filename: Optional[str] = None  # Optional for multi-file mode
    original_filenames: Optional[List[str]] = None  # For multiple filenames
    is_image: bool = False
    is_tabular: bool = False
    status: str = "success"  # Added status field with default value
    temp_file_path: Optional[str] = None
    multi_file_mode: bool = False  # Flag to indicate multiple file processing
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )  # Unique session identifier for tracking upload groups


class NeighborsQuery(BaseModel):
    text: str
    file_id: str
    n_neighbors: int = 3


class ModelInitRequest(BaseModel):
    model_choice: str = Field(
        ..., description="The chosen language model (including Gemini)"
    )
    file_id: Optional[str] = None
    # model_config = {"protected_namespaces": ()}


class ImageAnalysisUpload(BaseModel):
    file_id: str
    original_filename: str
    is_image: bool
    analysis: str


class EmbeddingCreationRequest(BaseModel):
    file_id: str
    is_image: bool = Field(..., description="Whether the file is an image or not")


class EmbeddingsCheckRequest(BaseModel):
    file_ids: List[str] = Field(
        ..., description="List of file IDs to check (can be single item for one file)"
    )
    model_choice: str = Field(..., description="The chosen language model")

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure file_ids is not empty
        if not self.file_ids:
            raise BaseAppError(
                ErrorRegistry.ERROR_BAD_REQUEST,
                "file_ids must contain at least one file ID",
                details={"file_ids": self.file_ids},
            )


class ChatRequest(BaseModel):
    # Support both simple chat and Custom GPT mode
    model: Optional[str] = None  # Simple mode: model name
    message: Optional[str] = None  # Simple mode: single message

    # Custom GPT mode fields (alternative to model/message)
    text: Optional[List[str]] = None  # Custom GPT: conversation history
    model_choice: Optional[str] = None  # Custom GPT: model name
    user_id: Optional[str] = None  # Custom GPT: user identifier
    session_id: Optional[str] = None  # Custom GPT: session tracking

    # Common fields
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)

    # Custom GPT specific
    custom_gpt: bool = False
    system_prompt: Optional[str] = None
    generate_visualization: bool = False
    step_name: Optional[str] = None  # Identifies Custom GPT creation step for logging

    def __init__(self, **data):
        super().__init__(**data)
        # Validate: either (model + message) OR (text + model_choice) must be provided
        has_simple = bool(self.model) and bool(self.message)
        has_custom_gpt = bool(self.text) and bool(self.model_choice)

        if not has_simple and not has_custom_gpt:
            raise BaseAppError(
                ErrorRegistry.ERROR_BAD_REQUEST,
                "Either (model + message) for simple chat OR (text + model_choice) for Custom GPT must be provided",
                details={
                    "custom_gpt": self.custom_gpt,
                    "has_model": bool(self.model),
                    "has_message": bool(self.message),
                    "has_text": bool(self.text),
                    "has_model_choice": bool(self.model_choice),
                },
            )


class ImageGenerationRequest(BaseModel):
    prompt: List[str] = Field(
        ...,
        description="Array of prompts for context-aware generation. "
        "Format: ['prompt1', 'prompt2', 'current_prompt']. "
        "Last element is the current prompt, previous elements are history. "
        "Similar to chat text array.",
    )
    size: str = Field(
        default="1024x1024", description="Size of the generated image (e.g., 1024x1024)"
    )
    n: int = Field(default=1, description="Number of images to generate")
    model_choice: Optional[str] = Field(
        default=None,
        description="Model to use for image generation (dall-e-3, imagen-3.0, or NanoBanana)",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for tracking conversation context across image generations",
    )
    reference_image_file_id: Optional[str] = Field(
        default=None,
        description="File ID of the reference image (for tracking in frontend database)",
    )
    input_image_base64: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Base64-encoded image data for image-to-image editing (NanoBanana only). "
        "Can be a single string (legacy) or list of strings (multi-image support). "
        "Format: 'data:image/png;base64,<data>' or raw base64 string. Max 10MB per image. "
        "Supports 1-3 images for multi-image editing.",
    )


class CleanupRequest(BaseModel):
    is_manual: bool = True


class FileDeleteRequest(BaseModel):
    file_ids: Union[str, List[str]]  # Can be single ID or list of IDs
    include_gcs: bool = False
    username: str


class DeleteRequest(BaseModel):
    file_ids: Union[str, List[str]]  # Can be single ID or list of IDs
    include_gcs: bool = False


class TitleGenerationRequest(BaseModel):
    conversation: List[str] = Field(
        ...,
        description="Array of strings alternating between user questions and assistant answers",
    )
    model_choice: Optional[str] = Field(
        default="gpt_4_1_nano",
        description="Model to use for title generation (defaults to gpt_4_1_nano)",
    )
    mode: Optional[str] = Field(
        default="text",
        description="Mode for title generation: 'text' for normal chat or 'image' for image generation",
    )
