from typing import List, Optional, Union

from pydantic import BaseModel, Field


class Query(BaseModel):
    text: List[str]
    file_id: str
    model_choice: str = Field(..., description="The chosen language model")
    user_id: str
    model_config = {"protected_namespaces": ()}


class PreprocessRequest(BaseModel):
    file_id: str
    is_image: bool


class FileUploadResponse(BaseModel):
    message: str
    file_id: str
    original_filename: str
    is_image: bool
    status: str = "success"  # Added status field with default value
    temp_file_path: Optional[str] = None


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
    file_id: str
    model_choice: str = Field(..., description="The chosen language model")


class FileDeleteRequest(BaseModel):
    file_ids: List[str]


class ChatRequest(BaseModel):
    model: str
    message: str


class CleanupRequest(BaseModel):
    is_manual: bool = True


class DeleteRequest(BaseModel):
    file_ids: Union[str, List[str]]  # Can be single ID or list of IDs
    include_gcs: bool = False
