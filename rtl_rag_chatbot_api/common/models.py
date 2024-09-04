from typing import Optional

from pydantic import BaseModel, Field


class Query(BaseModel):
    text: str
    file_id: str
    model_choice: str = Field(..., description="The chosen language model")
    model_config = {"protected_namespaces": ()}


class PreprocessRequest(BaseModel):
    file_id: str
    is_image: bool


class FileUploadResponse(BaseModel):
    message: str
    file_id: str
    original_filename: str
    is_image: bool


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
    model_choice: str = Field(
        ..., description="The chosen language model for creating embeddings"
    )
    is_image: bool = Field(..., description="Whether the file is an image or not")
