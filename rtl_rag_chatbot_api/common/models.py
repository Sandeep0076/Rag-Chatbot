from pydantic import BaseModel, Field


class Query(BaseModel):
    text: str
    file_id: str
    model_choice: str = Field(..., description="The chosen language model")
    model_config = {"protected_namespaces": ()}


class PreprocessRequest(BaseModel):
    file_id: str
    contain_multimedia: bool


class FileUploadResponse(BaseModel):
    message: str
    file_id: str
    original_filename: str
    contain_multimedia: bool
