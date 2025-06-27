from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    api_key: str
    endpoint: str
    deployment: str
    api_version: str
    model_name: str
    model_config = {"protected_namespaces": ()}


class AzureLLMConfig(BaseModel):
    models: Dict[str, ModelConfig] = Field(default_factory=dict)


class AzureEmbeddingConfig(BaseModel):
    azure_embedding_api_key: str
    azure_embedding_endpoint: str
    azure_embedding_api_version: str
    azure_embedding_deployment: str
    azure_embedding_model_name: str


class LLMHyperParams(BaseModel):
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    stop: Optional[List[str]]


class ChatbotConfig(BaseModel):
    system_prompt_plain_llm: str
    system_prompt_rag_llm: str
    vector_db_collection_name: str
    image_file_path: str
    title: str
    description: str
    info_text: str
    max_input_size: int
    num_outputs: int
    max_chunk_overlap: float
    chunk_size_limit: int
    n_neighbours: int


class GCPResourceConfig(BaseModel):
    gcp_project: str
    bucket_name: str
    embeddings_folder: str


class GeminiConfig(BaseModel):
    model_flash: str = "gemini-2.5-flash"
    model_pro: str = "gemini-2.5-pro"
    project: str
    location: str
    model_config = {"protected_namespaces": ()}


class AzureDalle3Config(BaseModel):
    api_key: str
    endpoint: str
    api_version: str
    model_name: str
    deployment: str
    model_config = {"protected_namespaces": ()}


class VertexAIImagenConfig(BaseModel):
    model_name: str = "imagen-3.0-generate-002"
    project: str = "dat-itowe-dev"
    location: str = "europe-west4"
    model_config = {"protected_namespaces": ()}


class CleanupConfig(BaseModel):
    staleness_threshold_minutes: int = 240  # 4 hours default
    min_cleanup_interval: int = 30  # 30 minutes default
    cleanup_interval_minutes: int = 60  # 1 hour default


class Config(BaseSettings):
    azure_embedding: AzureEmbeddingConfig
    azure_llm: AzureLLMConfig
    chatbot: ChatbotConfig
    llm_hyperparams: LLMHyperParams
    gcp_resource: GCPResourceConfig
    gemini: GeminiConfig
    azure_dalle_3: AzureDalle3Config
    vertexai_imagen: VertexAIImagenConfig = Field(default_factory=VertexAIImagenConfig)
    cleanup: CleanupConfig = Field(default_factory=lambda: CleanupConfig())

    class Config:
        env_nested_delimiter = "__"
