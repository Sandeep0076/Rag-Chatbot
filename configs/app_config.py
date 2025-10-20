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


class AzureEmbedding3LargeConfig(BaseModel):
    azure_embedding_3_large_api_key: str
    azure_embedding_3_large_endpoint: str
    azure_embedding_3_large_api_version: str
    azure_embedding_3_large_deployment: str
    azure_embedding_3_large_model_name: str


class LLMHyperParams(BaseModel):
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    stop: Optional[List[str]]
    # Optional advanced params (can be provided via env: LLM_HYPERPARAMS__REASONING_EFFORT, LLM_HYPERPARAMS__VERBOSITY)
    reasoning_effort: Optional[str] = None
    verbosity: Optional[str] = None


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
    default_embedding_type: str = "azure-3-large"


class GCPResourceConfig(BaseModel):
    gcp_project: str
    bucket_name: str
    embeddings_folder: str
    gcp_embeddings_folder: str


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


class AnthropicConfig(BaseModel):
    model_sonnet: str = "claude-sonnet-4@20250514"
    model_sonnet_45: str = "claude-sonnet-4-5"
    project: str = "dat-itowe-dev"
    location: str = "europe-west1"
    model_config = {"protected_namespaces": ()}


class CleanupConfig(BaseModel):
    staleness_threshold_minutes: int = 240  # 4 hours default
    min_cleanup_interval: int = 30  # 30 minutes default
    cleanup_interval_minutes: int = 60  # 1 hour default


class Config(BaseSettings):
    azure_embedding: AzureEmbeddingConfig
    azure_embedding_3_large: AzureEmbedding3LargeConfig
    azure_llm: AzureLLMConfig
    chatbot: ChatbotConfig
    llm_hyperparams: LLMHyperParams
    gcp_resource: GCPResourceConfig
    gemini: GeminiConfig
    azure_dalle_3: AzureDalle3Config
    vertexai_imagen: VertexAIImagenConfig = Field(default_factory=VertexAIImagenConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    cleanup: CleanupConfig = Field(default_factory=lambda: CleanupConfig())

    # Feature flags
    use_file_hash_db: bool = Field(
        default=True,
        description="Use database for file hash lookup instead of GCS (recommended for performance)",
    )
    generate_visualization: bool = Field(
        default=True, description="Enable chart generation and visualization features"
    )
    save_extracted_text_diagnostic: bool = Field(
        default=False,
        description="Save extracted text to diagnostic files for debugging",
    )
    save_extracted_text_in_metadata: bool = Field(
        default=False,
        description="Save extracted text in file metadata to avoid duplicate extraction",
    )

    class Config:
        env_nested_delimiter = "__"
