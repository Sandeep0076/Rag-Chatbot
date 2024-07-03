from typing import List, Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class AzureLLMConfig(BaseModel):
    azure_llm_api_key: str
    azure_llm_endpoint: str
    azure_llm_api_version: str
    azure_llm_deployment: str
    azure_llm_model_name: str


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


class Config(BaseSettings):
    azure_embedding: AzureEmbeddingConfig
    azure_llm: AzureLLMConfig
    chatbot: ChatbotConfig
    llm_hyperparams: LLMHyperParams
    gcp_resource: GCPResourceConfig
