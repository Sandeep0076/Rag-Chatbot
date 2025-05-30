from pydantic import BaseModel
from pydantic_settings import BaseSettings


class GCSConfig(BaseModel):
    gcp_project: str
    bucket_name: str
    embeddings_root_folder: str


class WorkflowConfig(BaseModel):
    db_deleted_user_id: str


class Config(BaseSettings):
    gcp: GCSConfig
    workflow: WorkflowConfig


config = Config()
