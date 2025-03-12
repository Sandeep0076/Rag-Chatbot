from pydantic import BaseModel
from pydantic_settings import BaseSettings


class GCSConfig(BaseModel):
    gcp_project: str
    bucket_name: str
    embeddings_root_folder: str


class Config(BaseSettings):
    gcp: GCSConfig


config = Config()
