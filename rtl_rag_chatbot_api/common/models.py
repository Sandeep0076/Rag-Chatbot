from pydantic import BaseModel


class Query(BaseModel):
    text: str
    file_id: str
    model_choice: str = "gpt-3.5-turbo"
    model_config = {"protected_namespaces": ()}


class PreprocessRequest(BaseModel):
    file_id: str
