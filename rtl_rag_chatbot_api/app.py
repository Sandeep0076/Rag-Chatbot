import os

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette_exporter import PrometheusMiddleware, handle_metrics

from configs.app_config import Config
from rtl_rag_chatbot_api.chatbot.chatbot_creator import Chatbot
from rtl_rag_chatbot_api.chatbot.gcs_handler import GCSHandler
from rtl_rag_chatbot_api.common.embeddings import run_preprocessor

"""
Main FastAPI application for the RAG PDF API.
"""


class Query(BaseModel):
    """
    Pydantic model for chat query requests.

    Attributes:
    text (str): The query text.
    file_id (str): The ID of the file to query against.
    model_choice (str): The model to use for the query (default: "gpt-3.5-turbo").
    """

    text: str
    file_id: str
    model_choice: str = "gpt-3.5-turbo"
    model_config = {"protected_namespaces": ()}


class PreprocessRequest(BaseModel):
    """
    Pydantic model for preprocessing requests.

    Attributes:
    file_id (str): The ID of the file to preprocess.
    """

    file_id: str


configs = Config()
gcs_handler = GCSHandler(configs)
app = FastAPI()

# expose prometheus metrics at /metrics rest endpoint
app.add_middleware(
    PrometheusMiddleware,
    group_paths=True,
    app_name="api",
    filter_unhandled_paths=True,
    # do not report on health and metrics endpoint in order to keep the
    # size of the metrics low
    skip_paths=["/health", "/metrics"],
)
app.add_route("/metrics", handle_metrics)


@app.get("/health")
async def health():
    """
    Shows application health information.
    In the future this could do some actual checks.
    """
    return {"status": "up"}


@app.get("/info")
async def info():
    """
    Displays arbitrary application info.
    This could be the service name and some build info like
    e.g. the commit hash or the build time.
    """
    return {
        "title": configs.chatbot.title,
        "description": configs.chatbot.description,
        "info_text": configs.chatbot.info_text,
    }


@app.post("/pdf/preprocess")
async def preprocess(request: PreprocessRequest):
    """
    Endpoint to preprocess a PDF file. Downloads the pdf from Bucket, creates embeddings
    and upload the generated embeddings to Bucket
    """
    # bucket_name = "chatbotui"
    """"""
    bucket_name = configs.gcp_resource.bucket_name
    folder_path = "pdfs-raw"
    file_id = request.file_id
    destination_file_path = "local_data/"

    try:
        folder_found = gcs_handler.check_and_download_folder(
            bucket_name, folder_path, file_id, destination_file_path
        )

        if folder_found:
            # Create a folder with file_id inside chroma_db
            chroma_db_path = f"./chroma_db/{file_id}"
            os.makedirs(chroma_db_path, exist_ok=True)

            run_preprocessor(
                configs=configs,
                text_data_folder_path="./local_data",
                file_id=file_id,
                chroma_db_path=chroma_db_path,
            )
            return {
                "status": "Files downloaded and processed successfully",
                "folder": file_id,
            }
        else:
            raise HTTPException(
                status_code=404, detail=f"Folder {file_id} not found in {folder_path}"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/pdf/chat")
async def chat(query: Query):
    """
    Endpoint to chat with the RAG model. Downloads the embeddings from Bucket and
    answers all the questions realted to PDF.
    """
    try:
        file_id = query.file_id
        chroma_db_path = f"./chroma_db/{file_id}"

        if not os.path.exists(chroma_db_path):
            print(f"Chroma DB path not found: {chroma_db_path}")
            try:
                gcs_handler.download_files_from_folder_by_id(file_id)
            except FileNotFoundError as e:
                raise HTTPException(status_code=404, detail=str(e))

        try:
            chatbot = Chatbot(configs, file_id, model_choice=query.model_choice)
            print("Chatbot setup successful")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            print(f"Error setting up chatbot: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error setting up chatbot: {str(e)}"
            )

        try:
            response = chatbot.get_answer(query.text)
            return {"response": response}
        except Exception as e:
            print(f"Error getting LLM answer: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error getting LLM answer: {str(e)}"
            )
    except Exception as e:
        print(f"Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/available-models")
async def get_available_models():
    """
    Endpoint to get the list of available models.
    """
    return {
        "models": [
            "gpt_3_5_turbo",
            "gpt_4",
            # Add other available models here
        ]
    }


def start():
    """
    Function to start the FastAPI application.
    Launched with `poetry run start` at root level
    """
    uvicorn.run("rtl_rag_chatbot_api.app:app", host="0.0.0.0", port=8080, reload=False)
