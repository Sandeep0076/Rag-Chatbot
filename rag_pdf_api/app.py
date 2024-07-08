import os

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette_exporter import PrometheusMiddleware, handle_metrics

from configs.app_config import Config
from rag_pdf_api.chatbot.chatbot_creator import Chatbot
from rag_pdf_api.chatbot.gcs_handler import GCSHandler
from rag_pdf_api.common.embeddings import run_preprocessor

configs = Config()
gcs_handler = GCSHandler(configs)


class Query(BaseModel):
    text: str
    file_id: str


class PreprocessRequest(BaseModel):
    file_id: str


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
    bucket_name = "chatbotui"
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


# TODO: when we call directly for chat, if the folder is not present,
# it download only a folder, not whole
@app.post("/pdf/chat")
async def chat(query: Query):
    """"""
    try:
        file_id = query.file_id
        chroma_db_path = f"./chroma_db/{file_id}"

        if not os.path.exists(chroma_db_path):
            print(f"Chroma DB path not found: {chroma_db_path}")
            # If not found locally, download from GCS
            try:
                gcs_handler.download_files_from_folder_by_id(file_id)
            except FileNotFoundError as e:
                raise HTTPException(status_code=404, detail=str(e))

        # Setup chatbot with the specific file_id
        try:
            chatbot = Chatbot(configs, file_id)
            print("Chatbot setup successful")
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


def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("rag_pdf_api.app:app", host="0.0.0.0", port=8080, reload=False)
