import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette_exporter import PrometheusMiddleware, handle_metrics

from configs.app_config import Config
from rag_pdf_api.chatbot.chatbot_creator import setup_chatbot
from rag_pdf_api.chatbot.gcs_handler import GCSHandler
from rag_pdf_api.common.embeddings import run_preprocessor

configs = Config()
#chatbot, timestamp = setup_chatbot(configs)


class Query(BaseModel):
    text: str


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
async def preprocess():
    bucket_name = "chatbotui"
    folder_path = "pdfs-raw"
    specific_folder = "2bf2c97f-a40f"
    destination_file_path = "local_data/"

    gcs_handler = GCSHandler(configs)
    
    try:
        folder_found = gcs_handler.check_and_download_folder(
            bucket_name, folder_path, specific_folder, destination_file_path
        )

        if folder_found:
            # TODO: Add any additional processing steps here
            run_preprocessor(configs=configs, text_data_folder_path="./local_data",specific_folder=specific_folder)
            return {"status": "Files downloaded successfully", "folder": specific_folder}
        else:
            raise HTTPException(status_code=404, detail=f"Folder {specific_folder} not found in {folder_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")  
   
@app.post("/pdf/chat")
async def chat(query: Query):
    try:
        response = chatbot.get_llm_answer(query.text)

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("rag_pdf_api.app:app", host="0.0.0.0", port=8080, reload=False)
