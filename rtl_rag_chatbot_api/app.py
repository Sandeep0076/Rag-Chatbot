import json
import logging
import os

# import shutil
import uuid
from pathlib import Path

# import chromadb
import uvicorn

# from chromadb.config import Settings
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette_exporter import PrometheusMiddleware, handle_metrics

from configs.app_config import Config
from rtl_rag_chatbot_api.chatbot.chatbot_creator import Chatbot
from rtl_rag_chatbot_api.chatbot.embedding_handler import EmbeddingHandler
from rtl_rag_chatbot_api.chatbot.file_handler import FileHandler
from rtl_rag_chatbot_api.chatbot.gcs_handler import GCSHandler
from rtl_rag_chatbot_api.chatbot.gemini_handler import GeminiHandler
from rtl_rag_chatbot_api.chatbot.image_reader import analyze_images
from rtl_rag_chatbot_api.chatbot.model_handler import ModelHandler
from rtl_rag_chatbot_api.common.models import (
    EmbeddingCreationRequest,
    FileUploadResponse,
    ModelInitRequest,
    NeighborsQuery,
    Query,
)

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


"""
Main FastAPI application for the RAG PDF API.
"""
logging.basicConfig(level=logging.INFO)

configs = Config()
gcs_handler = GCSHandler(configs)
file_handler = FileHandler(configs, gcs_handler)
model_handler = ModelHandler(configs, gcs_handler)
embedding_handler = EmbeddingHandler(configs, gcs_handler)

app = FastAPI()
global gemini_handler
gemini_handler = None
# Global dictionary to store initialized chatbots
# depreacted
initialized_chatbots = {}

initialized_models = {}

# Global variables to store initialized models
initialized_azure_model = None
initialized_gemini_model = None

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.post("/file/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    is_image: bool = Form(...),
):
    try:
        file_id = str(uuid.uuid4())
        original_filename = file.filename

        result = await file_handler.process_file(file, file_id, is_image)

        if result["status"] == "existing":
            if file_handler.download_existing_file(result["file_id"]):
                message = "File already exists. Embeddings downloaded."
            else:
                message = "File already exists, but error downloading embeddings."
        else:
            message = "File uploaded, encrypted, and processed successfully"

        return FileUploadResponse(
            message=message,
            file_id=result["file_id"],
            original_filename=original_filename,
            is_image=is_image,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/initialize")
async def initialize_model(request: ModelInitRequest):
    try:
        file_info = embedding_handler.get_embeddings_info(request.file_id)
        if not file_info:
            raise HTTPException(
                status_code=404, detail="Embeddings not found for this file"
            )

        if request.model_choice.lower() in ["gemini-flash", "gemini-pro"]:
            embedding_type = "gemini"
        else:
            embedding_type = "azure"

        model = model_handler.initialize_model(
            request.model_choice, request.file_id, embedding_type
        )
        initialized_models[request.file_id] = model
        return {"message": f"Model {request.model_choice} initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/file/chat")
async def chat(query: Query):
    try:
        if query.file_id not in initialized_models:
            raise HTTPException(
                status_code=404, detail="Model not initialized for this file"
            )

        model = initialized_models[query.file_id]
        logging.info(f"Model type: {type(model)}")

        if isinstance(model, dict):
            model = model["model"]
            logging.info(f"Model extracted from dict: {type(model)}")

        logging.info(f"Calling get_answer on model: {type(model)}")
        response = model.get_answer(query.text)

        return {"response": response}
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embeddings/create")
async def create_embeddings(request: EmbeddingCreationRequest):
    try:
        embedding_handler = EmbeddingHandler(configs, gcs_handler)

        if embedding_handler.embeddings_exist(request.file_id):
            embeddings_info = embedding_handler.get_embeddings_info(request.file_id)
            if embeddings_info:
                return {
                    "message": "Embeddings already exist for this file",
                    "info": embeddings_info,
                }
            else:
                return {"message": "Embeddings exist but info not found"}

        result = await embedding_handler.create_and_upload_embeddings(
            request.file_id, request.is_image
        )

        return result
    except Exception as e:
        logging.error(f"Error in create_embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/available-models")
async def get_available_models():
    azure_models = list(configs.azure_llm.models.keys())
    gemini_models = ["gemini-flash", "gemini-pro"]
    all_models = azure_models + gemini_models
    return {"models": all_models}


@app.post("/file/cleanup")
async def cleanup_files():
    """
    Endpoint to clean-up local files in chroma_db and local_data folders,
    as well as cache files in the project.
    """
    try:
        gcs_handler = GCSHandler(configs)
        gcs_handler.cleanup_local_files()
        return {"status": "Cleanup completed successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred during cleanup: {str(e)}"
        )


async def initialize_chatbot(file_id: str, model_choice: str):
    chroma_db_path = f"./chroma_db/{file_id}"
    if not os.path.exists(chroma_db_path):
        logging.error(f"Chroma DB path not found: {chroma_db_path}")
        try:
            gcs_handler.download_files_from_folder_by_id(file_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    try:
        chatbot = Chatbot(configs, file_id, model_choice=model_choice)
        logging.info(f"Chatbot setup successful for file_id: {file_id}")
        initialized_chatbots[file_id] = chatbot
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error setting up chatbot: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error setting up chatbot: {str(e)}"
        )


@app.post("/file/neighbors")
async def get_neighbors(query: NeighborsQuery):
    try:
        if query.file_id not in initialized_models:
            raise HTTPException(
                status_code=404, detail="Model not initialized for this file"
            )

        model = initialized_models[query.file_id]

        if isinstance(model, GeminiHandler):
            neighbors = model.get_n_nearest_neighbours(
                query.text, query.file_id, query.n_neighbors
            )
        else:  # Assuming it's a Chatbot instance for Azure models
            neighbors_with_metadata = model.get_n_nearest_neighbours(
                query.text, n_neighbours=query.n_neighbors
            )
            neighbors = [neighbor.node.text for neighbor in neighbors_with_metadata]

        return {"neighbors": neighbors}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/image/analyze")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    try:
        # Get the file extension from the original filename
        file_extension = Path(file.filename).suffix

        # Create a temporary file with the original extension
        temp_file_path = f"temp_{uuid.uuid4()}{file_extension}"

        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Analyze the image
        result = analyze_images(temp_file_path)

        # Generate a unique filename for the result
        result_filename = f"image_analysis_{uuid.uuid4()}.json"
        result_file_path = os.path.join("processed_data", result_filename)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(result_file_path), exist_ok=True)

        # Save the result to a JSON file
        with open(result_file_path, "w", encoding="utf-8") as f:
            json.dump({"analysis": result}, f, ensure_ascii=False, indent=4)

        # Clean up the temporary file
        os.remove(temp_file_path)

        return JSONResponse(
            content={
                "message": "Image analyzed successfully",
                "result_file": result_filename,
                "analysis": result,
            }
        )

    except Exception as e:
        logging.error(f"Error in image analysis: {str(e)}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(
            status_code=500, detail=f"An error occurred during image analysis: {str(e)}"
        )


def start():
    """
    Function to start the FastAPI application.
    Launched with `poetry run start` at root level
    Streamlit : streamlit run streamlit_app.py
    """
    uvicorn.run("rtl_rag_chatbot_api.app:app", host="0.0.0.0", port=8080, reload=False)
