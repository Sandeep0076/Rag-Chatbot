import json
import logging
import os
import shutil
import uuid
from pathlib import Path

import chromadb
import uvicorn
from chromadb.config import Settings
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# from pydantic import BaseModel
from starlette_exporter import PrometheusMiddleware, handle_metrics

from configs.app_config import Config
from rtl_rag_chatbot_api.chatbot.chatbot_creator import Chatbot
from rtl_rag_chatbot_api.chatbot.gcs_handler import GCSHandler
from rtl_rag_chatbot_api.chatbot.gemini_handler import GeminiHandler
from rtl_rag_chatbot_api.chatbot.image_reader import analyze_images
from rtl_rag_chatbot_api.common.embeddings import run_preprocessor
from rtl_rag_chatbot_api.common.encryption_utils import encrypt_file
from rtl_rag_chatbot_api.common.models import (
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


@app.post("/file/chat")
async def chat(query: Query):
    try:
        file_id = query.file_id
        model_choice = query.model_choice.lower()

        if file_id not in initialized_models:
            raise HTTPException(
                status_code=404, detail="Model not initialized for this file"
            )

        model_info = initialized_models[file_id]

        if model_info["type"] == "gemini":
            if gemini_handler is None:
                raise HTTPException(
                    status_code=404, detail="Gemini handler not initialized"
                )
            response = gemini_handler.get_answer(query.text, file_id)
        else:
            response = model_info["model"].get_answer(query.text)
        logging.info(f"{model_choice} is used for chatting.")
        return {"response": response}
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        ) @ app.get("/available-models")


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


@app.post("/file/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    is_image: bool = Form(...),
    model_choice: str = Form(...),
):
    global gemini_handler, initialized_models
    try:
        original_filename = file.filename
        logging.info(
            f"Received file for upload: {original_filename}, model: {model_choice}"
        )

        existing_file_id = gcs_handler.find_existing_file(original_filename)
        if existing_file_id:
            logging.info(
                f"File {original_filename} already exists with ID: {existing_file_id}"
            )
            chroma_db_path = f"./chroma_db/{existing_file_id}"
            os.makedirs(chroma_db_path, exist_ok=True)

            try:
                gcs_handler.download_files_from_folder_by_id(existing_file_id)
                logging.info(
                    f"Embeddings downloaded for existing file: {existing_file_id}"
                )

                return FileUploadResponse(
                    message="File already exists. Embeddings downloaded.",
                    file_id=existing_file_id,
                    original_filename=original_filename,
                    is_image=is_image,
                )
            except Exception as e:
                logging.error(f"Error downloading embeddings: {str(e)}")
                raise HTTPException(
                    status_code=500, detail=f"Error processing existing file: {str(e)}"
                )

        # If the file doesn't exist, proceed with the new file upload process
        file_id = str(uuid.uuid4())

        logging.info(f"File {original_filename} is new. Processing....")
        file_extension = os.path.splitext(original_filename)[1]
        encrypted_filename = f"{original_filename}.encrypted"

        temp_file_path = f"temp_{file_id}_{file_extension}"

        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        encrypted_file_path = encrypt_file(temp_file_path)

        raw_files_folder = "files-raw"
        destination_blob_name = f"{raw_files_folder}/{file_id}/{encrypted_filename}"
        gcs_handler.upload_to_gcs(
            configs.gcp_resource.bucket_name,
            {
                "file": (encrypted_file_path, destination_blob_name),
                "metadata": (
                    {"is_image": is_image, "model_choice": model_choice},
                    f"{raw_files_folder}/{file_id}/metadata.json",
                ),
            },
        )

        logging.info(f"File encrypted and uploaded to GCS: {destination_blob_name}")

        os.remove(temp_file_path)
        os.remove(encrypted_file_path)

        chroma_db_path = f"./chroma_db/{file_id}"
        os.makedirs(chroma_db_path, exist_ok=True)
        os.chmod(chroma_db_path, 0o755)

        destination_file_path = f"local_data/{file_id}/"
        os.makedirs(destination_file_path, exist_ok=True)

        analysis_json_path = None
        decrypted_file_path = None
        try:
            decrypted_file_path = gcs_handler.download_and_decrypt_file(
                file_id, destination_file_path
            )
            logging.info(f"File decrypted: {decrypted_file_path}")

            # Verify that the file is readable
            with open(decrypted_file_path, "rb") as test_file:
                test_content = test_file.read(1024)  # Read first 1KB to test
            if not test_content:
                raise IOError(
                    f"Decrypted file at {decrypted_file_path} appears to be empty or unreadable"
                )

            # logging.info(f"Successfully verified decrypted file: {decrypted_file_path}")

            if is_image:
                logging.info("Processing image file...")
                image_analysis_result = analyze_images(decrypted_file_path)
                if os.path.exists(decrypted_file_path):
                    os.remove(decrypted_file_path)

                analysis_json_path = os.path.join(
                    destination_file_path, f"{file_id}_analysis.json"
                )
                with open(analysis_json_path, "w") as f:
                    json.dump(image_analysis_result, f)

            # Initialize the selected model before processing
            if model_choice.lower() in ["gemini-flash", "gemini-pro"]:
                if gemini_handler is None:
                    gemini_handler = GeminiHandler(configs, gcs_handler)
                gemini_model = (
                    configs.gemini.model_flash
                    if model_choice.lower() == "gemini-flash"
                    else configs.gemini.model_pro
                )
                gemini_handler.initialize(model=gemini_model)
                initialized_models[file_id] = {
                    "type": "gemini",
                    "model": gemini_handler,
                }
            else:
                # Initialize Azure OpenAI model
                azure_model = Chatbot(
                    configs, file_id=file_id, model_choice=model_choice
                )
                initialized_models[file_id] = {"type": "azure", "model": azure_model}

            # Use the initialized model based on the choice
            if model_choice.lower() in ["gemini-flash", "gemini-pro"]:
                gemini_handler.process_file(file_id, decrypted_file_path)
                gemini_handler.upload_embeddings_to_gcs(file_id)
                logging.info(f"Gemini embeddings uploaded to GCP for file: {file_id}")
            else:
                run_preprocessor(
                    configs=configs,
                    text_data_folder_path=destination_file_path,
                    file_id=file_id,
                    chroma_db_path=chroma_db_path,
                    chroma_db=chromadb.PersistentClient(
                        path=chroma_db_path,
                        settings=Settings(allow_reset=True, is_persistent=True),
                    ),
                    is_image=is_image,
                    gcs_handler=gcs_handler,
                )
                logging.info(f"Azure embeddings uploaded to GCP for file: {file_id}")

        except Exception as e:
            logging.error(f"Error during file processing: {str(e)}")
            raise
        finally:
            if decrypted_file_path and os.path.exists(decrypted_file_path):
                os.remove(decrypted_file_path)
            if analysis_json_path and os.path.exists(analysis_json_path):
                os.remove(analysis_json_path)
            shutil.rmtree(destination_file_path, ignore_errors=True)

        return FileUploadResponse(
            message="File uploaded, encrypted, and preprocessed successfully",
            file_id=file_id,
            original_filename=original_filename,
            is_image=is_image,
        )

    except Exception as e:
        logging.error(
            f"Error in file upload and preprocessing: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"An error occurred during file upload: {str(e)}"
        )


@app.post("/model/initialize")
async def initialize_model(request: ModelInitRequest):
    global initialized_azure_model, gemini_handler
    try:
        model_choice = request.model_choice.lower()

        if model_choice in ["gemini-flash", "gemini-pro"]:
            gemini_model = (
                configs.gemini.model_flash
                if model_choice == "gemini-flash"
                else configs.gemini.model_pro
            )
            gemini_handler = GeminiHandler(configs, gcs_handler)
            gemini_handler.initialize(model=gemini_model)
        else:
            # Initialize Azure OpenAI model
            initialized_azure_model = Chatbot(
                configs, file_id=request.file_id, model_choice=model_choice
            )

        return {"message": f"Model {model_choice} initialized successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error initializing model: {str(e)}"
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
        file_id = query.file_id

        if file_id not in initialized_models:
            raise HTTPException(
                status_code=404, detail="Model not initialized for this file"
            )

        model_info = initialized_models[file_id]

        if model_info["type"] == "gemini":
            if gemini_handler is None:
                raise HTTPException(
                    status_code=404, detail="Gemini handler not initialized"
                )
            neighbors = gemini_handler.get_n_nearest_neighbours(
                query.text, file_id, query.n_neighbors
            )
        else:
            chatbot = model_info["model"]
            neighbors_with_metadata = chatbot.get_n_nearest_neighbours(
                query.text, n_neighbours=query.n_neighbors
            )
            neighbors = [neighbor.node.text for neighbor in neighbors_with_metadata]

        return {"neighbors": neighbors}
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Unexpected error in neighbors endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


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
