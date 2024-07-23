import os
import logging
import uvicorn
import shutil
from fastapi import FastAPI, HTTPException
from starlette_exporter import PrometheusMiddleware, handle_metrics
from configs.app_config import Config
from rtl_rag_chatbot_api.common.models import Query, PreprocessRequest
from rtl_rag_chatbot_api.chatbot.chatbot_creator import Chatbot
from rtl_rag_chatbot_api.chatbot.gcs_handler import GCSHandler
from rtl_rag_chatbot_api.common.embeddings import run_preprocessor
import chromadb
from chromadb.config import Settings
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse
from rtl_rag_chatbot_api.common.encryption_utils import encrypt_file
import uuid
"""
Main FastAPI application for the RAG PDF API.
"""

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


@app.post("/file/preprocess")
async def preprocess(request: PreprocessRequest):
    file_id = request.file_id
    chroma_db_path = f"./chroma_db/{file_id}"
    bucket_name = configs.gcp_resource.bucket_name
    embeddings_folder = "file-embeddings"
    raw_files_folder = "files-raw"
    destination_file_path = "local_data/"

    try:
        # 1. Check if embeddings exist locally
        if os.path.exists(chroma_db_path):
            logging.info(f"Embeddings are ready {file_id}")
            return {"status": "Embeddings are ready", "folder": file_id}

        # 2. Check if embeddings exist in GCS
        embeddings_prefix = f"{embeddings_folder}/{file_id}/"
        embeddings_blobs = list(gcs_handler.bucket.list_blobs(prefix=embeddings_prefix))

        if embeddings_blobs:
            # 3. Download embeddings from GCS
            logging.info(f"Downloading embeddings for {file_id} from GCS")
            gcs_handler.download_files_from_folder_by_id(file_id)
            logging.info(f"Embeddings downloaded for {file_id}")
            return {"status": "Embeddings downloaded from GCS", "folder": file_id}

        # 4. If no embeddings, process the raw files
        logging.info(f"No embeddings found for {file_id}. Processing raw files.")

        # Check for raw files in GCS
        raw_files_found, raw_file_paths = gcs_handler.check_and_download_folder(
            bucket_name, raw_files_folder, file_id, destination_file_path
        )

        if not raw_files_found:
            raise HTTPException(
                status_code=404,
                detail=f"No raw files found for {file_id}"
            )

        # Ensure the Chroma DB directory exists and has write permissions
        os.makedirs(chroma_db_path, exist_ok=True)
        os.chmod(chroma_db_path, 0o755)  # Ensure write permissions

        # Initialize Chroma DB with proper permissions
        db = chromadb.PersistentClient(
            path=chroma_db_path,
            settings=Settings(allow_reset=True, is_persistent=True)
        )

        # Run preprocessor
        run_preprocessor(
            configs=configs,
            text_data_folder_path=destination_file_path,
            file_id=file_id,
            chroma_db_path=chroma_db_path,
            chroma_db=db
        )

        # Clean up the downloaded files after preprocessing
        for file_path in raw_file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)

        # 5. Upload new embeddings to GCS
        #gcs_handler.upload_embeddings_to_gcs(chroma_db_path, file_id)

        return {
            "status": "Files processed and embeddings created successfully",
            "folder": file_id,
        }

    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")



@app.post("/file/chat")
async def chat(query: Query):
    """
    Endpoint to chat with the RAG model. Downloads the embeddings from Bucket and
    answers all the questions related to PDF.
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
        raise HTTPException(status_code=500, detail=f"An error occurred during cleanup: {str(e)}")

@app.post("/file/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        original_filename = file.filename
        file_extension = os.path.splitext(original_filename)[1]
        encrypted_filename = f"{file_id}{file_extension}.encrypted"

        # Create a temporary file to store the uploaded content
        temp_file_path = f"temp_{file_id}{file_extension}"
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Encrypt the file
        encrypted_file_path = encrypt_file(temp_file_path)

        # Upload the encrypted file to GCS
        gcs_handler = GCSHandler(configs)
        bucket_name = configs.gcp_resource.bucket_name
        destination_blob_name = f"files-raw/{file_id}/{encrypted_filename}"
        
        gcs_handler.upload_file_to_gcs(bucket_name, encrypted_file_path, destination_blob_name)

        # Clean up temporary files
        os.remove(temp_file_path)
        os.remove(encrypted_file_path)

        return JSONResponse(content={
            "message": "File uploaded and encrypted successfully",
            "file_id": file_id,
            "original_filename": original_filename
        }, status_code=200)

    except Exception as e:
        return JSONResponse(content={
            "message": f"An error occurred: {str(e)}"
        }, status_code=500)

def start():
    """
    Function to start the FastAPI application.
    Launched with `poetry run start` at root level
    """
    uvicorn.run("rtl_rag_chatbot_api.app:app", host="0.0.0.0", port=8080, reload=False)