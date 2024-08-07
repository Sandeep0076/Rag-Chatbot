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
from pydantic import BaseModel
from starlette_exporter import PrometheusMiddleware, handle_metrics

from configs.app_config import Config
from rtl_rag_chatbot_api.chatbot.chatbot_creator import Chatbot
from rtl_rag_chatbot_api.chatbot.gcs_handler import GCSHandler
from rtl_rag_chatbot_api.chatbot.image_reader import analyze_images
from rtl_rag_chatbot_api.common.embeddings import run_preprocessor
from rtl_rag_chatbot_api.common.encryption_utils import encrypt_file
from rtl_rag_chatbot_api.common.models import (
    FileUploadResponse,
    NeighborsQuery,
    PreprocessRequest,
    Query,
)


class ImageAnalysisUpload(BaseModel):
    file_id: str
    original_filename: str
    is_image: bool
    analysis: str


"""
Main FastAPI application for the RAG PDF API.
"""
logging.basicConfig(level=logging.INFO)

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


# Deprecated
@app.post("/file/preprocess")
async def preprocess(request: PreprocessRequest):
    file_id = request.file_id
    chroma_db_path = f"./chroma_db/{file_id}"
    bucket_name = configs.gcp_resource.bucket_name
    embeddings_folder = "file-embeddings"
    raw_files_folder = "files-raw"
    destination_file_path = "local_data/"

    try:
        is_image = False
        if os.path.exists(chroma_db_path):
            logging.info(f"Embeddings are ready {file_id}")
            # Load the metadata to check if it includes is_image
            metadata_path = os.path.join(chroma_db_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                is_image = metadata.get("is_image", False)
            else:
                is_image = False
            return {
                "status": "Embeddings are ready",
                "folder": file_id,
                "is_image": is_image,
            }

        embeddings_prefix = f"{embeddings_folder}/{file_id}/"
        embeddings_blobs = list(gcs_handler.bucket.list_blobs(prefix=embeddings_prefix))

        if embeddings_blobs:
            logging.info(f"Downloading embeddings for {file_id} from GCS")
            gcs_handler.download_files_from_folder_by_id(file_id)
            logging.info(f"Embeddings downloaded for {file_id}")
            # Load the metadata to get is_image info
            metadata_path = os.path.join(chroma_db_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                is_image = metadata.get("is_image", False)
            else:
                is_image = False
            return {
                "status": "Embeddings downloaded from GCS",
                "folder": file_id,
                "is_image": is_image,
            }

        logging.info(f"No embeddings found for {file_id}. Processing raw files.")

        raw_files_found, raw_file_paths = gcs_handler.check_and_download_folder(
            bucket_name, raw_files_folder, file_id, destination_file_path
        )

        if not raw_files_found:
            raise HTTPException(
                status_code=404, detail=f"No raw files found for {file_id}"
            )

        os.makedirs(chroma_db_path, exist_ok=True)
        os.chmod(chroma_db_path, 0o755)

        db = chromadb.PersistentClient(
            path=chroma_db_path, settings=Settings(allow_reset=True, is_persistent=True)
        )

        metadata_path = f"{raw_files_folder}/{file_id}/metadata.json"
        local_metadata_path = os.path.join(destination_file_path, "metadata.json")
        gcs_handler.download_files_from_gcs(
            bucket_name, metadata_path, local_metadata_path
        )

        with open(local_metadata_path, "r") as f:
            metadata = json.load(f)

        if is_image:
            pass
            # will implement this in seperate container and call api here
        else:
            run_preprocessor(
                configs=configs,
                text_data_folder_path=destination_file_path,
                file_id=file_id,
                chroma_db_path=chroma_db_path,
                chroma_db=db,
                is_image=is_image,
                gcs_handler=gcs_handler,
            )

        for file_path in raw_file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)

        return {
            "status": "Files processed and embeddings created successfully",
            "folder": file_id,
            "is_image": is_image,
        }

    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/file/chat")
async def chat(query: Query):
    try:
        file_id = query.file_id

        if file_id not in initialized_chatbots:
            raise HTTPException(
                status_code=404, detail="Chatbot not initialized for this file"
            )

        chatbot = initialized_chatbots[file_id]

        response = chatbot.get_answer(query.text)
        """
        # Check if the response includes a request to generate a chart or graph
        if (
            "generate chart" in query.text.lower()
            or "generate graph" in query.text.lower()
        ):
            chart_data = chatbot.generate_chart(query.text)
            return {"response": response, "chart_data": chart_data}
        """
        return {"response": response}
    except Exception as e:
        print(f"Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/available-models")
async def get_available_models():
    """
    Endpoint to get the list of available models.
    """
    return {"models": list(configs.azure_llm.models.keys())}


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
async def upload_file(file: UploadFile = File(...), is_image: bool = Form(...)):
    try:
        # Following logic, check if the file is already uploaded.
        # If finds the file, it download the embeddings.
        original_filename = file.filename
        logging.info(f"Received file for upload: {original_filename}")

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

                # Initialize the chatbot for the existing file
                await initialize_chatbot(
                    existing_file_id, "gpt_3_5_turbo"
                )  # Use a default model
                logging.info(
                    f"Chatbot initialized for existing file: {existing_file_id}"
                )

                return FileUploadResponse(
                    message="File already exists. Embeddings downloaded and chatbot initialized.",
                    file_id=existing_file_id,
                    original_filename=original_filename,
                    is_image=is_image,
                )
            except Exception as e:
                logging.error(
                    f"Error downloading embeddings or initializing chatbot: {str(e)}"
                )
                raise HTTPException(
                    status_code=500, detail="Error processing existing file"
                )

        logging.info(
            f"File {original_filename} is new. Proceeding with upload and processing."
        )
        file_id = str(uuid.uuid4())
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
                    {"is_image": is_image},
                    f"{raw_files_folder}/{file_id}/metadata.json",
                ),
            },
        )

        logging.info(f"File uploaded to GCS: {destination_blob_name}")

        os.remove(temp_file_path)
        os.remove(encrypted_file_path)

        chroma_db_path = f"./chroma_db/{file_id}"
        os.makedirs(chroma_db_path, exist_ok=True)
        os.chmod(chroma_db_path, 0o755)

        db = chromadb.PersistentClient(
            path=chroma_db_path, settings=Settings(allow_reset=True, is_persistent=True)
        )

        destination_file_path = f"local_data/{file_id}/"
        os.makedirs(destination_file_path, exist_ok=True)

        analysis_json_path = None
        decrypted_file_path = None
        try:
            decrypted_file_path = gcs_handler.download_and_decrypt_file(
                file_id, destination_file_path
            )
            logging.info(f"File decrypted: {decrypted_file_path}")

            # Check if the file is an image or PDF
            if is_image:
                logging.info("Processing image file: ..")
                image_analysis_result = analyze_images(decrypted_file_path)
                if os.path.exists(decrypted_file_path):
                    os.remove(decrypted_file_path)

                # Save the analysis result to a JSON file
                analysis_json_path = os.path.join(
                    destination_file_path, f"{file_id}_analysis.json"
                )
                with open(analysis_json_path, "w") as f:
                    json.dump(image_analysis_result, f)

            # Run preprocessor with the analysis JSON file
            run_preprocessor(
                configs=configs,
                text_data_folder_path=destination_file_path,
                file_id=file_id,
                chroma_db_path=chroma_db_path,
                chroma_db=db,
                is_image=is_image,
                gcs_handler=gcs_handler,
            )
            logging.info(f"Processing completed for file: {file_id}")

            await initialize_chatbot(file_id, "gpt_3_5_turbo")
            logging.info(f"Chatbot initialized for new file: {file_id}")
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
            message="File uploaded, encrypted, preprocessed, and chatbot initialized successfully",
            file_id=file_id,
            original_filename=original_filename,
            is_image=is_image,
        )

    except Exception as e:
        logging.error(f"Error in file upload and preprocessing: {str(e)}")
        return JSONResponse(
            content={"message": f"An error occurred: {str(e)}"}, status_code=500
        )


# Global dictionary to store initialized chatbots
initialized_chatbots = {}


async def initialize_chatbot(file_id: str, model_choice: str):
    chroma_db_path = f"./chroma_db/{file_id}"
    if not os.path.exists(chroma_db_path):
        print(f"Chroma DB path not found: {chroma_db_path}")
        try:
            gcs_handler.download_files_from_folder_by_id(file_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    try:
        chatbot = Chatbot(configs, file_id, model_choice=model_choice)
        print("Chatbot setup successful")
        initialized_chatbots[file_id] = chatbot
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error setting up chatbot: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error setting up chatbot: {str(e)}"
        )


@app.post("/file/neighbors")
async def get_neighbors(query: NeighborsQuery):
    try:
        file_id = query.file_id

        if file_id not in initialized_chatbots:
            raise HTTPException(
                status_code=404, detail="Chatbot not initialized for this file"
            )

        chatbot = initialized_chatbots[file_id]

        neighbors = chatbot.get_n_nearest_neighbours(
            query.text, n_neighbours=query.n_neighbors
        )

        # Extract the relevant information from the neighbors
        neighbor_texts = [neighbor.node.text for neighbor in neighbors]

        return {"neighbors": neighbor_texts}
    except Exception as e:
        print(f"Unexpected error in neighbors endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


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
