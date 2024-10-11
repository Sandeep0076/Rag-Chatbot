"""
Main FastAPI application for the RAG PDF API.
"""
import json
import logging
import os
import shutil
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # , Depends,
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette_exporter import PrometheusMiddleware, handle_metrics

from configs.app_config import Config
from rtl_rag_chatbot_api.chatbot.chatbot_creator import Chatbot
from rtl_rag_chatbot_api.chatbot.csv_handler import TabularDataHandler
from rtl_rag_chatbot_api.chatbot.embedding_handler import EmbeddingHandler
from rtl_rag_chatbot_api.chatbot.file_handler import FileHandler
from rtl_rag_chatbot_api.chatbot.gcs_handler import GCSHandler
from rtl_rag_chatbot_api.chatbot.gemini_handler import GeminiHandler
from rtl_rag_chatbot_api.chatbot.image_reader import analyze_images
from rtl_rag_chatbot_api.chatbot.model_handler import ModelHandler
from rtl_rag_chatbot_api.common.models import (
    ChatRequest,
    EmbeddingCreationRequest,
    FileDeleteRequest,
    FileUploadResponse,
    ModelInitRequest,
    NeighborsQuery,
    Query,
)
from rtl_rag_chatbot_api.common.prepare_sqlitedb_from_csv_xlsx import (
    PrepareSQLFromTabularData,
)

# from rtl_rag_chatbot_api.oauth.get_current_user import get_current_user

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


logging.basicConfig(level=logging.INFO)

configs = Config()
gcs_handler = GCSHandler(configs)
file_handler = FileHandler(configs, gcs_handler)
model_handler = ModelHandler(configs, gcs_handler)
embedding_handler = EmbeddingHandler(configs, gcs_handler)


title = "RAG PDF API"
description = """
RAG PDF API is a FastAPI-based application for processing and
querying PDF documents using Retrieval-Augmented Generation (RAG).

## Workflow

1. Upload a PDF file using the `/file/upload` endpoint.
2. Create embeddings for the uploaded file with the `/embeddings/create` endpoint.
3. Initialize the model using the `/model/initialize` endpoint. By default
GPT4_omni is selected. Its optional and mainly used when we need to change model for chatting.
4. Chat with the PDF content using the `/file/chat` endpoint.

Additional features:
- Analyze images with the `/image/analyze` endpoint.
- Get nearest neighbors for a query with the `/file/neighbors` endpoint.
- View available models using the `/available-models` endpoint.
- Clean up files with the `/file/cleanup` endpoint.
- For chatting with Google models without RAG or file context./chat/gemini")

"""

app = FastAPI(
    title=title,
    description=description,
    version="3.1.0",
)

global gemini_handler
gemini_handler = None
# Global dictionary to store initialized chatbots
# depreacted
initialized_chatbots = {}

initialized_models = {}
# Global dictionary to store initialized handlers
initialized_handlers = {}


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
    # allow_origins=[os.getenv("ALLOWED_ORIGIN")],
    allow_origins=["http://localhost:8501"],  # Streamlit default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_route("/metrics", handle_metrics)


@app.get("/health")
async def health():
    # async def health(current_user=Depends(get_current_user)):
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
    username: str = Form(...),
    # current_user = Depends(get_current_user)
):
    """
    Handles the uploading of a file with optional image flag.

    Args:
        file (UploadFile): The file to be uploaded.
        is_image (bool): Flag indicating if the file is an image.
        username (str): The username associated with the uploaded file.

    Returns:
        FileUploadResponse: Response containing details of the uploaded file.
    """
    try:
        file_id = str(uuid.uuid4())
        original_filename = file.filename
        file_extension = os.path.splitext(original_filename)[1].lower()

        result = await file_handler.process_file(file, file_id, is_image, username)

        print(f"Result from process_file: {result}")
        if result["status"] == "existing":
            file_id = result["file_id"]
            if file_handler.download_existing_file(file_id):
                message = "File already exists. Required files downloaded."
            else:
                message = "File already exists, but error downloading necessary files."

            # Add temp_file_path for existing files
            temp_file_path = f"local_data/{file_id}_{original_filename}"
            result["temp_file_path"] = temp_file_path
        else:
            message = result["message"]

        temp_file_path = result["temp_file_path"]

        # If it's a CSV or Excel file and it's a new upload, prepare the SQLite database
        if file_extension in [".csv", ".xlsx", ".xls"] and result["status"] == "new":
            await prepare_sqlite_db(file_id, temp_file_path)

        return FileUploadResponse(
            message=message,
            file_id=file_id,
            original_filename=original_filename,
            is_image=is_image,
        )
    except Exception as e:
        print(f"Exception in upload_file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def prepare_sqlite_db(file_id: str, temp_file_path: str):
    """
    Handles the preparation of a SQLite database for tabular data from the uploaded file.
    Downloads and decrypts the file, prepares the SQLite database,
    uploads it to GCS, and cleans up the decrypted file if successful.
    """
    try:
        data_dir = f"./chroma_db/{file_id}"
        os.makedirs(data_dir, exist_ok=True)

        # Check if the database already exists
        db_path = os.path.join(data_dir, "tabular_data.db")
        if os.path.exists(db_path):
            logging.info(f"SQLite database already exists for file_id: {file_id}")
            return

        # Prepare SQLite database
        data_preparer = PrepareSQLFromTabularData(temp_file_path, data_dir)
        data_preparer.run_pipeline()

        # Upload the SQLite database to GCS
        gcs_handler.upload_to_gcs(
            configs.gcp_resource.bucket_name,
            source=db_path,
            destination_blob_name=f"file-embeddings/{file_id}/tabular_data.db",
        )

    except Exception as e:
        logging.error(f"Error preparing SQLite database: {str(e)}")
        raise


@app.post("/model/initialize")
async def initialize_model(request: ModelInitRequest):
    # async def initialize_model(request: ModelInitRequest, current_user = Depends(get_current_user)):
    """
    Endpoint to initialize a model based on the specified model choice, file ID, and embedding type.

    Args:
        request (ModelInitRequest): Request object containing model choice and file ID.

    Returns:
        dict: A message indicating the successful initialization of the specified model.
    Raises:
        HTTPException: If embeddings are not found for the specified file or if an error occurs during initialization.
    """
    try:
        file_info = embedding_handler.get_embeddings_info(request.file_id)
        if not file_info:
            raise HTTPException(
                status_code=404, detail="Embeddings not found for this file"
            )

            # Check if the file is a tabular data file
        db_path = f"./chroma_db/{request.file_id}/tabular_data.db"
        if os.path.exists(db_path):
            # Initialize TabularDataHandler for CSV/Excel files
            model = TabularDataHandler(configs, request.file_id, request.model_choice)
        else:
            embedding_type = (
                "google"
                if request.model_choice.lower() in ["gemini-flash", "gemini-pro"]
                else "azure"
            )
            chroma_db_path = f"./chroma_db/{request.file_id}/{embedding_type}"

            logging.info(f"Initializing model for {embedding_type} embeddings")
            logging.info(f"Chroma DB path: {chroma_db_path}")
            logging.info(
                f"Contents of chroma_db folder: {os.listdir(f'./chroma_db/{request.file_id}')}"
            )

            if not os.path.exists(chroma_db_path):
                logging.warning(
                    f"{embedding_type} embeddings not found locally. Downloading..."
                )
                gcs_handler.download_files_from_folder_by_id(request.file_id)

            logging.info(f"Contents of {chroma_db_path}: {os.listdir(chroma_db_path)}")
            logging.info(
                f"model choice: {request.model_choice} { request.file_id}, { embedding_type}"
            )
            model = model_handler.initialize_model(
                request.model_choice, request.file_id, embedding_type
            )
        initialized_models[request.file_id] = model
        return {"message": f"Model {request.model_choice} initialized successfully"}
    except Exception as e:
        logging.error(f"Error in initialize_model: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while initializing the model: {str(e)}",
        )


@app.post("/file/chat")
async def chat(query: Query):
    # async def chat(query: Query, current_user = Depends(get_current_user)):
    """
    Endpoint to interact with the chatbot using a specific file.
    Checks if the model is initialized for the given file, retrieves the model,
    and calls get_answer on the model with the provided text.
    Returns the response from the model. Handles exceptions and logs errors appropriately.
    """
    try:
        if query.file_id not in initialized_models:
            raise HTTPException(
                status_code=404, detail="Model not initialized for this file"
            )

        model = initialized_models[query.file_id]
        logging.info(f"Model type: {type(model)}")

        if isinstance(model, TabularDataHandler):
            logging.info("Debugging database contents:")
            model.debug_database()

        logging.info(f"Calling get_answer on model: {type(model)}")
        response = model.get_answer(query.text)

        # Check if the response is a list (tabular data)
        if isinstance(response, list) and len(response) > 1:
            headers = response[0]
            rows = response[1:]

            # Format the table as a string
            table_str = "| " + " | ".join(str(h) for h in headers) + " |\n"
            table_str += "|" + "|".join(["---" for _ in headers]) + "|\n"
            for row in rows:
                table_str += "| " + " | ".join(str(cell) for cell in row) + " |\n"

            return {"response": table_str}
        else:
            # For non-tabular data (e.g., PDF, image analysis)
            return {"response": str(response)}

    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embeddings/create")
async def create_embeddings(request: EmbeddingCreationRequest):
    # async def create_embeddings(request: EmbeddingCreationRequest,
    #  current_user = Depends(get_current_user)):
    """
    Endpoint to create embeddings for a file based on the provided request.

    Parameters:
        request (EmbeddingCreationRequest): Request object containing file_id and is_image flag.

    Returns:
        dict: A dictionary with a message indicating the status of the embeddings creation process.
            If embeddings already exist, returns info if available; otherwise, returns appropriate messages.
            Raises HTTPException with status code 500 in case of errors.
    """
    try:
        embedding_handler = EmbeddingHandler(configs, gcs_handler)

        # Get file info
        file_info = gcs_handler.get_file_info(request.file_id)
        if not file_info:
            raise HTTPException(status_code=404, detail="File info not found")

        if file_info.get("embeddings_status") == "completed":
            return {
                "message": "Embeddings already exist for this file",
                "info": file_info.get("embeddings"),
            }

        # Construct the path to the temporary file
        temp_file_path = (
            f"local_data/{request.file_id}_{file_info.get('original_filename', '')}"
        )

        if not os.path.exists(temp_file_path):
            raise HTTPException(status_code=404, detail="Temporary file not found")

        try:
            result = await embedding_handler.create_and_upload_embeddings(
                request.file_id, file_info.get("is_image", False), temp_file_path
            )

            # Update file info to indicate embeddings are completed
            gcs_handler.update_file_info(
                request.file_id, {"embeddings_status": "completed"}
            )

            return result
        finally:
            # Clean up the temporary file after successful embedding creation
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    except Exception as e:
        logging.error(f"Error in create_embeddings: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/available-models")
async def get_available_models():
    # async def get_available_models(current_user = Depends(get_current_user)):
    """
    Endpoint to retrieve a list of available models including Azure LLM models and Gemini models.
    """
    azure_models = list(configs.azure_llm.models.keys())
    gemini_models = ["gemini-flash", "gemini-pro"]
    all_models = azure_models + gemini_models
    return {"models": all_models}


@app.post("/file/cleanup")
async def cleanup_files():
    # async def cleanup_files(current_user = Depends(get_current_user)):
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
    # async def initialize_chatbot(file_id: str, model_choice: str,
    # current_user = Depends(get_current_user)):
    """
    Initialize a chatbot with the given file ID and model choice.

    Parameters:
    - file_id (str): The ID of the file to initialize the chatbot with.
    - model_choice (str): The choice of model for the chatbot.

    Raises:
    - HTTPException: If there is an error during chatbot setup, such as missing files or invalid model choice.
    """
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
    # async def get_neighbors(query: NeighborsQuery, current_user = Depends(get_current_user)):
    """
    Endpoint to retrieve nearest neighbors for a given text query and file ID.
    Checks if the model is initialized for the specified file, then retrieves the nearest neighbors accordingly.
    Returns a dictionary containing the list of neighbors.
    Handles exceptions and returns appropriate HTTP status codes with error details.
    """
    if query.file_id not in initialized_models:
        raise HTTPException(
            status_code=404, detail="Model not initialized for this file"
        )

    try:
        model = initialized_models[query.file_id]

        if isinstance(model, GeminiHandler):
            neighbors = model.get_n_nearest_neighbours(
                query.text, query.file_id, query.n_neighbors
            )
        else:  # Assuming it's a Chatbot instance for Azure models
            neighbors_with_metadata = model.get_n_nearest_neighbours(
                query.text, n_neighbours=query.n_neighbors
            )
            neighbors = [
                neighbor.node.text if hasattr(neighbor, "node") else neighbor
                for neighbor in neighbors_with_metadata
            ]

        return {"neighbors": neighbors}
    except Exception as e:
        logging.error(f"Error in get_neighbors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/image/analyze")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    # async def analyze_image_endpoint(file: UploadFile = File(...),
    #  current_user = Depends(get_current_user)):
    """
    Endpoint to analyze an uploaded image file.
    Saves the analysis result to a JSON file and returns the result details.
    Handles file upload, temporary file creation, analysis, result saving, and error handling.
    """
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


@app.delete("/files")
async def delete_files(request: FileDeleteRequest):
    # async def delete_files(request: FileDeleteRequest, current_user = Depends(get_current_user)):
    """
    Delete files and their embeddings based on the provided file IDs.

    Args:
        file_ids (List[str]): List of file IDs to delete.

    Returns:
        dict: A message indicating the status of the deletion process,
        along with the list of deleted files and any errors encountered.
    """
    file_ids = request.file_ids
    deleted_files = []
    errors = []

    gcs_handler = GCSHandler(configs)

    for file_id in file_ids:
        try:
            # Delete file and embeddings from GCS
            gcs_handler.delete_file_and_embeddings(file_id)

            # Delete local Chroma DB files
            chroma_db_path = f"./chroma_db/{file_id}"
            if os.path.exists(chroma_db_path):
                shutil.rmtree(chroma_db_path)

            # Remove the file from initialized_models if it exists
            if file_id in initialized_models:
                del initialized_models[file_id]

            deleted_files.append(file_id)
        except Exception as e:
            errors.append({"file_id": file_id, "error": str(e)})

    if errors:
        return {
            "message": "Some files could not be deleted",
            "deleted_files": deleted_files,
            "errors": errors,
        }
    else:
        return {
            "message": "All files and their embeddings have been deleted successfully",
            "deleted_files": deleted_files,
        }


@app.post("/chat/gemini")
async def get_gemini_response_stream(request: ChatRequest):
    # async def get_gemini_response_stream(request: ChatRequest,
    # current_user = Depends(get_current_user)):
    """
    Endpoint for chatting with Gemini models (Flash or Pro) without RAG or file context.

    This endpoint allows direct interaction with either the Gemini Flash or Gemini Pro model.
    It takes a model choice and a message as input, and returns the model's response.

    Args:
        request (ChatRequest): A Pydantic model containing:
            - model (str): The Gemini model to use. Must be either "gemini-flash" or "gemini-pro".
            - message (str): The user's input message or query for the model.

    Returns:
        dict: A dictionary containing the model's response:
            - response (str): The text response generated by the Gemini model.
    """
    if request.model not in ["gemini-flash", "gemini-pro"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid model choice. Use 'gemini-flash' or 'gemini-pro'.",
        )

    try:
        model_handler = ModelHandler(configs, gcs_handler)
        model = model_handler.initialize_model(
            request.model, file_id=None, embedding_type="gemini"
        )
        return StreamingResponse(
            model.get_gemini_response_stream(request.message), media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


def start():
    """
    Function to start the FastAPI application.
    Launched with `poetry run start` at root level
    Streamlit : streamlit run streamlit_app.py
    """
    uvicorn.run("rtl_rag_chatbot_api.app:app", host="0.0.0.0", port=8080, reload=False)
