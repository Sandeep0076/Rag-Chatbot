"""
Main FastAPI application for the RAG PDF API.
"""


import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Tuple

import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import (
    BackgroundTasks,
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette_exporter import PrometheusMiddleware, handle_metrics

from configs.app_config import Config
from rtl_rag_chatbot_api.chatbot.chatbot_creator import AzureChatbot as Chatbot
from rtl_rag_chatbot_api.chatbot.csv_handler import TabularDataHandler
from rtl_rag_chatbot_api.chatbot.embedding_handler import EmbeddingHandler
from rtl_rag_chatbot_api.chatbot.file_handler import FileHandler
from rtl_rag_chatbot_api.chatbot.gcs_handler import GCSHandler
from rtl_rag_chatbot_api.chatbot.gemini_handler import GeminiHandler
from rtl_rag_chatbot_api.chatbot.image_reader import analyze_images
from rtl_rag_chatbot_api.chatbot.model_handler import ModelHandler
from rtl_rag_chatbot_api.chatbot.utils.encryption import encrypt_file
from rtl_rag_chatbot_api.common.chroma_manager import ChromaDBManager
from rtl_rag_chatbot_api.common.cleanup_coordinator import CleanupCoordinator
from rtl_rag_chatbot_api.common.models import (
    ChatRequest,
    CleanupRequest,
    DeleteRequest,
    EmbeddingCreationRequest,
    FileUploadResponse,
    NeighborsQuery,
    Query,
)
from rtl_rag_chatbot_api.common.prepare_sqlitedb_from_csv_xlsx import (
    PrepareSQLFromTabularData,
)
from rtl_rag_chatbot_api.common.prompts_storage import VISUALISATION_PROMPT
from rtl_rag_chatbot_api.oauth.get_current_user import get_current_user

# from rtl_rag_chatbot_api.oauth.get_current_user import get_current_user

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# disable logger for apscheduler because they are not nice. use own.
apscheduler_log = logging.getLogger("apscheduler")
apscheduler_log.setLevel(logging.ERROR)

configs = Config()
gcs_handler = GCSHandler(configs)
gemini_handler = GeminiHandler(configs, gcs_handler)
file_handler = FileHandler(configs, gcs_handler, gemini_handler)
model_handler = ModelHandler(configs, gcs_handler)
embedding_handler = EmbeddingHandler(configs, gcs_handler)

# database connection
if os.getenv("DB_INSTANCE"):
    logging.info("Using DB_INSTANCE env variable to connect to database")
    DATABASE_URL = f"postgresql://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@127.0.0.1:5432/chatbot_ui"
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
else:
    logging.warning(
        "No DB_INSTANCE env variable present. Not able to connect to database."
    )
    SessionLocal = None

title = "RAG PDF API"
description = """
RAG PDF API is a FastAPI-based application for processing and querying various document types
including PDFs, images, and tabular data (CSV/Excel) using Retrieval-Augmented Generation (RAG)
and SQL querying capabilities.

## Workflow

1. Upload a file (PDF, image, CSV, or Excel) using the `/file/upload` endpoint.
2. Create embeddings for the uploaded file with the `/embeddings/create` endpoint (for PDFs and images).
3. Initialize the model using the `/model/initialize` endpoint. By default
GPT4_omni_mini is selected. It's optional and mainly used when we need to change model for chatting.
4. Chat with the content using the `/file/chat` endpoint.

Additional features:
- Query tabular data (CSV/Excel) using natural language with SQL-like capabilities.
- Analyze images with the `/image/analyze` endpoint.
- Get nearest neighbors for a query with the `/file/neighbors` endpoint.
- View available models using the `/available-models` endpoint.
- Clean up files with the `/file/cleanup` endpoint.
- Chat with Google Gemini models without RAG or file context using `/chat/gemini`.
- Delete files using the `/files` DELETE endpoint.

Note: File storage in GCP has been removed from this version.
"""


@asynccontextmanager
async def start_scheduler(app: FastAPI):
    cleanup_coordinator = CleanupCoordinator(configs, SessionLocal)
    scheduler = BackgroundScheduler()
    scheduler.configure(logger=logging.getLogger("apscheduler"))

    # Use config value for interval
    scheduler.add_job(
        cleanup_coordinator.cleanup,
        trigger="interval",
        minutes=configs.cleanup.cleanup_interval_minutes,  # Use configured interval
        id="cleanup_job",
    )

    scheduler.start()
    try:
        yield
    finally:
        # Proper cleanup of resources
        scheduler.shutdown()
        # Clean up ChromaDB
        if chroma_manager:
            chroma_manager.cleanup()
        # Clean up any initialized models
        for model in initialized_models.values():
            if hasattr(model, "cleanup"):
                model.cleanup()
        # Clean up handlers
        for handler in initialized_handlers.values():
            if hasattr(handler, "cleanup"):
                handler.cleanup()
        # Clean up chatbots
        for chatbot in initialized_chatbots.values():
            if hasattr(chatbot, "cleanup"):
                chatbot.cleanup()


app = FastAPI(
    title=title, description=description, version="3.1.0", lifespan=start_scheduler
)

# Initialize ChromaDBManager at app level
chroma_manager = ChromaDBManager()

initialized_models = {}
initialized_chatbots = {}
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
    # Streamlit and NextJS localhost development
    allow_origins=os.getenv("ALLOWED_ORIGIN").split(","),
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
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    is_image: bool = Form(...),
    username: str = Form(...),
    current_user=Depends(get_current_user),
):
    """
    Handles file upload and automatically creates embeddings for PDFs and images.
    Made asynchronous to prevent blocking during long operations.
    """
    try:
        # Generate file_id with UUID
        file_id = str(uuid.uuid4())
        original_filename = file.filename
        file_extension = os.path.splitext(original_filename)[1].lower()

        # Process the file upload asynchronously
        result = await file_handler.process_file(file, file_id, is_image, username)
        file_id = result[
            "file_id"
        ]  # Use the file_id from result in case it's an existing file
        temp_file_path = result["temp_file_path"]

        # Handle CSV/Excel/Database files in background
        if (
            file_extension in [".csv", ".xlsx", ".xls", ".db", ".sqlite"]
            and result["status"] == "new"
        ):
            background_tasks.add_task(prepare_sqlite_db, file_id, temp_file_path)
            # For tabular files, we can create file_info.json immediately
            gcs_handler.temp_metadata[
                "embeddings_status"
            ] = "completed"  # No embeddings needed
            background_tasks.add_task(
                gcs_handler.upload_to_gcs,
                configs.gcp_resource.bucket_name,
                {
                    "metadata": (
                        gcs_handler.temp_metadata,
                        f"file-embeddings/{file_id}/file_info.json",
                    )
                },
            )

        # Create embeddings for PDF and Image files in background
        elif file_extension == ".pdf" or is_image:
            if result["status"] != "existing":  # Only create embeddings for new files
                background_tasks.add_task(
                    create_embeddings_background,
                    file_id=file_id,
                    temp_file_path=temp_file_path,
                    embedding_handler=embedding_handler,
                    configs=configs,
                    SessionLocal=SessionLocal,
                )

        # Clean up temporary file in background
        if temp_file_path and os.path.exists(temp_file_path):
            background_tasks.add_task(os.remove, temp_file_path)

        response = FileUploadResponse(
            file_id=file_id,
            message=result["message"],
            status="success",
            original_filename=original_filename,
            is_image=is_image,
        )

        return JSONResponse(content=response.dict(), background=background_tasks)

    except Exception as e:
        logging.error(f"Error in upload_file: {str(e)}")
        # Ensure cleanup of any temporary files
        if (
            "temp_file_path" in locals()
            and temp_file_path
            and os.path.exists(temp_file_path)
        ):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))


async def prepare_sqlite_db(file_id: str, temp_file_path: str):
    """
    Handles the preparation of a SQLite database for tabular data from the uploaded file.
    For CSV/Excel files: Creates a new SQLite database
    For SQLite DB files: Validates and copies the database
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

        # Get file extension
        file_extension = os.path.splitext(temp_file_path)[1].lower()

        # Prepare SQLite database
        data_preparer = PrepareSQLFromTabularData(temp_file_path, data_dir)
        success = data_preparer.run_pipeline()

        if not success:
            raise ValueError("Failed to prepare database from input file")

        # Encrypt the database before upload
        encrypted_db_path = encrypt_file(db_path)
        try:
            # Upload the encrypted SQLite database to GCS
            gcs_handler.upload_to_gcs(
                configs.gcp_resource.bucket_name,
                source=encrypted_db_path,
                destination_blob_name=f"file-embeddings/{file_id}/tabular_data.db.encrypted",
            )
        finally:
            # Clean up encrypted file
            if os.path.exists(encrypted_db_path):
                os.remove(encrypted_db_path)

        # Update file_info.json with success status and file type
        metadata = {
            "embeddings_status": "completed",
            "file_type": "database"
            if file_extension in [".db", ".sqlite"]
            else "tabular",
            "processing_status": "success",
        }
        gcs_handler.upload_to_gcs(
            configs.gcp_resource.bucket_name,
            {
                "metadata": (
                    metadata,
                    f"file-embeddings/{file_id}/file_info.json",
                )
            },
        )

    except Exception as e:
        logging.error(f"Error preparing SQLite database: {str(e)}")
        # Update file_info.json with error status
        metadata = {
            "embeddings_status": "failed",
            "file_type": "database"
            if file_extension in [".db", ".sqlite"]
            else "tabular",
            "processing_status": "error",
            "error": str(e),
        }
        try:
            gcs_handler.upload_to_gcs(
                configs.gcp_resource.bucket_name,
                {
                    "metadata": (
                        metadata,
                        f"file-embeddings/{file_id}/file_info.json",
                    )
                },
            )
        except Exception as upload_error:
            logging.error(f"Error updating file_info.json: {str(upload_error)}")
        raise


async def create_embeddings_background(
    file_id: str, temp_file_path: str, embedding_handler, configs, SessionLocal
):
    """Background task for creating embeddings."""
    try:
        embedding_result = await embedding_handler.ensure_embeddings_exist(
            file_id=file_id, temp_file_path=temp_file_path
        )
        if embedding_result["status"] == "error":
            logging.error(
                f"Error creating embeddings for {file_id}: {embedding_result['message']}"
            )
            cleanup_coordinator = CleanupCoordinator(configs, SessionLocal)
            cleanup_coordinator.cleanup_chroma_instance(file_id, include_gcs=True)
    except Exception as e:
        logging.error(f"Error in create_embeddings_background: {str(e)}")
        cleanup_coordinator = CleanupCoordinator(configs, SessionLocal)
        cleanup_coordinator.cleanup_chroma_instance(file_id, include_gcs=True)


@app.post("/embeddings/create")
async def create_embeddings(
    request: EmbeddingCreationRequest, current_user=Depends(get_current_user)
):
    """
    Creates embeddings for an uploaded file using Azure or Google Gemini models.

    This endpoint handles the generation and storage of embeddings for document processing.
    It checks if embeddings already exist, creates new ones if needed, and updates the file status.

    Args:
        request (EmbeddingCreationRequest): Request body containing:
            - file_id (str): Unique identifier for the uploaded file
            - is_image (bool): Flag indicating if the file is an image
        current_user: Authenticated user information (handled by dependency)

    Returns:
        dict: Dictionary containing:
            - message (str): Status message about embedding creation
            - For tabular data: Returns formatted markdown table if applicable

    Raises:
        HTTPException:
            - 404: If file not found or model not initialized
            - 500: For embedding creation or storage errors

    Notes:
        - Embeddings are stored using ChromaDB
        - Status is tracked in GCS file metadata
        - Supports both text and image files
    """
    try:
        file_info = gcs_handler.get_file_info(request.file_id)
        if not file_info:
            raise HTTPException(status_code=404, detail="File info not found")

        # Check local embeddings first
        azure_path = f"./chroma_db/{request.file_id}/azure"
        gemini_path = f"./chroma_db/{request.file_id}/google"
        local_exists = (
            os.path.exists(azure_path)
            and os.path.exists(gemini_path)
            and os.path.exists(os.path.join(azure_path, "chroma.sqlite3"))
            and os.path.exists(os.path.join(gemini_path, "chroma.sqlite3"))
        )

        if local_exists:
            return {
                "message": "Embeddings already exist locally",
                "status": "existing",
                "can_chat": True,
            }

        # Rest of your existing embedding creation code
        original_filename = file_info.get("original_filename")
        if not original_filename:
            raise HTTPException(status_code=400, detail="Original filename not found")

        temp_file_path = f"local_data/{request.file_id}_{original_filename}"
        if not os.path.exists(temp_file_path):
            raise HTTPException(
                status_code=404, detail=f"File not found at {temp_file_path}"
            )

        embedding_handler = EmbeddingHandler(configs, gcs_handler)
        result = await embedding_handler.ensure_embeddings_exist(
            request.file_id, temp_file_path
        )

        if isinstance(result, dict) and result.get("status") == "error":
            return JSONResponse(
                status_code=400,
                content={"message": result.get("message", "Unknown error")},
            )

        return result

    except Exception as e:
        logging.error(f"Error in create_embeddings: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "message": "An unexpected error occurred while processing your file.",
                "error": str(e),
                "can_chat": False,
            },
        )


def check_db_existence(
    file_id: str, query: Query, configs: dict, initialized_models: dict
) -> Tuple[str, bool, Any]:
    """
    Check if database exists and initialize tabular handler if needed.

    Args:
        file_id: The ID of the file to check
        query: Query parameters including model choice
        configs: Application configurations
        initialized_models: Dictionary of initialized models

    Returns:
        Tuple containing:
        - database path
        - boolean indicating if it's a tabular database
        - initialized model if tabular, None otherwise
    """
    db_path = f"./chroma_db/{file_id}/tabular_data.db"
    is_tabular = os.path.exists(db_path)
    model = None

    if is_tabular:
        logging.info("Initializing TabularDataHandler")
        try:
            model = TabularDataHandler(configs, query.file_id, query.model_choice)
            model_key = f"{query.file_id}_{query.user_id}_{query.model_choice}"
            initialized_models[model_key] = model
        except ValueError as ve:
            logging.error(f"Error initializing TabularDataHandler: {str(ve)}")
            raise HTTPException(
                status_code=400,
                detail=(
                    "The CSV file appears to be corrupted or contains no valid data. "
                    "Please try uploading a different file."
                ),
            )

    return db_path, is_tabular, model


def initialize_rag_model(
    query: Query,
    configs: dict,
    gcs_handler: Any,
    file_info: dict,
):
    """
    Initialize RAG model based on query parameters.

    Args:
        query: Query parameters including model choice
        configs: Application configurations
        gcs_handler: GCS handler instance
        file_info: File information from GCS

    Returns:
        Initialized model instance
    """
    chroma_path = f"./chroma_db/{query.file_id}"
    is_gemini = query.model_choice.lower() in ["gemini-flash", "gemini-pro"]
    embedding_type = "google" if is_gemini else "azure"
    model_path = os.path.join(chroma_path, embedding_type, "chroma.sqlite3")

    # If local embeddings don't exist, check GCS
    if not os.path.exists(model_path):
        if not file_info.get("embeddings_status") == "completed":
            raise HTTPException(
                status_code=400,
                detail="Embeddings are not ready yet. Please wait a moment.",
            )
        print("No local embeddings found, downloading from GCS")
        gcs_handler.download_files_from_folder_by_id(query.file_id)

    # Initialize model with ChromaDB
    if is_gemini:
        model = GeminiHandler(configs, gcs_handler)
        model.initialize(
            model=query.model_choice,
            file_id=query.file_id,
            embedding_type="google",
            collection_name=f"rag_collection_{query.file_id}",
            user_id=query.user_id,
        )
    else:
        model = Chatbot(configs, gcs_handler)
        model.initialize(
            model_choice=query.model_choice,
            file_id=query.file_id,
            embedding_type="azure",
            collection_name=f"rag_collection_{query.file_id}",
            user_id=query.user_id,
        )

    return model


def format_table_response(response: list[Any]) -> dict[str, Any]:
    """
    Format table response into markdown and structured data.

    Args:
        response: List containing headers and rows

    Returns:
        Dictionary containing formatted response
    """
    if len(response) > 1:
        headers = response[0]
        rows = response[1:]

        # Create markdown table
        table_str = "| " + " | ".join(str(h) for h in headers) + " |\n"
        table_str += "|" + "|".join(["---" for _ in headers]) + "|\n"
        for row in rows:
            table_str += "| " + " | ".join(str(cell) for cell in row) + " |\n"

        return {
            "response": table_str,
            "is_table": True,
            "headers": headers,
            "rows": rows,
        }

    return {"response": "No data found", "is_table": False}


@app.post("/file/chat")
async def chat(query: Query, current_user=Depends(get_current_user)):
    """
    Process chat queries against document content using specified language models.

    Handles multiple model types (Azure LLM, Gemini) and data formats (text, tabular).
    Automatically initializes or switches models based on request parameters.

    Args:
        query (Query): Request body containing:
            - text (str): User's query text
            - file_id (str): ID of the file to query against
            - model_choice (str): Selected model (e.g., 'gpt_4o_mini', 'gemini-pro')
        current_user: Authenticated user information (handled by dependency)

    Returns:
        dict: Response containing:
            - response (str): Model's answer to the query
            - For tabular data: Returns formatted markdown table if applicable

    """
    try:
        logging.info(
            f"Graphic generation flag is  {query.generate_visualization} for file {query.file_id}"
        )
        if len(query.text) == 0:
            raise HTTPException(status_code=400, detail="Text array cannot be empty")

        file_info = gcs_handler.get_file_info(query.file_id)
        if not file_info:
            logging.info(f"File info not found for {query.file_id}")
            raise HTTPException(
                status_code=400,
                detail="File appears to be corrupted or empty. Please try uploading a different file.",
            )

        model_key = f"{query.file_id}_{query.user_id}_{query.model_choice}"
        model = initialized_models.get(model_key)

        try:
            if not model:
                db_path, is_tabular, model = check_db_existence(
                    query.file_id, query, configs, initialized_models
                )
                if not is_tabular:
                    model = initialize_rag_model(query, configs, gcs_handler, file_info)
                    initialized_models[model_key] = model
                    logging.info(
                        f"Model initialized: {query.file_id}, user: {query.user_id}, model: {query.model_choice}"
                    )
            if query.generate_visualization:
                current_question = query.text[-1] + VISUALISATION_PROMPT

            else:
                current_question = query.text[-1]

            # For GPT-3.5, skip previous messages to stay within token limits
            if query.model_choice.lower() == "gpt_3_5_turbo":
                chat_context = current_question
            else:
                chat_context = (
                    "\n".join([f"Previous message: {msg}" for msg in query.text[:-1]])
                    + f"\nCurrent question: {current_question}"
                    if len(query.text) > 1
                    else current_question
                )

            response = model.get_answer(chat_context)

            if isinstance(response, list):
                return format_table_response(response)

            if query.generate_visualization:
                try:
                    # Debug logging to see the response
                    logging.info(f"Response type: {type(response)}")
                    logging.info(f"Response content: {response}")

                    # Parse the response string into a JSON object if it's a string
                    if isinstance(response, str):
                        # Replace Python boolean values with JSON boolean values
                        response = response.replace("True", "true").replace(
                            "False", "false"
                        )
                        response = response.strip()
                        # Remove markdown code block markers if present
                        response = (
                            response.replace("```json", "").replace("```", "").strip()
                        )
                        chart_config = json.loads(response)
                    else:
                        chart_config = response

                    return JSONResponse(
                        content={
                            "chart_config": chart_config,  # Return chart_config directly
                            "is_table": False,
                        }
                    )
                except json.JSONDecodeError as je:
                    logging.error(
                        f"Invalid chart configuration JSON at position {je.pos}: {je.msg}"
                    )
                    logging.error(f"JSON string: {response}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid chart configuration format: {str(je)}",
                    )
                except Exception as e:
                    logging.error(f"Error generating chart: {str(e)}")
                    raise HTTPException(
                        status_code=500, detail="Failed to generate chart"
                    )

            return {"response": str(response), "is_table": False}

        finally:
            # Cleanup if model is TabularDataHandler
            if isinstance(model, TabularDataHandler):
                model.cleanup()

    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        error_message = str(e)
        if "coroutine" in error_message.lower():
            error_message = "Internal server error: Model response handling failed. Please try again."
        raise HTTPException(status_code=500, detail=error_message)


@app.get("/available-models")
async def get_available_models(current_user=Depends(get_current_user)):
    """
    Endpoint to retrieve a list of available models including Azure LLM models and Gemini models.
    """
    azure_models = list(configs.azure_llm.models.keys())
    gemini_models = ["gemini-flash", "gemini-pro"]
    all_models = azure_models + gemini_models
    return {"models": all_models}


@app.post("/file/cleanup")
async def manual_cleanup(
    request: CleanupRequest = Body(default=CleanupRequest()),
    current_user=Depends(get_current_user),
):
    """Endpoint to manually trigger cleanup."""
    try:
        cleanup_coordinator = CleanupCoordinator(configs, SessionLocal)
        cleanup_coordinator.cleanup(is_manual=request.is_manual)
        return {"status": "Cleanup completed successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred during cleanup: {str(e)}"
        )


@app.post("/file/neighbors")
async def get_neighbors(query: NeighborsQuery, current_user=Depends(get_current_user)):
    """Endpoint to retrieve nearest neighbors for a given text query."""
    try:
        # Get the model instance using the combined key
        model_key = f"{query.file_id}_{current_user.id}"  # or however you get user ID
        if model_key not in initialized_models:
            raise HTTPException(
                status_code=404, detail="Model not initialized for this file"
            )

        model = initialized_models[model_key]
        neighbors = model.get_n_nearest_neighbours(
            query.text, n_neighbours=query.n_neighbors
        )

        # Process results if they're from Azure model
        if isinstance(model, Chatbot):
            neighbors = [
                neighbor.node.text if hasattr(neighbor, "node") else neighbor
                for neighbor in neighbors
            ]

        return {"neighbors": neighbors}
    except Exception as e:
        logging.error(f"Error in get_neighbors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/analyze-image", response_model=Dict[str, Any])
async def analyze_image_endpoint(
    file: UploadFile = File(...), current_user=Depends(get_current_user)
):
    """
    Endpoint to analyze an uploaded image file.
    Saves the analysis result to a JSON file and returns the result details.
    Handles file upload, temporary file creation, analysis, result saving, and error handling.
    """
    try:
        file_extension = Path(file.filename).suffix

        # Create a temporary file with the original extension
        temp_file_path = f"temp_{uuid.uuid4()}{file_extension}"

        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Analyze the image
        result = await analyze_images(temp_file_path)

        # Generate a unique filename for the result
        result_filename = f"image_analysis_{uuid.uuid4()}.json"
        result_file_path = os.path.join("processed_data", result_filename)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(result_file_path), exist_ok=True)

        # Save the result to a JSON file
        with open(result_file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        return {
            "message": "Image analysis completed successfully",
            "result_file": result_filename,
            "analysis": result,
        }

    except Exception as e:
        logging.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logging.info(f"Cleaned up temporary file: {temp_file_path}")


@app.delete("/delete")
async def delete_resources(
    request: DeleteRequest, current_user=Depends(get_current_user)
):
    """
    Delete ChromaDB embeddings and associated resources for one or multiple files.
    include_gcs: Boolean flag to determine if GCS cleanup should be performed.

    Args:
        request (DeleteRequest): Request body containing:
            - file_ids (Union[str, List[str]]): Single file ID or list of file IDs
            - include_gcs (bool): Whether to include GCS cleanup (default: False)
        current_user: Authenticated user information

    Returns:
        dict: Response containing status for each file ID
    """
    try:
        # Convert single file_id to list for consistent processing
        file_ids = (
            [request.file_ids]
            if isinstance(request.file_ids, str)
            else request.file_ids
        )

        results = {}
        for file_id in file_ids:
            try:
                cleanup_coordinator = CleanupCoordinator(
                    configs, SessionLocal, gcs_handler
                )
                cleanup_coordinator.cleanup_chroma_instance(
                    file_id, include_gcs=request.include_gcs
                )
                results[file_id] = "Success"
            except Exception as e:
                results[file_id] = f"Error: {str(e)}"
                logging.error({"file_id": file_id, "error": str(e)})

        # If only one file_id was provided, maintain original response format
        if isinstance(request.file_ids, str):
            if results[request.file_ids] == "Success":
                return {
                    "message": f"ChromaDB embeddings for file_id {request.file_ids} have been deleted successfully"
                }
            else:
                raise HTTPException(status_code=500, detail=results[request.file_ids])

        # For multiple file_ids, return status of all operations
        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/gemini")
async def get_gemini_response_stream(
    request: ChatRequest, current_user=Depends(get_current_user)
):
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
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_config="logging_config.json",
        log_level="info",
    )
