"""
Main FastAPI application for the RAG PDF API.
"""


import asyncio
import json
import logging
import os
import time
import uuid
from asyncio import Semaphore
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import BackgroundTasks, Body, Depends, FastAPI, File, Form, HTTPException
from fastapi import Query as QueryParam
from fastapi import Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette_exporter import PrometheusMiddleware, handle_metrics

from configs.app_config import Config
from rtl_rag_chatbot_api.chatbot.chatbot_creator import AzureChatbot as Chatbot
from rtl_rag_chatbot_api.chatbot.chatbot_creator import get_azure_non_rag_response
from rtl_rag_chatbot_api.chatbot.combined_image_handler import CombinedImageGenerator
from rtl_rag_chatbot_api.chatbot.csv_handler import TabularDataHandler
from rtl_rag_chatbot_api.chatbot.dalle_handler import DalleImageGenerator
from rtl_rag_chatbot_api.chatbot.data_visualization import detect_visualization_need
from rtl_rag_chatbot_api.chatbot.embedding_handler import EmbeddingHandler
from rtl_rag_chatbot_api.chatbot.file_handler import FileHandler
from rtl_rag_chatbot_api.chatbot.gcs_handler import GCSHandler
from rtl_rag_chatbot_api.chatbot.gemini_handler import (
    GeminiHandler,
    get_gemini_non_rag_response,
)
from rtl_rag_chatbot_api.chatbot.image_reader import analyze_images
from rtl_rag_chatbot_api.chatbot.imagen_handler import ImagenGenerator
from rtl_rag_chatbot_api.chatbot.model_handler import ModelHandler
from rtl_rag_chatbot_api.chatbot.utils.encryption import encrypt_file
from rtl_rag_chatbot_api.common.chroma_manager import ChromaDBManager
from rtl_rag_chatbot_api.common.cleanup_coordinator import CleanupCoordinator
from rtl_rag_chatbot_api.common.models import (
    ChatRequest,
    CleanupRequest,
    DeleteRequest,
    EmbeddingsCheckRequest,
    FileDeleteRequest,
    FileUploadResponse,
    ImageGenerationRequest,
    NeighborsQuery,
    Query,
)
from rtl_rag_chatbot_api.common.prepare_sqlitedb_from_csv_xlsx import (
    PrepareSQLFromTabularData,
)
from rtl_rag_chatbot_api.common.prompts_storage import (
    CHART_DETECTION_PROMPT,
    VISUALISATION_PROMPT,
)
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
dalle_handler = DalleImageGenerator(configs)
imagen_handler = ImagenGenerator(configs)
combined_image_handler = CombinedImageGenerator(configs, dalle_handler, imagen_handler)

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

# Concurrency control for file processing
# This allows multiple file processing requests to run in parallel
file_processing_semaphore = Semaphore(
    10
)  # Allow up to 10 concurrent file processing operations

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
    allow_origins=os.getenv("ALLOWED_ORIGIN", "http://localhost:8080").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_route("/metrics", handle_metrics)


@app.get("/internal/healthy")
async def health():
    """
    Shows application health information.
    In the future this could do some actual checks.
    """
    return {"status": "healthy"}


@app.get("/internal/ready")
async def ready():
    """
    Hit by readiness probes to check if the application is ready to serve traffic.
    If the API is blocked, due to long running tasks, there is no traffic send to the pod.
    """
    return {"status": "ready"}


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


# Asynchronous file processing function that runs concurrently with a semaphore limit
async def process_file_with_semaphore(file_handler, file, file_id, is_image, username):
    async with file_processing_semaphore:
        logging.info(
            f"Starting parallel processing for file: {file.filename} with ID: {file_id}"
        )
        start_time = time.time()
        # below function is for new file only encrypts and uploads to GCS
        result = await file_handler.process_file(file, file_id, is_image, username)
        elapsed = time.time() - start_time
        logging.info(
            f"Completed processing for file: {file.filename} in {elapsed:.2f}s"
        )
        return result


async def process_url_content(
    file_handler, embedding_handler, urls, username, background_tasks
):
    """
    Process content from URLs and create embeddings.

    Args:
        file_handler: The handler for file processing
        embedding_handler: The handler for embedding creation
        urls: String containing URLs (comma or newline separated)
        username: Username for file ownership
        background_tasks: BackgroundTasks for async operations

    Returns:
        dict: Response data including file IDs and status
    """
    logging.info(f"Processing URL content for user {username}")
    temp_file_id = str(uuid.uuid4())

    url_result = await file_handler.process_urls(
        urls, username, temp_file_id, background_tasks, embedding_handler
    )

    # Check if URL processing returned an error status
    if url_result.get("status") == "error":
        logging.error(f"Error processing URLs: {url_result.get('message')}")
        raise HTTPException(
            status_code=400,
            detail=url_result.get("message", "Error processing URLs"),
        )

    # Format the response for multiple file IDs
    if "file_ids" in url_result:
        logging.info(f"Successfully processed {len(url_result['file_ids'])} URLs")

        # Maintain backward compatibility by including file_id in response if possible
        if url_result["file_ids"]:
            url_result["file_id"] = url_result["file_ids"][0]

        # Ensure multi_file_mode is set if multiple files
        if len(url_result["file_ids"]) > 1:
            url_result["multi_file_mode"] = True
            logging.info(
                f"Setting multi_file_mode to TRUE, file_ids: {url_result['file_ids']}"
            )

    # Generate a unique session_id for this URL content batch
    session_id = str(uuid.uuid4())

    # Create a proper response object to ensure all fields are included
    response_data = {
        "file_id": url_result.get("file_id"),
        "file_ids": url_result.get("file_ids", []),
        "multi_file_mode": url_result.get("multi_file_mode", False),
        "message": url_result.get("message", ""),
        "status": url_result.get("status", ""),
        "original_filename": url_result.get("original_filename", "url_content.txt"),
        "is_image": url_result.get("is_image", False),
        "is_tabular": url_result.get("is_tabular", False),
        "temp_file_path": url_result.get("temp_file_path"),
        "session_id": session_id,  # Include the session_id in the response
    }

    logging.info(
        f"Returning URL processing response: file_ids={response_data['file_ids']}, "
        f"multi_file_mode={response_data['multi_file_mode']}"
    )
    return response_data


def prepare_file_list(file, files):
    """
    Combine files from both the single and multiple file parameters.

    Args:
        file: Single file upload parameter
        files: Multiple files upload parameter

    Returns:
        list: Combined list of files to process
    """
    all_files = []

    # Add files from the 'files' parameter (multi-file)
    if files and len(files) > 0:
        logging.info(f"Received {len(files)} files in the 'files' parameter")
        for i, f in enumerate(files):
            logging.info(f"  File {i+1}: {f.filename}")
            all_files.append(f)

    # Add file from the 'file' parameter (single-file) if it exists and not already included
    if file and (not files or file not in files):
        logging.info(f"Received a file in the 'file' parameter: {file.filename}")
        all_files.append(file)

    return all_files


async def process_files_in_parallel(file_handler, all_files, is_image, username):
    """
    Process multiple files in parallel using asyncio.

    Args:
        file_handler: The handler for file processing
        all_files: List of files to process
        is_image: Flag indicating if files are images
        username: Username for file ownership

    Returns:
        tuple: (results, processed_file_ids, original_filenames, is_tabular_flags, statuses)
    """
    # Generate unique file_ids for each file
    file_ids = [str(uuid.uuid4()) for _ in range(len(all_files))]

    # Process all files in parallel using asyncio.gather()
    tasks = []
    for i, f in enumerate(all_files):
        logging.info(f"Creating task for file {i+1}/{len(all_files)}: {f.filename}")
        tasks.append(
            process_file_with_semaphore(
                file_handler, f, file_ids[i], is_image, username
            )
        )

    # Execute all tasks in parallel
    results = await asyncio.gather(*tasks)
    logging.info(f"All {len(all_files)} files processed concurrently")

    # Collect file_ids, filenames and other info from results
    processed_file_ids = []
    original_filenames = []
    is_tabular_flags = []
    statuses = []

    for i, result in enumerate(results):
        processed_file_ids.append(result["file_id"])
        original_filenames.append(all_files[i].filename)
        is_tabular_flags.append(result.get("is_tabular", False))
        statuses.append(result.get("status", "success"))

    return results, processed_file_ids, original_filenames, is_tabular_flags, statuses


def is_tabular_file(filename):
    """
    Check if a file is a tabular data file (CSV, Excel, DB).

    Args:
        filename: Name of the file to check

    Returns:
        bool: True if file is tabular, False otherwise
    """
    return filename.lower().endswith((".csv", ".xlsx", ".xls", ".db", ".sqlite"))


def process_tabular_file(
    background_tasks, file_id, temp_file_path, username, filename, is_image
):
    """
    Process tabular files in the background.

    Args:
        background_tasks: BackgroundTasks for async operations
        file_id: ID of the file
        temp_file_path: Path to the temporary file
        username: Username for file ownership
        filename: Original filename
        is_image: Flag indicating if file is an image

    Returns:
        bool: True if processed as tabular, False otherwise
    """
    if not file_id or not temp_file_path or not os.path.exists(temp_file_path):
        logging.warning(f"Invalid file or path for tabular processing: {filename}")
        return False

    # Prepare database in background based on the file type
    background_tasks.add_task(
        file_handler.prepare_db_from_file,
        file_path=temp_file_path,
        file_id=file_id,
        username=username,
    )

    # For tabular files, create file_info.json immediately
    file_metadata = {
        "embeddings_status": "completed",  # No embeddings needed
        "is_image": is_image,
        "is_tabular": True,
        "username": [username],
        "original_filename": filename,
        "file_id": file_id,
    }

    background_tasks.add_task(
        gcs_handler.upload_to_gcs,
        configs.gcp_resource.bucket_name,
        {
            "metadata": (
                file_metadata,
                f"file-embeddings/{file_id}/file_info.json",
            )
        },
    )

    logging.info(f"Scheduled tabular file processing for {filename} with ID {file_id}")
    return True


def is_document_file(file_extension, is_image):
    """
    Check if a file needs embeddings (PDF, TXT, etc.).

    Args:
        file_extension: Extension of the file
        is_image: Flag indicating if file is an image

    Returns:
        bool: True if file needs document embeddings, False otherwise
    """
    return file_extension in [".pdf", ".txt", ".doc", ".docx"] or is_image


def process_document_file(
    background_tasks, file_id, temp_file_path, username_list, file_metadata=None
):
    """
    Process a document file that needs embeddings.
    With the decoupled approach, local embeddings are generated immediately
    and cloud operations run as background tasks.

    Args:
        background_tasks: BackgroundTasks for async operations
        file_id: ID of the file
        temp_file_path: Path to the temporary file
        username_list: List of usernames for file ownership
        file_metadata: Optional file-specific metadata to pass through the chain
    """
    if not file_id or not temp_file_path:
        logging.warning(f"Invalid file or path for document processing: {file_id}")
        return

    # Get existing metadata from GCS handler if available
    if not file_metadata:
        file_metadata = gcs_handler.temp_metadata

    # Add username_list to metadata
    if file_metadata and username_list:
        file_metadata["username"] = username_list

    logging.info(
        f"Processing document with file_id {file_id}, metadata: {file_metadata}"
    )

    # Initialize local embeddings_status if not present
    if file_metadata and "embeddings_status" not in file_metadata:
        file_metadata["embeddings_status"] = "in_progress"

    # Schedule embedding creation in the background
    # The modified embedding_handler will mark embeddings as "ready_for_chat"
    # as soon as local embeddings are available
    background_tasks.add_task(
        create_embeddings_background,
        file_id,
        temp_file_path,
        embedding_handler,
        configs,
        SessionLocal,
        username_list,
        file_metadata,
    )
    logging.info(f"Scheduled embeddings creation for file ID {file_id}")


def format_upload_response(processed_file_ids, original_filenames, is_tabular_flags):
    """
    Format the response for the upload_file endpoint.

    Args:
        processed_file_ids: List of file IDs
        original_filenames: List of original filenames
        is_tabular_flags: List of flags indicating if files are tabular

    Returns:
        JSONResponse: Formatted response
    """
    # Generate a unique session_id for this upload batch
    session_id = str(uuid.uuid4())

    # Return a properly formatted response with all IDs
    response_data = {
        "file_ids": processed_file_ids,
        "original_filenames": original_filenames,
        "status": "success",
        "message": f"Successfully processed {len(processed_file_ids)} files",
        "session_id": session_id,  # Include the session_id in the response
    }

    # Set multi_file_mode flag if more than one file
    if len(processed_file_ids) > 1:
        response_data["multi_file_mode"] = True
    else:
        response_data["multi_file_mode"] = False
        # For backward compatibility, include single file ID and tabular flag
        if processed_file_ids:
            response_data["file_id"] = processed_file_ids[0]
            response_data["is_tabular"] = (
                is_tabular_flags[0] if is_tabular_flags else False
            )

    logging.info(
        f"Returning file upload response: file_ids={response_data['file_ids']}, "
        f"multi_file_mode={response_data.get('multi_file_mode', False)}"
    )
    return JSONResponse(content=response_data)


async def process_files_by_type(
    background_tasks,
    all_files,
    results,
    processed_file_ids,
    original_filenames,
    is_tabular_flags,
    is_image,
    username,
):
    """Process each uploaded file based on its type (tabular or document)."""

    # Separate document files and tabular files for different processing
    document_files = []
    tabular_files = []

    for i, result in enumerate(results):
        file_id = processed_file_ids[i]
        temp_file_path = result.get("temp_file_path")

        # Skip invalid files
        if not file_id or not temp_file_path:
            logging.warning(f"Skipping invalid file: {original_filenames[i]}")
            continue

        # Split files by type for batch processing
        if is_tabular_file(all_files[i].filename):
            tabular_files.append(
                {
                    "index": i,
                    "file_id": file_id,
                    "temp_file_path": temp_file_path,
                    "filename": all_files[i].filename,
                    "result": result,
                }
            )
        else:
            document_files.append(
                {
                    "index": i,
                    "file_id": file_id,
                    "temp_file_path": temp_file_path,
                    "filename": all_files[i].filename,
                    "result": result,
                }
            )

    # Process tabular files individually (they're handled differently)
    for file_info in tabular_files:
        i = file_info["index"]
        is_tabular_flags[i] = process_tabular_file(
            background_tasks,
            file_info["file_id"],
            file_info["temp_file_path"],
            username,
            file_info["filename"],
            is_image,
        )

    # Process document files in parallel if there are multiple
    if len(document_files) > 1:
        await process_document_files_parallel(document_files, username, is_image)
    # Process single document file normally
    elif len(document_files) == 1:
        file_info = document_files[0]
        await process_document_type_file(
            background_tasks,
            file_info["file_id"],
            file_info["temp_file_path"],
            file_info["filename"],
            file_info["result"],
            username,
            is_image,
        )


async def process_document_type_file(
    background_tasks, file_id, temp_file_path, filename, result, username, is_image
):
    """Process a document file, handling new uploads and existing files differently."""

    # Check for document files that need embeddings
    file_extension = os.path.splitext(filename)[1].lower()

    # Process new document file
    if is_document_file(file_extension, is_image):
        logging.info(f"Processing file with embedding support: {filename}")

        # Get document metadata
        file_metadata = result.get("metadata")
        if not file_metadata:
            local_gcs_handler = GCSHandler(configs)
            file_metadata = local_gcs_handler.get_file_info(file_id)

        # Create embeddings for document file
        process_document_file(
            background_tasks, file_id, temp_file_path, [username], file_metadata
        )

    # Handle existing files - check if they need new embeddings
    if result["status"] == "existing":
        await handle_existing_file(
            background_tasks, file_id, temp_file_path, username, result
        )


async def process_document_files_parallel(document_files, username, is_image):
    """
    Process multiple document files in parallel, handling both new uploads and existing files.
    This significantly speeds up embedding creation when multiple files are uploaded.

    Args:
        document_files: List of document file information dictionaries
        username: Username for file ownership
        is_image: Flag indicating if files are images
    """
    # Import here to avoid circular import
    from rtl_rag_chatbot_api.chatbot.parallel_embedding_creator import (
        create_embeddings_parallel,
    )

    # Process all document files in parallel - including existence and embeddings checks
    async def process_single_file(file_info):
        file_id = file_info["file_id"]
        temp_file_path = file_info["temp_file_path"]
        filename = file_info["filename"]
        result = file_info["result"]

        logging.info(f"Processing file in parallel: {filename} (ID: {file_id})")

        # Check file extension
        file_extension = os.path.splitext(filename)[1].lower()
        if not is_document_file(file_extension, is_image):
            logging.info(f"Skipping non-document file: {filename}")
            return None

        # Initialize handlers within the task to avoid sharing between tasks
        gcs_handler = GCSHandler(configs)

        # Check existing files - now done in parallel
        if result.get("status") == "existing":
            existing_file_info = gcs_handler.get_file_info(file_id)
            if existing_file_info:
                # Check if embeddings exist
                embeddings_exist = embedding_handler.check_embeddings_exist(
                    file_id, "default"
                )["embeddings_exist"]
                if embeddings_exist:
                    # Just update username - this is fast so we can do it synchronously
                    current_username_list = existing_file_info.get("username", [])
                    if not isinstance(current_username_list, list):
                        current_username_list = [current_username_list]

                    if username not in current_username_list:
                        current_username_list.append(username)
                        # Update username in background without waiting
                        asyncio.create_task(
                            update_username_in_background(
                                file_id, current_username_list
                            )
                        )
                        logging.info(
                            f"Scheduled username list update for file {file_id}: {current_username_list}"
                        )
                    return None  # No need for embedding creation

        # If we reach here, file needs embeddings
        file_metadata = result.get("metadata") or gcs_handler.get_file_info(file_id)
        return {
            "file_id": file_id,
            "temp_file_path": temp_file_path,
            "username_list": [username],
            "file_metadata": file_metadata,
        }

    # Helper function to update username lists in background
    async def update_username_in_background(file_id, username_list):
        try:
            gcs_handler = GCSHandler(configs)
            gcs_handler.update_username_list(file_id, username_list)
            logging.info(
                f"Updated username list in background for file {file_id}: {username_list}"
            )
        except Exception as e:
            logging.error(f"Error updating username list for file {file_id}: {str(e)}")

    # Run parallel processing for all files - including existence checks
    logging.info(
        f"Starting parallel processing for {len(document_files)} document files"
    )
    tasks = [process_single_file(file_info) for file_info in document_files]
    processing_results = await asyncio.gather(*tasks)

    # Filter out None results (files that don't need embedding)
    files_to_process = [result for result in processing_results if result]

    # Process all files needing embeddings in parallel
    if files_to_process:
        # Extract lists for parallel processing
        file_ids = [f["file_id"] for f in files_to_process]
        file_paths = [f["temp_file_path"] for f in files_to_process]
        username_lists = [f["username_list"] for f in files_to_process]
        file_metadata_list = [f["file_metadata"] for f in files_to_process]

        # Determine optimal number of concurrent tasks
        # Start with 4 workers or match number of files if fewer
        max_concurrent = min(len(files_to_process), 4)

        logging.info(
            f"Starting parallel embedding creation for {len(files_to_process)} files with {max_concurrent} workers"
        )

        # Run parallel embedding creation
        await create_embeddings_parallel(
            file_ids,
            file_paths,
            embedding_handler,
            configs,
            SessionLocal,
            username_lists,
            file_metadata_list,
            max_concurrent,
        )

        logging.info(
            f"Completed parallel embedding creation for {len(files_to_process)} files"
        )
    else:
        logging.info(
            "No files need embedding creation, all files already have valid embeddings"
        )


async def handle_existing_file(
    background_tasks, file_id, temp_file_path, username, result
):
    """Handle an existing file, checking if embeddings need to be recreated."""

    logging.info(f"File {file_id} already exists, checking if it needs new embeddings")

    # Check if embeddings exist for both model types
    google_result = await embedding_handler.check_embeddings_exist(
        file_id, "gemini-flash"
    )
    azure_result = await embedding_handler.check_embeddings_exist(
        file_id, "gpt_4o_mini"
    )

    # Get existing username list to ensure we preserve all users
    gcs_handler = GCSHandler(configs)
    current_file_info = gcs_handler.get_file_info(file_id)
    current_username_list = current_file_info.get("username", [])

    # Ensure current_username_list is a list
    if not isinstance(current_username_list, list):
        current_username_list = [current_username_list]

    # If embeddings already exist, just update the username list directly
    if google_result["embeddings_exist"] and azure_result["embeddings_exist"]:
        await update_username_for_existing_file(
            file_id, username, current_username_list
        )
    # If any embeddings are missing, recreate them
    else:
        await recreate_embeddings_for_existing_file(
            background_tasks,
            file_id,
            temp_file_path,
            username,
            result,
            current_username_list,
        )


async def update_username_for_existing_file(file_id, username, current_username_list):
    """Update the username list for an existing file with valid embeddings."""

    logging.info(
        f"Valid embeddings exist for file_id {file_id}, updating username list only"
    )

    # Add new username if not already present
    if username not in current_username_list:
        current_username_list.append(username)

    # Update username list directly
    gcs_handler = GCSHandler(configs)
    gcs_handler.update_username_list(file_id, current_username_list)
    logging.info(f"Updated username list for existing file: {current_username_list}")


async def recreate_embeddings_for_existing_file(
    background_tasks, file_id, temp_file_path, username, result, current_username_list
):
    """Recreate embeddings for an existing file with missing embeddings."""

    logging.info(f"Recreating embeddings for existing file {file_id}")

    # Get existing username list + new username
    username_list = result.get("username_list", [username])

    # Ensure current usernames are preserved
    for current_user in current_username_list:
        if current_user not in username_list:
            username_list.append(current_user)

    process_document_file(background_tasks, file_id, temp_file_path, username_list)


@app.post("/file/upload", response_model=FileUploadResponse)
async def upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    files: List[UploadFile] = File(None),
    is_image: bool = Form(False),
    username: str = Form(...),
    urls: str = Form(None),
    current_user=Depends(get_current_user),
):
    """
    Handles file upload and automatically creates embeddings for PDFs and images.
    Supports three modes of operation:
    1. URL processing: Extract content from URLs and create embeddings
    2. Multiple file upload: Process multiple files concurrently using asyncio.gather()
    3. Single file upload: Process a single file (for backward compatibility)

    All file processing is done asynchronously and in parallel when multiple files are uploaded.
    Embeddings creation and other post-processing happens in background tasks.
    """
    try:
        # Handle URL processing
        if urls:
            response_data = await process_url_content(
                file_handler, embedding_handler, urls, username, background_tasks
            )
            return JSONResponse(content=response_data)

        # Combine files from both parameters
        all_files = prepare_file_list(file, files)

        # Process all files in parallel
        if len(all_files) > 0:
            logging.info(f"Processing {len(all_files)} files with parallel processing")

            # Process files in parallel and collect results
            (
                results,
                processed_file_ids,
                original_filenames,
                is_tabular_flags,
                statuses,
            ) = await process_files_in_parallel(
                file_handler, all_files, is_image, username
            )

            # Process each file based on its type
            await process_files_by_type(
                background_tasks,
                all_files,
                results,
                processed_file_ids,
                original_filenames,
                is_tabular_flags,
                is_image,
                username,
            )

            # Format and return the response
            return format_upload_response(
                processed_file_ids, original_filenames, is_tabular_flags
            )

        # No files provided case
        return JSONResponse(
            status_code=400,
            content={
                "message": "No files provided in either 'file' or 'files' parameters."
            },
        )

    except Exception as e:
        logging.exception(f"Error in upload_file: {str(e)}")
        # Return a more detailed error response with stack trace in dev mode
        if configs.app.dev:
            import traceback

            stack_trace = traceback.format_exc()
            raise HTTPException(
                status_code=500,
                detail={
                    "error": str(e),
                    "stack_trace": stack_trace,
                    "message": "File upload failed. See error details.",
                },
            )
        else:
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


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


async def update_metadata_in_background(
    configs, file_id, updated_metadata, error_context=""
):
    """
    Asynchronous function for non-critical GCS operations that can be run in background.
    Updates file metadata in GCS without blocking the main workflow.

    Args:
        configs: Application configuration
        file_id: The ID of the file to update metadata for
        updated_metadata: New metadata to update
        error_context: Optional context for error messages
    """
    try:
        background_gcs = GCSHandler(configs)
        background_gcs.update_file_info(file_id, updated_metadata)
        logging.info(f"Updated metadata in background for file {file_id}")
    except Exception as update_error:
        logging.error(f"Error updating metadata {error_context}: {str(update_error)}")


async def update_usernames_in_background(configs, file_id, new_usernames):
    """
    Asynchronous function to merge and update usernames in background.
    Preserves existing usernames and adds new ones without duplicates.

    Args:
        configs: Application configuration
        file_id: The ID of the file to update usernames for
        new_usernames: List of new usernames to add
    """
    try:
        # Initialize GCS handler within the task
        gcs_handler = GCSHandler(configs)

        # Get current file info to preserve existing usernames
        current_file_info = gcs_handler.get_file_info(file_id)
        current_username_list = current_file_info.get("username", [])

        # Ensure current_username_list is a list
        if not isinstance(current_username_list, list):
            current_username_list = [current_username_list]

        # Merge username lists without duplicates
        merged_usernames = list(set(current_username_list))

        # Add any new usernames that aren't in the merged list
        for username in new_usernames:
            if username not in merged_usernames:
                merged_usernames.append(username)

        # Update the file_info.json with the combined username list
        gcs_handler.update_username_list(file_id, merged_usernames)
        logging.info(
            f"Updated username list in background for file_id {file_id}: {merged_usernames}"
        )
    except Exception as update_error:
        logging.error(f"Error updating username list: {str(update_error)}")


async def calculate_file_hash(temp_file_path, configs, gcs_handler):
    """
    Calculate the hash of a file asynchronously.

    Args:
        temp_file_path: Path to the temporary file
        configs: Application configuration
        gcs_handler: GCS handler for file operations

    Returns:
        str: The calculated file hash or None if calculation fails
    """
    try:
        with open(temp_file_path, "rb") as f:
            file_content = f.read()
            from rtl_rag_chatbot_api.chatbot.file_handler import FileHandler

            dummy_handler = FileHandler(configs, gcs_handler)
            file_hash = dummy_handler.calculate_file_hash(file_content)
            del file_content  # Free memory
            return file_hash
    except Exception as hash_error:
        logging.error(f"Failed to calculate file hash: {str(hash_error)}")
        return None


async def initialize_file_metadata(
    file_id, temp_file_path, configs, username_list=None
):
    """
    Initialize file metadata by fetching existing metadata or creating new metadata.
    Also calculates file hash if temp file exists.

    Args:
        file_id: The ID of the file
        temp_file_path: Path to the temporary file
        configs: Application configuration
        username_list: Optional list of usernames to include in metadata

    Returns:
        dict: Initialized file metadata
    """
    # Run concurrent tasks for metadata initialization
    gcs_handler = GCSHandler(configs)
    tasks = []

    # Task 1: Fetch existing metadata from GCS
    tasks.append(
        asyncio.create_task(asyncio.to_thread(gcs_handler.get_file_info, file_id))
    )

    # Task 2: Calculate file hash if temp file exists
    file_hash_task = None
    if temp_file_path and os.path.exists(temp_file_path):
        file_hash_task = asyncio.create_task(
            calculate_file_hash(temp_file_path, configs, gcs_handler)
        )
        tasks.append(file_hash_task)

    # Wait for all initialization tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    existing_metadata = (
        results[0] if results and not isinstance(results[0], Exception) else None
    )

    file_hash = None
    if file_hash_task:
        file_hash_result_idx = tasks.index(file_hash_task)
        if len(results) > file_hash_result_idx:
            file_hash = results[file_hash_result_idx]
            if file_hash:
                logging.info(f"Calculated file_hash {file_hash} for {file_id}")

    # Use existing metadata or create new
    if existing_metadata:
        file_metadata = existing_metadata
        # Ensure file_id is correct
        if file_metadata.get("file_id") != file_id:
            file_metadata["file_id"] = file_id
        # Ensure embeddings_status is set
        if "embeddings_status" not in file_metadata:
            file_metadata["embeddings_status"] = "in_progress"
    else:
        logging.info(f"Creating new metadata for {file_id}")
        file_metadata = {
            "file_id": file_id,
            "username": username_list if username_list else [],
            "embeddings_status": "in_progress",
        }

    # Add file hash if calculated
    if file_hash:
        file_metadata["file_hash"] = file_hash

    return file_metadata


async def run_cleanup_after_error(configs, SessionLocal, file_id):
    """
    Runs cleanup operations after an error occurs during embedding creation.

    Args:
        configs: Application configuration
        SessionLocal: Database session factory
        file_id: The ID of the file to clean up
    """
    try:
        await asyncio.to_thread(
            CleanupCoordinator(configs, SessionLocal).cleanup_chroma_instance,
            file_id,
            include_gcs=True,
        )
    except Exception as cleanup_error:
        logging.error(
            f"Error during cleanup after embedding failure: {str(cleanup_error)}"
        )


async def create_embeddings_background(
    file_id: str,
    temp_file_path: str,
    embedding_handler,
    configs,
    SessionLocal,
    username_list=None,
    file_metadata=None,
):
    """
    Background task for creating embeddings.
    With the decoupled approach, local embeddings are created first,
    marked as 'ready_for_chat', and cloud operations happen in the background.

    This function is optimized for parallel execution with multiple files.

    Args:
        file_id: The ID of the file to create embeddings for
        temp_file_path: Path to the temporary file
        embedding_handler: Handler for creating embeddings
        configs: Application configuration
        SessionLocal: Database session factory
        username_list: Optional list of usernames to preserve in file_info.json
        file_metadata: Optional file metadata to pass through the chain
    """
    try:
        # Initialize metadata if not provided
        if not file_metadata:
            file_metadata = await initialize_file_metadata(
                file_id, temp_file_path, configs, username_list
            )

        # Update metadata status to in_progress asynchronously
        file_metadata["embeddings_status"] = "in_progress"
        asyncio.create_task(
            update_metadata_in_background(
                configs, file_id, file_metadata, "before embedding creation"
            )
        )

        # Log details about the metadata
        file_hash = file_metadata.get("file_hash") if file_metadata else None
        logging.info(f"Using file-specific metadata for {file_id}: {file_metadata}")
        logging.info(f"File hash for {file_id}: {file_hash}")

        # CRITICAL SECTION: Create embeddings with file-specific metadata
        # This is the core functionality that must be completed before returning
        embedding_result = await embedding_handler.ensure_embeddings_exist(
            file_id=file_id, temp_file_path=temp_file_path, file_metadata=file_metadata
        )

        # Process username updates in background if embedding creation succeeded
        if embedding_result["status"] != "error" and username_list:
            asyncio.create_task(
                update_usernames_in_background(configs, file_id, username_list)
            )

        # Handle errors in embedding creation
        if embedding_result["status"] == "error":
            logging.error(
                f"Error creating embeddings for {file_id}: {embedding_result.get('message', 'Unknown error')}"
            )
            # Run cleanup in background to avoid blocking
            asyncio.create_task(run_cleanup_after_error(configs, SessionLocal, file_id))

        return embedding_result

    except Exception as e:
        logging.error(f"Error in create_embeddings_background for {file_id}: {str(e)}")
        # Run cleanup in background to avoid blocking
        asyncio.create_task(run_cleanup_after_error(configs, SessionLocal, file_id))
        return {"status": "error", "message": str(e), "file_id": file_id}


@app.post("/embeddings/check", response_model=Dict[str, Any])
async def check_embeddings(
    request: EmbeddingsCheckRequest, current_user=Depends(get_current_user)
):
    """
    Check if embeddings exist for the specified file and model choice.

    Args:
        request (EmbeddingsCheckRequest): Request containing file_id and model_choice
        current_user: Authenticated user information

    Returns:
        Dict containing:
            - embeddings_exist (bool): Whether embeddings exist
            - model_type (str): Type of model (azure/google)
            - file_id (str): The checked file ID
    """
    try:
        # Initialize the embedding handler
        embedding_handler = EmbeddingHandler(
            configs=configs, gcs_handler=gcs_handler, file_info=[]
        )

        # Check if embeddings exist
        result = await embedding_handler.check_embeddings_exist(
            file_id=request.file_id, model_choice=request.model_choice
        )
        return result

    except Exception as e:
        logging.error(f"Error checking embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/embeddings/status/{file_id}", response_model=Dict[str, Any])
async def get_embedding_status(file_id: str, current_user=Depends(get_current_user)):
    """
    Get the current status of embeddings for a specific file.
    The frontend can poll this endpoint to know when embeddings are ready for chat.

    Args:
        file_id: The ID of the file to check
        current_user: Authenticated user information

    Returns:
        Dict containing:
            - status: Current status of embeddings ("not_started", "in_progress", "ready_for_chat", or "completed")
            - can_chat: Whether chat is available with this file
            - file_id: The checked file ID
            - message: Human-readable message about the current status
    """
    try:
        # Check local file_info.json first
        local_info_path = os.path.join("./chroma_db", file_id, "file_info.json")

        if os.path.exists(local_info_path):
            with open(local_info_path, "r") as f:
                file_info = json.load(f)

            status = file_info.get("embeddings_status", "not_started")
            can_chat = status in ["ready_for_chat", "completed"]

            return {
                "status": status,
                "can_chat": can_chat,
                "file_id": file_id,
                "message": "Ready for chat" if can_chat else "Embeddings not ready yet",
            }

        # If no local file, check GCS
        try:
            # Download and check file_info.json from GCS
            file_info = gcs_handler.get_file_info(file_id)

            if file_info:
                status = file_info.get("embeddings_status", "not_started")
                can_chat = status in ["ready_for_chat", "completed"]

                return {
                    "status": status,
                    "can_chat": can_chat,
                    "file_id": file_id,
                    "message": "Ready for chat"
                    if can_chat
                    else "Embeddings not ready yet",
                }

        except Exception as gcs_error:
            logging.warning(f"Could not get file info from GCS: {str(gcs_error)}")

        # If we get here, embeddings don't exist or are in an unknown state
        return {
            "status": "not_started",
            "can_chat": False,
            "file_id": file_id,
            "message": "Embeddings not found or still being generated",
        }

    except Exception as e:
        logging.error(f"Error checking embedding status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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
            initialized_models[model_key] = {"model": model, "is_tabular": is_tabular}
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
        # Accept both "ready_for_chat" and "completed" as valid states
        if file_info.get("embeddings_status") not in ["ready_for_chat", "completed"]:
            raise HTTPException(
                status_code=400,
                detail="Embeddings are not ready yet. Please wait a moment.",
            )
        print("No local embeddings found, downloading from GCS")
        gcs_handler.download_files_from_folder_by_id(query.file_id)

    # Initialize model with ChromaDB
    if is_gemini:
        # For GeminiHandler, create instance first then call initialize()
        model = GeminiHandler(configs, gcs_handler)
        model.initialize(
            model=query.model_choice,
            file_id=query.file_id,
            embedding_type="google",
            collection_name=f"rag_collection_{query.file_id}",
            user_id=query.user_id,
        )
    else:
        # For AzureChatbot, all initialization happens in the constructor
        # Pass all required parameters directly
        model = Chatbot(
            configs=configs,
            gcs_handler=gcs_handler,
            model_choice=query.model_choice,
            file_id=query.file_id,
            collection_name_prefix="rag_collection_",
            user_id=query.user_id,
        )
        # AzureChatbot doesn't have an initialize() method, it's all done in __init__

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


def handle_visualization(
    response: Any,
    query: Query,
    is_tabular: bool,
    configs: dict,
) -> JSONResponse:
    """
    Handle visualization generation for chat responses.

    Args:
        response: The initial response from the model
        query: The query object containing user request details
        is_tabular: Whether the data is tabular
        configs: Application configuration dictionary

    Returns:
        JSONResponse containing the chart configuration

    Raises:
        HTTPException: If visualization generation fails
    """
    try:
        if is_tabular:
            current_question = query.text[-1] + response + VISUALISATION_PROMPT
            if query.model_choice.startswith("gemini"):
                response = get_gemini_non_rag_response(
                    configs, current_question, query.model_choice
                )
            else:
                response = get_azure_non_rag_response(configs, current_question)

        # Parse response into JSON
        if isinstance(response, str):
            response = response.replace("True", "true").replace("False", "false")
            response = response.strip()
            response = response.replace("```json", "").replace("```", "").strip()
            chart_config = json.loads(response)
        else:
            chart_config = response
        logging.info(f"Generated chart config: {chart_config}")
        return JSONResponse(
            content={
                "chart_config": chart_config,
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
        raise HTTPException(status_code=500, detail="Failed to generate chart")


async def _initialize_chat_model(
    query: Query,
    configs: dict,
    gcs_handler: GCSHandler,
    initialized_models: dict,  # This will be modified
    model_key: str,
    is_multi_file: bool,
    all_file_infos: Dict[str, Any],
    file_id_logging: str,  # For logging
) -> Tuple[Any, Optional[bool]]:  # Returns (model, determined_is_tabular)
    """Helper function to initialize and cache chat models."""
    model: Any = None
    determined_is_tabular: Optional[bool] = None

    if is_multi_file:
        # First check if any of the files are tabular
        has_tabular_files = False
        tabular_file_ids = []
        non_tabular_file_ids = []

        # Check each file to see if it has a SQLite database
        for file_id in query.file_ids:
            db_path = f"./chroma_db/{file_id}/tabular_data.db"
            if os.path.exists(db_path):
                has_tabular_files = True
                tabular_file_ids.append(file_id)
            else:
                non_tabular_file_ids.append(file_id)

        # If we have mixed files (some tabular, some not), we need special handling
        if has_tabular_files and non_tabular_file_ids:
            logging.info(
                "Mixed file types detected in multi-file chat (tabular + non-tabular)"
            )
            logging.info(f"Tabular files: {tabular_file_ids}")
            logging.info(f"Non-tabular files: {non_tabular_file_ids}")
            # For mixed files, use RAG model as it can handle non-tabular content
            determined_is_tabular = False
            logging.info("Using RAG model for mixed file types")
            model = Chatbot(
                configs=configs,
                gcs_handler=gcs_handler,
                model_choice=query.model_choice,
                file_ids=query.file_ids,  # type: ignore
                all_file_infos=all_file_infos,
            )
        elif has_tabular_files and not non_tabular_file_ids:
            # All files are tabular - special handling required
            logging.info(f"All files are tabular in multi-file: {query.file_ids}")
            determined_is_tabular = True

            # When all files are tabular, we should use the TabularDataHandler with all files
            logging.info("Initializing TabularDataHandler with all tabular files")
            if len(tabular_file_ids) > 0:
                try:
                    # Pass all file_ids to the TabularDataHandler to handle multiple databases
                    model = TabularDataHandler(
                        configs,
                        file_id=tabular_file_ids[
                            0
                        ],  # For backward compatibility, set primary file_id
                        model_choice=query.model_choice,
                        file_ids=tabular_file_ids,  # Pass all tabular file_ids
                    )
                    logging.info(
                        f"Successfully initialized TabularDataHandler for {len(tabular_file_ids)} files"
                    )
                except Exception as e:
                    logging.error(f"Failed to initialize TabularDataHandler: {str(e)}")
                    # Fall back to standard Chatbot even though it won't work well with tabular data
                    logging.info("Falling back to RAG model for tabular data")
                    model = Chatbot(
                        configs=configs,
                        gcs_handler=gcs_handler,
                        model_choice=query.model_choice,
                        file_ids=query.file_ids,  # type: ignore
                        all_file_infos=all_file_infos,
                    )
            else:
                # Should never happen as we already checked tabular_file_ids is not empty
                logging.warning("No tabular files found despite earlier check")
                determined_is_tabular = False
                model = Chatbot(
                    configs=configs,
                    gcs_handler=gcs_handler,
                    model_choice=query.model_choice,
                    file_ids=query.file_ids,  # type: ignore
                    all_file_infos=all_file_infos,
                )
        else:
            # All files are non-tabular - use appropriate model based on model_choice
            logging.info(f"All files are non-tabular: {non_tabular_file_ids}")
            determined_is_tabular = False

            # Check if this is a Gemini model
            is_gemini = query.model_choice.lower() in ["gemini-flash", "gemini-pro"]

            if is_gemini:
                # Use GeminiHandler for Gemini models
                logging.info(
                    f"Using GeminiHandler for multi-file with model: {query.model_choice}"
                )
                model = GeminiHandler(
                    configs=configs,
                    gcs_handler=gcs_handler,
                    model_choice=query.model_choice,
                    file_ids=query.file_ids,  # type: ignore
                    all_file_infos=all_file_infos,
                    user_id=query.user_id,
                )
            else:
                # Use AzureChatbot for Azure models
                logging.info(
                    f"Using AzureChatbot for multi-file with model: {query.model_choice}"
                )
                model = Chatbot(
                    configs=configs,
                    gcs_handler=gcs_handler,
                    model_choice=query.model_choice,
                    file_ids=query.file_ids,  # type: ignore
                    all_file_infos=all_file_infos,
                )

        logging.info(f"Model initialized for multi-file: {query.file_ids}")
        logging.info(f"  User: {query.user_id}, Model: {query.model_choice}")
        logging.info(f"  Is Tabular: {determined_is_tabular}")
    elif query.file_id:
        # For single file, check for DB existence which also determines 'is_tabular'
        _db_path, is_tabular_single, model_single = check_db_existence(
            query.file_id, query, configs, initialized_models
        )
        determined_is_tabular = is_tabular_single
        model = model_single  # This could be a TabularDataHandler or None

        if (
            not determined_is_tabular
        ):  # If not tabular, it's a RAG model for single file
            single_file_info_for_init = all_file_infos.get(query.file_id)
            if not single_file_info_for_init:
                raise HTTPException(
                    status_code=500,
                    detail="Internal error: file_info missing for single file RAG model initialization.",
                )
            logging.info(f"Initializing RAG model for single file: {query.file_id}")
            logging.info(f"  User: {query.user_id}, Model: {query.model_choice}")
            model = initialize_rag_model(
                query, configs, gcs_handler, single_file_info_for_init
            )
    else:
        # This case should ideally not be reached if prior checks in `chat` are correct
        raise HTTPException(
            status_code=500,
            detail=(
                "Internal server error: Model initialization path unclear "
                "(neither multi-file nor single file_id provided)."
            ),
        )

    # Cache the newly initialized model
    initialized_models[model_key] = {
        "model": model,
        "is_tabular": determined_is_tabular,
    }
    logging.info(f"Model initialized and cached with key: {model_key}")
    logging.info(f"  File(s): {file_id_logging}, User: {query.user_id}")
    logging.info(f"  Model: {query.model_choice}, Is Tabular: {determined_is_tabular}")
    return model, determined_is_tabular


async def _log_session_info(query: Query) -> None:
    """Log session information if provided in the query.

    Args:
        query: Query containing optional session_id
    """
    if query.session_id:
        logging.info(f"Request contains session_id: {query.session_id}")


async def _process_single_file(query: Query, gcs_handler: GCSHandler) -> Dict[str, Any]:
    """Process information for a single file query.

    Args:
        query: Query containing file_id
        gcs_handler: GCS handler for file operations

    Returns:
        Dictionary containing processed single file information
    """
    file_id_logging = f"file_id: {query.file_id}"
    logging.info(f"Single-file chat request for {file_id_logging}")
    file_info_single = gcs_handler.get_file_info(query.file_id)

    # Log session information if provided
    if query.session_id:
        logging.info(
            f"Processing single file {query.file_id} for session {query.session_id}"
        )

    if not file_info_single:
        embeddings_status = gcs_handler.check_embeddings_status(query.file_id)
        if embeddings_status == "in_progress":
            raise HTTPException(
                status_code=400,
                detail="URL content is still being processed. Please wait a moment and try again.",
            )
        raise HTTPException(
            status_code=404,
            detail="File appears to be corrupted, empty, or not found."
            " Please try uploading/selecting a different file.",
        )

    all_file_infos = {query.file_id: file_info_single}
    model_key = f"{query.file_id}_{query.user_id}_{query.model_choice}"

    return {
        "all_file_infos": all_file_infos,
        "model_key": model_key,
        "file_id_logging": file_id_logging,
    }


async def _check_file_embeddings(file_id: str, model_choice: str) -> Dict[str, Any]:
    """Check if embeddings exist for a specific file and model choice.

    Args:
        file_id: The ID of the file to check
        model_choice: The model choice to check embeddings for

    Returns:
        Dictionary containing embeddings check result
    """
    result = await embedding_handler.check_embeddings_exist(file_id, model_choice)
    if not result["embeddings_exist"]:
        raise HTTPException(
            status_code=400,
            detail=f"Embeddings not found for file {file_id}. Please create embeddings first.",
        )
    return result


async def _classify_file_types(file_ids: List[str]) -> Tuple[List[str], List[str]]:
    """Classify files as tabular or non-tabular.

    Args:
        file_ids: List of file IDs to classify

    Returns:
        Tuple of (tabular_files, non_tabular_files) lists
    """
    tabular_files = []
    non_tabular_files = []

    for f_id in file_ids:
        # Check if the file has a SQLite database (indicating tabular data)
        db_path = f"./chroma_db/{f_id}/tabular_data.db"
        is_file_tabular = os.path.exists(db_path)

        if is_file_tabular:
            tabular_files.append(f_id)
        else:
            non_tabular_files.append(f_id)

    return tabular_files, non_tabular_files


async def _get_file_info_multi(
    file_ids: List[str], gcs_handler: GCSHandler
) -> Dict[str, Dict[str, Any]]:
    """Get file information for multiple files.

    Args:
        file_ids: List of file IDs to get information for
        gcs_handler: GCS handler for file operations

    Returns:
        Dictionary mapping file IDs to their information
    """
    all_file_infos = {}

    for f_id in file_ids:
        # Get file info
        f_info = gcs_handler.get_file_info(f_id)
        if not f_info:
            embeddings_status = gcs_handler.check_embeddings_status(f_id)
            if embeddings_status == "in_progress":
                raise HTTPException(
                    status_code=400,
                    detail=f"URL content for file {f_id} is still being processed. Please wait and try again.",
                )
            raise HTTPException(
                status_code=404,
                detail=f"File info not found for file_id: {f_id}. It may be corrupted or not yet processed.",
            )
        all_file_infos[f_id] = f_info

    return all_file_infos


async def _determine_tabular_status(
    tabular_files: List[str], non_tabular_files: List[str]
) -> bool:
    """Determine if the query is tabular based on file classification.

    Args:
        tabular_files: List of tabular file IDs
        non_tabular_files: List of non-tabular file IDs

    Returns:
        Boolean indicating if the query is tabular
    """
    if tabular_files and non_tabular_files:
        logging.info(
            f"Mixed file types detected: {len(tabular_files)} tabular and {len(non_tabular_files)} non-tabular"
        )
        logging.info(f"Tabular files: {tabular_files}")
        logging.info(f"Non-tabular files: {non_tabular_files}")
        # When mixed, we default to non-tabular mode for now as it's the safer option
        return False
    elif tabular_files and not non_tabular_files:
        logging.info(f"All files are tabular: {tabular_files}")
        # All files are tabular, but we still need to handle this specially
        return True
    else:
        logging.info(f"All files are non-tabular: {non_tabular_files}")
        return False


async def _process_multi_files(query: Query, gcs_handler: GCSHandler) -> Dict[str, Any]:
    """Process information for multiple file query.

    Args:
        query: Query containing file_ids
        gcs_handler: GCS handler for file operations

    Returns:
        Dictionary containing processed multi-file information
    """
    if not query.file_ids:
        raise HTTPException(
            status_code=400,
            detail="file_ids must be provided for multi-file chat.",
        )

    file_id_logging = f"file_ids: {query.file_ids}"
    logging.info(f"Multi-file chat request for {file_id_logging}")

    # The frontend is responsible for sending only the file_ids relevant to the current session
    # We trust that the file_ids provided in the query belong to the current session
    if query.session_id:
        logging.info(
            f"Processing files for session {query.session_id}: {query.file_ids}"
        )

    if not query.file_ids:
        raise HTTPException(
            status_code=404,
            detail="No files specified for this chat session. Please upload files first.",
        )

    # Check if embeddings exist for all files
    for file_id in query.file_ids:
        await _check_file_embeddings(file_id, query.model_choice)

    # Check each file to see if it's tabular
    tabular_files, non_tabular_files = await _classify_file_types(query.file_ids)

    # Get file info for all files
    all_file_infos = await _get_file_info_multi(query.file_ids, gcs_handler)

    # Determine if the query is tabular
    is_tabular = await _determine_tabular_status(tabular_files, non_tabular_files)

    # Create model key for caching
    files_key_part = "_".join(sorted(query.file_ids))
    model_key = f"multi_{files_key_part}_{query.user_id}_{query.model_choice}"

    return {
        "all_file_infos": all_file_infos,
        "model_key": model_key,
        "file_id_logging": file_id_logging,
        "is_tabular": is_tabular,
    }


async def _format_chat_response(
    response: Any,
    query: Query,
    generate_visualization: bool,
    is_tabular: bool,
    is_multi_file: bool,
    configs: dict,
) -> Dict[str, Any]:
    """
    Helper function to format the chat response based on response type and query parameters.

    Args:
        response: The raw response from the model
        query: Original query object
        generate_visualization: Whether visualization was requested
        is_tabular: Whether the response is for tabular data
        is_multi_file: Whether multiple files were queried
        configs: Application configuration

    Returns:
        Formatted response dictionary
    """
    if isinstance(response, list):
        return format_table_response(response)

    if generate_visualization:
        return handle_visualization(response, query, is_tabular, configs)

    final_response_data = {
        "response": str(response),
        "is_table": is_tabular if is_tabular is not None else False,
    }

    if is_multi_file and query.file_ids:
        final_response_data["sources"] = query.file_ids
    elif query.file_id:
        final_response_data["sources"] = [query.file_id]

    return final_response_data


async def _process_file_info(
    query: Query, gcs_handler: GCSHandler, generate_visualization: bool
) -> Dict[str, Any]:
    """
    Helper function to process file information for chat queries.

    Args:
        query: Query containing file_id or file_ids and optional session_id
        gcs_handler: GCS handler for file operations
        generate_visualization: Whether visualization was requested

    Returns:
        Dictionary containing processed file information
    """
    # Log session ID if provided
    await _log_session_info(query)

    # Determine if this is a multi-file request
    is_multi_file = bool(query.file_ids and len(query.file_ids) > 0)

    # Initialize variables
    all_file_infos = {}
    model_key = ""
    file_id_logging = ""
    is_tabular = None

    # Process based on request type
    if is_multi_file:
        # Process multi-file request
        multi_file_info = await _process_multi_files(query, gcs_handler)

        # Extract relevant information
        all_file_infos = multi_file_info["all_file_infos"]
        model_key = multi_file_info["model_key"]
        file_id_logging = multi_file_info["file_id_logging"]
        is_tabular = multi_file_info["is_tabular"]

        # Adjust visualization settings for multi-file
        if generate_visualization:
            logging.info(
                "Visualization for multi-file chat is not currently supported. Proceeding without visualization."
            )
            generate_visualization = False

    elif query.file_id:
        # Process single-file request
        single_file_info = await _process_single_file(query, gcs_handler)

        # Extract relevant information
        all_file_infos = single_file_info["all_file_infos"]
        model_key = single_file_info["model_key"]
        file_id_logging = single_file_info["file_id_logging"]

        # is_tabular is determined later for single files
    else:
        # Neither file_id nor file_ids provided
        raise HTTPException(
            status_code=400,
            detail="Either file_id (for single chat) or file_ids (for multi-chat) must be provided.",
        )

    # Return consolidated result with consistent structure
    return {
        "is_multi_file": is_multi_file,
        "all_file_infos": all_file_infos,
        "model_key": model_key,
        "file_id_logging": file_id_logging,
        "is_tabular": is_tabular,
        "generate_visualization": generate_visualization,
    }


async def _detect_visualization_need(question: str, configs: dict) -> bool:
    """
    Helper function to detect if visualization is needed based on user question.

    Args:
        question: The user's question to analyze
        configs: Application configuration dictionary

    Returns:
        Boolean indicating whether visualization is needed
    """
    generate_visualization = False
    should_visualize_filter = detect_visualization_need(question)

    if should_visualize_filter:
        question_for_detection = CHART_DETECTION_PROMPT + question
        vis_detection_response = get_gemini_non_rag_response(
            configs, question_for_detection, "gemini-flash"
        )
        if (
            vis_detection_response.lower() == "true"
            or "true" in vis_detection_response.lower()
        ):
            generate_visualization = True

    return generate_visualization


@app.post("/file/chat")
async def chat(query: Query, current_user=Depends(get_current_user)):
    """
    Process chat queries against document content using specified language models.

    Handles single or multiple files, multiple model types (Azure LLM, Gemini),
    and data formats (text, tabular for single files).
    Automatically initializes or switches models based on request parameters.

    Args:
        query (Query): Request body containing:
            - text (List[str]): User's query text history
            - file_id (Optional[str]): ID of the file to query (for single file chat)
            - file_ids (Optional[List[str]]): IDs of files to query (for multi-file chat)
            - model_choice (str): Selected model
            - user_id (str): User identifier
        current_user: Authenticated user information

    Returns:
        dict: Response containing the model's answer and source file information.
    """
    file_id_logging = ""
    try:
        if not query.text:
            raise HTTPException(status_code=400, detail="Text array cannot be empty")

        current_actual_question = query.text[-1]

        generate_visualization = await _detect_visualization_need(
            current_actual_question, configs
        )

        # Process file information and build the model key
        file_data = await _process_file_info(query, gcs_handler, generate_visualization)

        is_multi_file = file_data["is_multi_file"]
        all_file_infos = file_data["all_file_infos"]
        model_key = file_data["model_key"]
        file_id_logging = file_data["file_id_logging"]
        is_tabular = file_data["is_tabular"]
        generate_visualization = file_data["generate_visualization"]

        logging.info(f"Graphic generation flag: {generate_visualization}")
        logging.info(f"For {file_id_logging}")

        model_info = initialized_models.get(model_key)
        model = model_info["model"] if model_info else None

        if model_info:
            is_tabular = model_info["is_tabular"]
        elif is_multi_file:
            is_tabular = (
                False  # Already set, but good for clarity if model_info was None
            )
        # If single file and model_info is None, is_tabular is still None, will be set by check_db_existence

        try:
            if not model:
                # Call the helper function to initialize the model
                model, is_tabular = await _initialize_chat_model(
                    query=query,
                    configs=configs,
                    gcs_handler=gcs_handler,
                    initialized_models=initialized_models,
                    model_key=model_key,
                    is_multi_file=is_multi_file,
                    all_file_infos=all_file_infos,
                    file_id_logging=file_id_logging,
                )

            if generate_visualization and not is_tabular:
                question_to_model = current_actual_question + VISUALISATION_PROMPT
            else:
                question_to_model = current_actual_question

            if len(query.text) > 1:
                previous_messages = "\n".join(
                    [f"Previous message: {msg}" for msg in query.text[:-1]]
                )
                chat_context = (
                    f"{previous_messages}\nCurrent question: {question_to_model}"
                )
            else:
                chat_context = question_to_model

            # Todo: If multiple file and tablular collect all responses
            # and give to non rag and return answer.
            response = model.get_answer(chat_context)

            return await _format_chat_response(
                response=response,
                query=query,
                generate_visualization=generate_visualization,
                is_tabular=is_tabular,
                is_multi_file=is_multi_file,
                configs=configs,
            )

        finally:
            if isinstance(model, TabularDataHandler):
                model.cleanup()

    except HTTPException:
        raise
    except Exception as e:
        logging.error(
            f"Error in chat endpoint: {str(e)} for {file_id_logging}", exc_info=True
        )
        error_message = str(e)
        if "coroutine" in error_message.lower() or "async" in error_message.lower():
            error_message = (
                "Internal server error: An unexpected issue occurred "
                "while processing your request. Please try again."
            )
        raise HTTPException(status_code=500, detail=error_message)


@app.get("/available-models")
async def get_available_models(current_user=Depends(get_current_user)):
    """
    Endpoint to retrieve a list of available models including Azure LLM models, Gemini models,
    and image generation models.
    """
    # Get available models from config
    azure_models = list(configs.azure_llm.models.keys())
    gemini_models = ["gemini-flash", "gemini-pro"]
    # Add individual image models and the combined option
    image_models = ["dall-e-3", configs.vertexai_imagen.model_name, "Dalle + Imagen"]

    # Return combined list with model categories
    return {
        "models": azure_models + gemini_models + image_models,
        "model_types": {"text": azure_models + gemini_models, "image": image_models},
    }


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
        logging.exception(f"Error in get_neighbors: {str(e)}")
        # Return a more detailed error response with stack trace in dev mode
        if configs.app.dev:
            import traceback

            stack_trace = traceback.format_exc()
            raise HTTPException(
                status_code=500, detail={"error": str(e), "stack_trace": stack_trace}
            )
        else:
            raise HTTPException(status_code=500, detail=str(e))


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
    request: FileDeleteRequest, current_user=Depends(get_current_user)
):
    """
    Delete ChromaDB embeddings and associated resources for one or multiple files based on username.

    If the username is the only one in file_info.json, delete the embeddings.
    If other usernames exist, update file_info.json to remove the username but keep the embeddings.

    Args:
        request (DeleteRequest): Request body containing:
            - file_ids (Union[str, List[str]]): Single file ID or list of file IDs
            - include_gcs (bool): Whether to include GCS cleanup (default: False)
            - username (str): Username to check against file_info.json
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
                # Get file_info.json to check usernames
                file_info = gcs_handler.get_file_info(file_id)

                if not file_info:
                    results[file_id] = "Error: File info not found"
                    continue

                # This section previously contained Streamlit-specific code
                # which has been removed as it doesn't belong in the FastAPI backend

                # Check if username exists in file_info
                usernames = file_info.get("username", [])
                if not isinstance(usernames, list):
                    usernames = [usernames]

                if request.username not in usernames:
                    results[
                        file_id
                    ] = f"Error: Username {request.username} not found in file info"
                    continue

                # If username is the only one, delete the embeddings
                if len(usernames) == 1 and usernames[0] == request.username:
                    cleanup_coordinator = CleanupCoordinator(
                        configs, SessionLocal, gcs_handler
                    )
                    cleanup_coordinator.cleanup_chroma_instance(
                        file_id, include_gcs=request.include_gcs
                    )
                    results[file_id] = "Success: Embeddings deleted"
                else:
                    # Remove username from the list and update file_info.json
                    usernames.remove(request.username)
                    gcs_handler.update_username_list(file_id, usernames)
                    results[file_id] = "Success: Username removed from file info"

            except Exception as e:
                results[file_id] = f"Error: {str(e)}"
                logging.error({"file_id": file_id, "error": str(e)})

        # If only one file_id was provided, maintain original response format
        if isinstance(request.file_ids, str):
            if "Success" in results[request.file_ids]:
                return {"message": results[request.file_ids].replace("Success: ", "")}
            else:
                raise HTTPException(status_code=500, detail=results[request.file_ids])

        # For multiple file_ids, return status of all operations
        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete_all_embeddings")
async def delete_all_resources(
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


@app.get("/long-task")
async def long_task():
    """
    This is an endpoint to simulate a long running task with time.sleep.
    The app is configured to handle long running tasks with a timeout of 60 seconds.
    Test is with a curl command:
    - `curl -X GET "http://localhost:8080/long-task" --max-time 30` (wait 30s) which should return a timeout error.
    - `curl -X GET "http://localhost:8080/long-task" --max-time 80` (wait 80s) which should return a success message.
    """
    import time

    # simulate a long-running task (e.g., 50 seconds)
    # the client should not receive a time out if the timeout_keep_alive is set to 60 seconds
    time.sleep(50)
    return {"message": "Task completed"}


# @app.delete("/delete-google-embeddings")
# async def delete_google_embeddings(current_user=Depends(get_current_user)):
#     """
#     Delete all Google embeddings from the GCS bucket.
#     This endpoint removes only the 'google' folders within each file ID directory in file-embeddings.

#     Returns:
#         dict: A message indicating the number of file IDs processed
#     """
#     try:
#         gcs_handler = GCSHandler(Config())
#         result = gcs_handler.delete_google_embeddings()
#         return result
#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Failed to delete Google embeddings: {str(e)}"
#         )


@app.get("/find-file-by-name")
def find_file_by_name(
    filename: str = QueryParam(..., description="Original filename to search for"),
    current_user=Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Find a file_id by searching through all file_info.json files for a matching original_filename.

    Args:
        filename (str): The original filename to search for
        current_user: Authenticated user information

    Returns:
        dict: Dictionary containing:
            - file_id (Optional[str]): The file ID if found, null otherwise
            - found (bool): Whether the file was found
    """
    try:
        gcs_handler = GCSHandler(configs)
        file_id = gcs_handler.find_file_by_original_name(filename)

        return {"file_id": file_id, "found": file_id is not None}
    except Exception as e:
        logging.error(f"Error in find_file_by_name: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error searching for file: {str(e)}"
        )


@app.post("/image/generate")
async def generate_image(
    request: ImageGenerationRequest, current_user=Depends(get_current_user)
):
    """
    Generate an image based on the provided text prompt using selected model (DALL-E 3 or Imagen).

    Args:
        request (ImageGenerationRequest): Request body containing:
            - prompt (str): Text prompt for image generation
            - size (str, optional): Size of the generated image (default: "1024x1024")
            - n (int, optional): Number of images to generate (default: 1)
            - model_choice (str, optional): Model to use ("dall-e-3" or "imagen-3.0") (default: "dall-e-3")
        current_user: Authenticated user information

    Returns:
        dict: Dictionary containing:
            - success (bool): Whether the image generation was successful
            - image_url (str): URL of the generated image (if successful)
            - error (str): Error message (if unsuccessful)
            - prompt (str): The original prompt
            - model (str): The model used for generation
    """
    try:
        # Select the appropriate image generator based on model_choice
        if request.model_choice and "imagen" in request.model_choice.lower():
            # Generate image using the imagen_handler
            logging.info(
                f"Using Vertex AI Imagen model for image generation with prompt: {request.prompt}"
            )
            result = imagen_handler.generate_image(
                prompt=request.prompt, size=request.size, n=request.n
            )
        else:
            # Default to DALL-E 3
            logging.info(
                f"Using DALL-E 3 model for image generation with prompt: {request.prompt}"
            )
            result = dalle_handler.generate_image(
                prompt=request.prompt, size=request.size, n=request.n
            )

        return result
    except Exception as e:
        logging.error(f"Error generating image: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "prompt": request.prompt,
            "model": request.model_choice or "dall-e-3",
        }


@app.post("/image/generate-combined")
async def generate_combined_images(
    request: ImageGenerationRequest, current_user=Depends(get_current_user)
):
    """
    Generate images using both DALL-E and Imagen models concurrently with the same prompt.

    Args:
        request (ImageGenerationRequest): Request body containing:
            - prompt (str): Text prompt for image generation
            - size (str, optional): Size of the generated image (default: "1024x1024")
            - n (int, optional): Number of images to generate per model (default: 1)
        current_user: Authenticated user information

    Returns:
        dict: Dictionary containing results from both models:
            - success (bool): Whether either image generation was successful
            - dalle_result (dict): Result from DALL-E model
            - imagen_result (dict): Result from Imagen model
            - prompt (str): The original prompt
            - models (list): List of models used for generation
    """
    try:
        logging.info(f"Generating images with both models for prompt: {request.prompt}")

        # Use the combined image handler to generate images from both models
        result = await combined_image_handler.generate_images(
            prompt=request.prompt, size=request.size, n=request.n
        )

        return result
    except Exception as e:
        logging.error(f"Error generating combined images: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "prompt": request.prompt,
            "models": ["dall-e-3", configs.vertexai_imagen.model_name],
        }


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
        timeout_keep_alive=60,  # maximum time to keep connection alive
    )
