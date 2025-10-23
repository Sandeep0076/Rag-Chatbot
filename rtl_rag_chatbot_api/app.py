"""
Main FastAPI application for the RAG PDF API.
"""


import asyncio
import json
import logging
import os
import pathlib
import pwd
import time
import uuid
from asyncio import Semaphore
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import BackgroundTasks, Body, Depends, FastAPI, File, Form, HTTPException
from fastapi import Query as QueryParam
from fastapi import Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from sqlalchemy import create_engine, text
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
    GeminiSafetyFilterError,
    get_gemini_non_rag_response,
)
from rtl_rag_chatbot_api.chatbot.image_reader import analyze_images
from rtl_rag_chatbot_api.chatbot.imagen_handler import ImagenGenerator
from rtl_rag_chatbot_api.chatbot.model_handler import ModelHandler
from rtl_rag_chatbot_api.chatbot.utils.encryption import encrypt_file
from rtl_rag_chatbot_api.common.chroma_manager import ChromaDBManager
from rtl_rag_chatbot_api.common.cleanup_coordinator import CleanupCoordinator
from rtl_rag_chatbot_api.common.errors import (
    EmbeddingCreationError,
    FileUploadError,
    SafetyFilterError,
    TabularInvalidDataError,
    UrlExtractionError,
    register_exception_handlers,
)
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
    TitleGenerationRequest,
)
from rtl_rag_chatbot_api.common.prepare_sqlitedb_from_csv_xlsx import (
    PrepareSQLFromTabularData,
)
from rtl_rag_chatbot_api.common.prompts_storage import (
    CHART_DETECTION_PROMPT,
    TITLE_GENERATION_PROMPT,
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
# Set file hash database flag based on config
file_handler.use_file_hash_db = configs.use_file_hash_db
logging.info(
    f"File hash database lookup {'enabled' if configs.use_file_hash_db else 'disabled'}"
)

model_handler = ModelHandler(configs, gcs_handler)
embedding_handler = EmbeddingHandler(configs, gcs_handler, file_handler)
# Initialize image handlers only once
dalle_handler = DalleImageGenerator(configs)
imagen_handler = ImagenGenerator(configs)
# Pass existing handlers to avoid duplicate initialization
combined_image_handler = CombinedImageGenerator(configs, dalle_handler, imagen_handler)

# database connection - only initialize if USE_FILE_HASH_DB is enabled
if configs.use_file_hash_db:
    if os.getenv("DB_INSTANCE"):
        logging.info("Using DB_INSTANCE env variable to connect to database")
        DATABASE_URL = (
            f"postgresql://{os.getenv('DB_USERNAME')}:"
            f"{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST', '127.0.0.1')}:"
            f"{os.getenv('DB_PORT', '5432')}/chatbot_ui"
        )
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(bind=engine)
    else:
        logging.warning(
            "USE_FILE_HASH_DB is enabled but no DB_INSTANCE env variable present. Not able to connect to database."
        )
        SessionLocal = None
else:
    logging.info("Database connection disabled - USE_FILE_HASH_DB is set to false")
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
    # Only initialize cleanup coordinator if database is enabled and SessionLocal is available
    if configs.use_file_hash_db and SessionLocal:
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
    else:
        if configs.use_file_hash_db and not SessionLocal:
            logging.info(
                "Skipping cleanup coordinator initialization - database is enabled but not connected"
            )
        else:
            logging.info(
                "Skipping cleanup coordinator initialization - database is disabled"
            )
        try:
            yield
        finally:
            pass

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

# Register centralized exception handlers
register_exception_handlers(app)

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
        "current_time": time.time(),
    }


@app.get("/__debug/runtime")
def runtime():
    """Get runtime information."""
    return {
        "uid": os.geteuid(),
        "user": pwd.getpwuid(os.geteuid()).pw_name,
        "gid": os.getegid(),
        "cwd": os.getcwd(),
        "chroma_exists": pathlib.Path("chroma_db").exists(),
    }


@contextmanager
def get_db_session():
    """
    Dependency to get a SQLAlchemy session.

    Returns:
        Session: SQLAlchemy session.
    """
    DATABASE_URL = (
        f"postgresql://{os.getenv('DB_USERNAME')}:"
        f"{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST', '127.0.0.1')}:"
        f"{os.getenv('DB_PORT', '5432')}/chatbot_ui"
    )
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)

    attempts = 0
    max_attempts = 10

    while attempts < max_attempts:
        # Try to establish a connection to the database
        try:
            attempts += 1
            db_session = Session()
            # Try to execute a simple query to check the connection
            db_session.execute(text("SELECT 1"))
            break
        except Exception as e:
            logging.warning(
                f"Database connection failed. Retrying in 5 seconds... Original error: {e}"
            )
            time.sleep(5)
        finally:
            db_session.close()

    try:
        db_session = Session()
        yield db_session
    except Exception as e:
        logging.error(f"Error in session: {e}")
    finally:
        db_session.close()


def get_conditional_db_session():
    """
    Conditional dependency to get a SQLAlchemy session only when USE_FILE_HASH_DB is enabled.

    Returns:
        Session: SQLAlchemy session if database is enabled, None otherwise.
    """
    if not configs.use_file_hash_db:
        logging.info("Database connection skipped - USE_FILE_HASH_DB is disabled")
        return None

    if SessionLocal is None:
        logging.info("Database connection skipped - SessionLocal is not available")
        return None

    return get_db_session()


# Asynchronous file processing function that runs concurrently with a semaphore limit
async def process_file_with_semaphore(file_handler, file, file_id, is_image, username):
    async with file_processing_semaphore:
        logging.info(
            f"Starting parallel processing for file: {file.filename} with ID: {file_id}"
        )

        # Determine is_image based on file extension, overriding the form parameter
        file_extension = os.path.splitext(file.filename)[1].lower()
        form_is_image = is_image  # Store original form parameter
        is_image = file_extension in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]
        logging.info(
            f"Form is_image parameter: {form_is_image}, Detected is_image: {is_image}"
        )

        start_time = time.time()
        # below function is for new file only encrypts and uploads to GCS
        result = await file_handler.process_file(file, file_id, is_image, username)
        elapsed = time.time() - start_time
        logging.info(
            f"Completed processing for file: {file.filename} in {elapsed:.2f}s"
        )
        return result


# chat with URL is deprecated, but keeping it for backward compatibility
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
        raise UrlExtractionError(
            url_result.get("message", "Error processing URLs"), details={"urls": urls}
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
            logging.info(f"  File {i + 1}: {f.filename}")
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
        logging.info(f"Creating task for file {i + 1}/{len(all_files)}: {f.filename}")
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
        # Use the file_id returned from process_file, which could be different
        # from the original file_id if existing embeddings were found
        if result.get("status") == "error":
            # For error cases, use the original file_id to maintain tracking
            processed_file_ids.append(file_ids[i])
            logging.warning(f"Skipping invalid file: {all_files[i].filename}")
        else:
            # For successful cases, use the returned file_id which might be from existing embeddings
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
                f"{configs.gcp_resource.gcp_embeddings_folder}/{file_id}/file_info.json",
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


def _split_processed_files_by_type(
    all_files,
    results,
    processed_file_ids,
    original_filenames,
    is_tabular_flags,
):
    """Split processed upload results into document and tabular buckets using precomputed flags."""
    document_files = []
    tabular_files = []

    for i, result in enumerate(results):
        file_id = processed_file_ids[i]
        temp_file_path = result.get("temp_file_path")

        if not file_id or not temp_file_path:
            logging.warning(
                f"Skipping invalid file: {original_filenames[i] if i < len(original_filenames) else 'unknown'}"
            )
            continue

        file_bucket = (
            tabular_files
            if (i < len(is_tabular_flags) and is_tabular_flags[i])
            else document_files
        )
        file_bucket.append(
            {
                "index": i,
                "file_id": file_id,
                "temp_file_path": temp_file_path,
                "filename": all_files[i].filename if i < len(all_files) else "",
                "result": result,
            }
        )

    return document_files, tabular_files


def _handle_tabular_files_results(tabular_files, is_tabular_flags):
    """Update flags and log outcomes for tabular files based on FileHandler results."""
    for file_info in tabular_files:
        i = file_info["index"]
        is_tabular_flags[i] = True

        result = file_info["result"]
        filename = file_info["filename"]
        file_id = file_info["file_id"]

        if result.get("status") == "success" and result.get("is_tabular"):
            logging.info(
                f"Tabular file {filename} with ID {file_id} already processed with database summary"
            )
        elif result.get("status") == "existing":
            logging.info(
                f"Existing tabular file {filename} with ID {file_id} already processed"
            )
        else:
            logging.warning(
                f"Unexpected status for tabular file {filename}: {result.get('status')}"
            )


def _augment_document_files_with_migration(document_files, username):
    """Augment document files with migration files to enable true parallel processing."""
    try:
        migration_ctx = globals().get("current_migration_context")
        if migration_ctx and migration_ctx.get("is_migration"):
            available_migration_files = migration_ctx.get(
                "files_to_migrate_details", []
            )
            for mig in available_migration_files:
                mig_file_id = mig.get("file_id")
                original_filename = mig.get("original_filename")
                if (
                    not mig_file_id
                    or not original_filename
                    or original_filename == "N/A"
                ):
                    continue
                import os as _os

                base_name = _os.path.basename(original_filename)
                preferred_temp_path = f"temp_files/{mig_file_id}_{base_name}"
                legacy_temp_path = f"temp_files/{mig_file_id}"
                temp_path = (
                    preferred_temp_path
                    if _os.path.exists(preferred_temp_path)
                    else legacy_temp_path
                )
                if not _os.path.exists(temp_path):
                    continue
                # Build metadata with merged usernames (preserved + current uploader). Duplicates allowed.
                preserved_usernames = mig.get("usernames", [])
                merged_usernames = (
                    preserved_usernames.copy()
                    if isinstance(preserved_usernames, list)
                    else []
                )
                merged_usernames.append(username)
                # Prefer file_hash from migration details, else from file_info, else compute
                file_hash = mig.get("file_hash")
                if not file_hash:
                    try:
                        gcs_info = GCSHandler(configs).get_file_info(mig_file_id) or {}
                        file_hash = gcs_info.get("file_hash")
                    except Exception:
                        file_hash = None
                if not file_hash:
                    try:
                        import hashlib as _hashlib

                        with open(temp_path, "rb") as _f:
                            file_hash = _hashlib.md5(_f.read()).hexdigest()
                    except Exception:
                        file_hash = None
                # Derive flags from extension or file_info
                ext = _os.path.splitext(base_name)[1].lower()
                is_image_flag = ext in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".gif",
                    ".bmp",
                    ".webp",
                ]
                is_tabular_flag = ext in [
                    ".csv",
                    ".xlsx",
                    ".xls",
                    ".db",
                    ".sqlite",
                    ".sqlite3",
                ]
                if not (is_image_flag or is_tabular_flag):
                    # check existing file_info type
                    try:
                        gcs_info = GCSHandler(configs).get_file_info(mig_file_id) or {}
                        ftype = gcs_info.get("file_type")
                        if ftype in ["tabular", "database"]:
                            is_tabular_flag = True
                        # images are covered by extension set above
                    except Exception:
                        pass
                file_metadata = {
                    "file_id": mig_file_id,
                    "original_filename": base_name,
                    "username": merged_usernames,
                    "is_image": is_image_flag,
                    "is_tabular": is_tabular_flag,
                    "file_hash": file_hash,
                    "migrated": True,  # This file is being migrated
                }
                document_files.append(
                    {
                        "index": -1,
                        "file_id": mig_file_id,
                        "temp_file_path": temp_path,
                        "filename": base_name,
                        "result": {"status": "success", "metadata": file_metadata},
                    }
                )
    except Exception as e:
        logging.warning(
            f"Failed to augment document files for migration parallelism: {e}"
        )
    return document_files


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

    # Use precomputed flags instead of recalculating from filenames
    document_files, tabular_files = _split_processed_files_by_type(
        all_files, results, processed_file_ids, original_filenames, is_tabular_flags
    )

    # Handle tabular files (flags, logging)
    _handle_tabular_files_results(tabular_files, is_tabular_flags)

    # Augment document files with migration context if present
    document_files = _augment_document_files_with_migration(document_files, username)

    # Process document files in parallel if there are multiple
    if len(document_files) > 1:
        await process_document_files_parallel(
            document_files, username, is_image, background_tasks
        )
    # Process single document file normally
    elif len(document_files) == 1:
        file_info = document_files[0]

        # Determine is_image based on file extension for this specific file
        file_extension = os.path.splitext(file_info["filename"])[1].lower()
        is_image = file_extension in [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
        ]

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

        # Import here to avoid circular import
        from rtl_rag_chatbot_api.chatbot.parallel_embedding_creator import (
            create_embeddings_parallel,
        )

        # For single file uploads, directly use create_embeddings_parallel to wait for embedding creation
        # This matches the behavior of multi-file uploads
        logging.info(f"Creating embeddings for single file: {filename} (ID: {file_id})")
        embedding_results = await create_embeddings_parallel(
            file_ids=[file_id],
            file_paths=[temp_file_path],
            embedding_handler=embedding_handler,
            configs=configs,
            session_local=SessionLocal,
            background_tasks=background_tasks,
            username_lists=[[username]],
            file_metadata_list=[file_metadata],
            max_concurrent_tasks=1,
        )

        # Check for errors in embedding creation results
        error_results = [
            result for result in embedding_results if result.get("status") == "error"
        ]
        if error_results:
            # Get the error message
            error_result = error_results[0]  # Single file, so only one result
            error_msg = error_result.get("error", "Unknown error")

            # Log the error
            logging.error(f"Embedding creation failed for file {file_id}: {error_msg}")

            # Raise structured error
            raise EmbeddingCreationError(
                f"Failed to create embeddings for file {filename}: {error_msg}",
                details={"file_id": file_id, "filename": filename},
            )

        logging.info(f"Completed embedding creation for single file: {file_id}")

    # Handle existing files - check if they need new embeddings
    if result["status"] == "existing":
        await handle_existing_file(
            background_tasks, file_id, temp_file_path, username, result
        )


async def process_document_files_parallel(
    document_files, username, is_image, background_tasks: BackgroundTasks = None
):
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

        # Check file extension and determine if this specific file is an image
        file_extension = os.path.splitext(filename)[1].lower()
        file_is_image = file_extension in [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
        ]

        if not is_document_file(file_extension, file_is_image):
            logging.info(f"Skipping non-document file: {filename}")
            return None

        # Initialize handlers within the task to avoid sharing between tasks
        gcs_handler = GCSHandler(configs)

        # Check existing files - now done in parallel
        if result.get("status") == "existing":
            existing_file_info = gcs_handler.get_file_info(file_id)
            if existing_file_info:
                # Check if embeddings exist
                embeddings_result = await embedding_handler.check_embeddings_exist(
                    file_id, "default"
                )
                embeddings_exist = embeddings_result["embeddings_exist"]
                if embeddings_exist:
                    # Update username using the correct function that appends every time
                    # This tracks how many times each user has uploaded the file
                    logging.info(
                        f"Found existing embeddings for file {file_id}, "
                        f"adding username '{username}' to track upload"
                    )

                    # Update username in background without waiting
                    # Use update_file_info which appends usernames (tracks frequency)
                    # instead of update_username_list which deduplicates
                    asyncio.create_task(
                        update_username_in_background_append(file_id, username)
                    )
                    logging.info(
                        f"Scheduled username append for existing file {file_id}: {username}"
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

    # Helper function to append username for existing files (tracks upload frequency)
    async def update_username_in_background_append(file_id, username):
        try:
            gcs_handler = GCSHandler(configs)

            # Use update_file_info which appends usernames every time
            # This tracks how many times each user has uploaded the file
            gcs_handler.update_file_info(file_id, {"username": username})
            logging.info(
                f"Appended username '{username}' for existing file {file_id} "
                f"(tracks upload frequency)"
            )
        except Exception as e:
            logging.error(f"Error appending username for file {file_id}: {str(e)}")
            # Re-raise to ensure the task failure is visible
            raise

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

        # Run parallel embedding creation with BackgroundTasks for non-blocking uploads
        embedding_results = await create_embeddings_parallel(
            file_ids,
            file_paths,
            embedding_handler,
            configs,
            SessionLocal,
            background_tasks,
            username_lists,
            file_metadata_list,
            max_concurrent,
        )

        # Check for errors in embedding creation results
        error_results = [
            result for result in embedding_results if result.get("status") == "error"
        ]
        if error_results:
            # Collect error messages
            error_messages = []
            for error_result in error_results:
                file_id = error_result.get("file_id", "unknown")
                error_msg = error_result.get("error", "Unknown error")
                error_messages.append(f"File {file_id}: {error_msg}")

            # Log the errors
            logging.error(
                f"Embedding creation failed for {len(error_results)} files: {error_messages}"
            )

            # Raise structured error with detailed information
            raise EmbeddingCreationError(
                f"Failed to create embeddings for {len(error_results)} file(s)",
                details={
                    "errors": error_messages,
                    "total_files": len(files_to_process),
                    "failed_files": len(error_results),
                    "successful_files": len(files_to_process) - len(error_results),
                },
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

    try:
        logging.info(
            f"File {file_id} already exists, checking if it needs new embeddings"
        )

        # Check if embeddings exist
        azure_result = await embedding_handler.check_embeddings_exist(
            file_id, "gpt_4o_mini"
        )

        # Get existing username list to ensure we preserve all users
        gcs_handler = GCSHandler(configs)
        current_file_info = gcs_handler.get_file_info(file_id)
        current_username_list = current_file_info.get("username", [])

        # If embeddings already exist, just update the username list directly
        if azure_result["embeddings_exist"]:
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
    finally:
        # Clean up empty directory if it exists
        import os

        os.rmdir(temp_file_path) if os.path.isdir(temp_file_path) and not os.listdir(
            temp_file_path
        ) else None


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
    # TODO: This is not correct. We should not be appending usernames. We should be replacing the username list.
    for current_user in current_username_list:
        if current_user not in username_list:
            username_list.append(current_user)

    process_document_file(background_tasks, file_id, temp_file_path, username_list)


async def _process_normal_upload_flow(
    all_files,
    parsed_existing_file_ids,
    is_image,
    username,
    background_tasks,
    is_migration: bool = False,
    migration_context: dict | None = None,
):
    existing_results = []
    if parsed_existing_file_ids:
        if is_migration and migration_context is not None:
            existing_results = await process_migration_file_ids_in_parallel(
                parsed_existing_file_ids,
                username,
                background_tasks,
                migration_context,
            )
        else:
            existing_results = await process_existing_file_ids_in_parallel(
                parsed_existing_file_ids, username, background_tasks
            )

    new_file_results = []
    if len(all_files) > 0:
        (
            results,
            processed_file_ids,
            original_filenames,
            is_tabular_flags,
            statuses,
        ) = await process_files_in_parallel(file_handler, all_files, is_image, username)

        # Check for errors in file processing
        error_results = [r for r in results if r.get("status") == "error"]
        if error_results:
            # If there are any errors, return the first error message
            error_result = error_results[0]
            logging.error(f"File processing failed: {error_result.get('message')}")
            return JSONResponse(
                status_code=400,
                content={
                    "message": (
                        f"File processing failed: {error_result.get('message')}"
                    ),
                    "file_id": error_result.get("file_id"),
                    "status": "error",
                },
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

        new_file_results = list(
            zip(processed_file_ids, original_filenames, is_tabular_flags)
        )

    # Fallback: If this is a migration scenario and there are no uploaded files left after filtering,
    # ensure migration files are embedded now using the migration context.
    try:
        if is_migration and migration_context and len(all_files) == 0:
            # Reuse augmentation helper to build document files from migration context
            document_files = _augment_document_files_with_migration([], username)
            if document_files:
                if len(document_files) > 1:
                    await process_document_files_parallel(
                        document_files, username, is_image, background_tasks
                    )
                else:
                    file_info = document_files[0]
                    # Determine is_image for this specific file
                    file_extension = os.path.splitext(file_info["filename"])[1].lower()
                    file_is_image = file_extension in [
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".gif",
                        ".bmp",
                        ".webp",
                    ]
                    await process_document_type_file(
                        background_tasks,
                        file_info["file_id"],
                        file_info["temp_file_path"],
                        file_info["filename"],
                        file_info["result"],
                        username,
                        file_is_image,
                    )
    except Exception as mig_err:
        logging.error(
            f"Migration fallback embedding processing failed: {str(mig_err)}",
            exc_info=True,
        )

    return combine_upload_results(existing_results, new_file_results)


@app.post("/file/upload", response_model=FileUploadResponse)
async def upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    files: List[UploadFile] = File(None),
    is_image: bool = Form(False),
    username: str = Form(...),
    urls: str = Form(None),
    existing_file_ids: str = Form(None),  # New parameter for existing file IDs
    current_user=Depends(get_current_user),
):
    """
    Handles file upload and automatically creates embeddings for PDFs and images.
    Now also supports existing file IDs alongside new document uploads.

    Supports four modes of operation:
    1. URL processing: Extract content from URLs and create embeddings
    2. Multiple file upload: Process multiple files concurrently using asyncio.gather()
    3. Single file upload: Process a single file (for backward compatibility)
    4. Existing file IDs: Process existing file IDs with embeddings validation and download

    All file processing is done asynchronously and in parallel when multiple files are uploaded.
    Embeddings creation and other post-processing happens in background tasks.
    """
    # FileHandler database flag is already set at module level

    try:
        # Parse existing file IDs if provided (do this before URL processing)
        parsed_existing_file_ids = []
        if existing_file_ids:
            try:
                # Split by comma, newline, or both and clean up
                parsed_existing_file_ids = [
                    fid.strip()
                    for fid in existing_file_ids.replace("\n", ",").split(",")
                    if fid.strip()
                ]
                logging.info(
                    f"Processing {len(parsed_existing_file_ids)} existing file IDs"
                )
            except Exception as e:
                logging.error(f"Error parsing existing_file_ids: {str(e)}")
                from rtl_rag_chatbot_api.common.errors import (
                    BaseAppError,
                    ErrorRegistry,
                )

                raise BaseAppError(
                    ErrorRegistry.ERROR_BAD_REQUEST,
                    f"Invalid format for existing_file_ids: {str(e)}",
                    details={"input": existing_file_ids},
                )

        # Combine files from both parameters
        all_files = prepare_file_list(file, files)

        # Validate existing file IDs early
        if parsed_existing_file_ids:
            validation_response = await validate_existing_file_ids(
                parsed_existing_file_ids, configs
            )
            if validation_response:
                return validation_response

        # ===== AUTO-MIGRATION LOGIC - SIMPLE AND TRANSPARENT =====
        # Check existing file IDs for old embeddings and auto-migrate if needed
        from rtl_rag_chatbot_api.chatbot.auto_migration_service import (
            AutoMigrationService,
        )

        auto_migration_service = AutoMigrationService(
            configs, gcs_handler, SessionLocal
        )

        if parsed_existing_file_ids:
            logging.info(
                f"Checking {len(parsed_existing_file_ids)} existing file IDs for migration needs"
            )

            # Check and migrate existing files if they have old embeddings
            migration_results = (
                await auto_migration_service.check_and_migrate_multiple_files(
                    file_ids=parsed_existing_file_ids,
                    embedding_handler=embedding_handler,
                    background_tasks=background_tasks,
                    max_concurrent=3,  # Process up to 3 files concurrently
                )
            )

            # Check for migration failures
            failed_migrations = [
                r
                for r in migration_results
                if isinstance(r, dict)
                and isinstance(r.get("migration_result"), dict)
                and r.get("migration_result", {}).get("status") == "error"
            ]
            if failed_migrations:
                failed_file_ids = [
                    r.get("migration_result", {}).get("file_id") or r.get("file_id")
                    for r in failed_migrations
                ]
                error_messages = [
                    r.get("migration_result", {}).get("message") or r.get("message")
                    for r in failed_migrations
                ]

                logging.error(
                    f"Migration failed for {len(failed_migrations)} file(s): {failed_file_ids}"
                )

                # Fail the request so the frontend does not show success
                raise EmbeddingCreationError(
                    (
                        "Migration failed for existing files. Please re-upload the original files "
                        "to create new embeddings. Files: "
                        + ", ".join(failed_file_ids)
                        + ". Errors: "
                        + "; ".join(error_messages)
                    ),
                    details={
                        "failed_file_ids": failed_file_ids,
                        "error_type": "migration_failed",
                    },
                )

            # Log successful migrations
            migrated_files = [r for r in migration_results if r.get("migrated", False)]
            if migrated_files:
                logging.info(
                    f"Auto-migrated {len(migrated_files)} file(s) during upload: "
                    f"{[r.get('migration_result', {}).get('file_id') for r in migrated_files]}"
                )

        # Proceed with normal upload flow
        response = await _process_normal_upload_flow(
            all_files,
            parsed_existing_file_ids,
            is_image,
            username,
            background_tasks,
        )

        return response

    except HTTPException as http_ex:
        # Re-raise HTTPException with proper status codes (these are expected API errors)
        logging.error(f"API error in upload_file: {http_ex.detail}")
        raise http_ex
    except Exception as e:
        logging.exception(f"Unexpected error in upload_file: {str(e)}")
        raise FileUploadError(
            f"File upload failed: {str(e)}", details={"error_type": "upload_failure"}
        )


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
            raise TabularInvalidDataError(
                "Failed to prepare database from input file",
                details={"file_id": file_id, "file_path": temp_file_path},
            )

        # Encrypt the database before upload
        encrypted_db_path = encrypt_file(db_path)
        try:
            # Upload the encrypted SQLite database to GCS
            gcs_handler.upload_to_gcs(
                configs.gcp_resource.bucket_name,
                source=encrypted_db_path,
                destination_blob_name=(
                    f"{configs.gcp_resource.gcp_embeddings_folder}/"
                    f"{file_id}/tabular_data.db.encrypted"
                ),
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
                    f"{configs.gcp_resource.gcp_embeddings_folder}/{file_id}/file_info.json",
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
                        f"{configs.gcp_resource.gcp_embeddings_folder}/{file_id}/file_info.json",
                    )
                },
            )
        except Exception as upload_error:
            logging.error(f"Error updating file_info.json: {str(upload_error)}")
        raise


async def process_existing_file_ids_in_parallel(
    file_ids: List[str], username: str, background_tasks: BackgroundTasks
) -> List[tuple]:
    """
    Process existing file IDs in parallel - validate embeddings and download if needed.
    Now handles errors gracefully by collecting all invalid file IDs and providing detailed error information.

    Args:
        file_ids: List of existing file IDs to process
        username: Username to add to the file's username list
        background_tasks: Background tasks for async operations

    Returns:
        List of tuples: (file_id, original_filename, is_tabular)

    Raises:
        HTTPException: With detailed information about invalid file IDs if any are found
    """

    async def process_single_existing_file_id(file_id: str) -> tuple:
        """Process a single existing file ID"""
        try:
            # Check if embeddings exist using existing endpoint logic
            results = await _check_file_embeddings_safe(
                [file_id], "gpt_4o_mini"
            )  # Default model for checking
            result = results[0]  # Get the first (and only) result

            if not result.get("embeddings_exist", False):
                logging.warning(f"Embeddings not found for file ID: {file_id}")
                return (
                    "error",
                    file_id,
                    f"Embeddings not found for file {file_id}. Please create embeddings first.",
                )

            # Get file info to determine filename and type
            file_info = gcs_handler.get_file_info(file_id)
            if not file_info:
                logging.warning(f"File info not found for file ID: {file_id}")
                return ("error", file_id, f"File info not found for file {file_id}")

            original_filename = file_info.get("original_filename", f"file_{file_id}")
            is_tabular = file_info.get("file_type") in ["tabular", "database"]

            # Check if embeddings exist locally, download if not
            local_embeddings_exist = embedding_handler.has_local_embeddings(file_id)
            if not local_embeddings_exist:
                logging.info(f"Downloading embeddings for file {file_id}")
                gcs_handler.download_files_from_folder_by_id(file_id)

            # Update username in background (similar to existing file handling)
            background_tasks.add_task(
                update_username_for_existing_file,
                file_id,
                username,
                file_info.get("username", []),
            )

            logging.info(f"Successfully processed existing file ID: {file_id}")
            return (file_id, original_filename, is_tabular)

        except Exception as e:
            logging.error(f"Error processing existing file ID {file_id}: {str(e)}")
            return ("error", file_id, f"Error processing file ID {file_id}: {str(e)}")

    # Process all file IDs in parallel
    try:
        results = await asyncio.gather(
            *[process_single_existing_file_id(file_id) for file_id in file_ids],
            return_exceptions=True,
        )

        # Separate successful results from errors
        processed_results = []
        error_details = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_details.append({"file_id": file_ids[i], "error": str(result)})
            elif isinstance(result, tuple) and len(result) == 3:
                if result[0] == "error":
                    error_details.append({"file_id": result[1], "error": result[2]})
                else:
                    processed_results.append(result)
            else:
                processed_results.append(result)

        # If there are errors, raise an HTTPException with detailed information
        if error_details:
            valid_file_count = len(processed_results)
            invalid_file_count = len(error_details)

            error_message = f"Found {invalid_file_count} invalid file ID(s) out of {len(file_ids)} total."
            if valid_file_count > 0:
                error_message += (
                    f" {valid_file_count} file(s) were processed successfully."
                )

            # Create detailed error information
            detailed_errors = {
                "message": error_message,
                "total_files": len(file_ids),
                "valid_files": valid_file_count,
                "invalid_files": invalid_file_count,
                "errors": error_details,
                "successfully_processed": [result[0] for result in processed_results]
                if processed_results
                else [],
            }

            logging.error(
                f"Error in parallel processing of existing file IDs: {error_message}"
            )
            raise EmbeddingCreationError(error_message, details=detailed_errors)

        return processed_results

    except HTTPException:
        raise
    except Exception as e:
        logging.error(
            f"Unexpected error in parallel processing of existing file IDs: {str(e)}"
        )
        raise EmbeddingCreationError(
            f"Unexpected error processing file IDs: {str(e)}",
            details={"error_type": "parallel_processing_failure"},
        )


async def _handle_migration_file_id(
    file_id: str,
    username: str,
    migration_details_map: dict,
) -> tuple:
    """Handle a single file_id that requires migration by preparing it for the pipeline.

    Returns a tuple of (file_id, original_filename, is_tabular_flag) on success,
    or ("error", file_id, error_message) on failure.
    """
    try:
        logging.info(f"Processing file {file_id} for migration (new embeddings)")

        # Get migration details
        file_detail = migration_details_map.get(file_id, {})
        preserved_usernames = file_detail.get("usernames", [])
        original_filename = file_detail.get("original_filename", f"file_{file_id}")

        # Determine temp file path (downloaded/persisted during migration)
        preferred_temp_path = None
        if original_filename and original_filename != f"file_{file_id}":
            import os as _os

            base_name = _os.path.basename(original_filename)
            preferred_temp_path = f"temp_files/{file_id}_{base_name}"
        legacy_temp_path = f"temp_files/{file_id}"
        temp_file_path = (
            preferred_temp_path
            if preferred_temp_path and os.path.exists(preferred_temp_path)
            else legacy_temp_path
        )

        if os.path.exists(temp_file_path):
            logging.info(
                f"Using downloaded/uploaded file for migration: {temp_file_path}"
            )
            # Use helper function to prepare migration file for standard pipeline
            return await _prepare_migration_file_for_pipeline(
                file_id=file_id,
                original_filename=original_filename,
                temp_file_path=temp_file_path,
                preserved_usernames=preserved_usernames,
                username=username,
                gcs_handler=gcs_handler,
                configs=configs,
                embedding_handler=embedding_handler,
            )

        error_msg = f"Migration file {file_id} not found in temp directory"
        logging.error(error_msg)
        return ("error", file_id, error_msg)

    except Exception as e:
        logging.error(f"Error processing migration file ID {file_id}: {str(e)}")
        return ("error", file_id, f"Error processing file ID {file_id}: {str(e)}")


async def _handle_existing_no_migration_file_id(
    file_id: str,
    username: str,
    background_tasks: BackgroundTasks,
) -> tuple:
    """Handle a single file_id that does not require migration using existing logic.

    Returns a tuple of (file_id, original_filename, is_tabular_flag) on success,
    or ("error", file_id, error_message) on failure.
    """
    try:
        logging.info(f"Processing file {file_id} (no migration needed)")

        # Check embeddings
        results = await _check_file_embeddings_safe([file_id], "gpt_4o_mini")
        result = results[0]

        if not result.get("embeddings_exist", False):
            logging.warning(f"Embeddings not found for file ID: {file_id}")
            return ("error", file_id, f"Embeddings not found for file {file_id}")

        # Get file info
        file_info = gcs_handler.get_file_info(file_id)
        if not file_info:
            return ("error", file_id, f"File info not found for file {file_id}")

        original_filename = file_info.get("original_filename", f"file_{file_id}")
        is_tabular = file_info.get("file_type") in ["tabular", "database"]

        # Ensure local embeddings are present
        local_embeddings_exist = embedding_handler.has_local_embeddings(file_id)
        if not local_embeddings_exist:
            logging.info(f"Downloading embeddings for file {file_id}")
            gcs_handler.download_files_from_folder_by_id(file_id)

        # Update username (normal existing file logic)
        background_tasks.add_task(
            update_username_for_existing_file,
            file_id,
            username,
            file_info.get("username", []),
        )

        return (file_id, original_filename, is_tabular)

    except Exception as e:
        logging.error(
            f"Error processing existing (no migration) file ID {file_id}: {str(e)}"
        )
        return ("error", file_id, f"Error processing file ID {file_id}: {str(e)}")


async def process_migration_file_ids_in_parallel(
    file_ids: List[str],
    username: str,
    background_tasks: BackgroundTasks,
    migration_context: dict,
) -> List[tuple]:
    """
    Process file IDs in migration context - handle both files needing migration and existing files.

    Args:
        file_ids: List of file IDs to process (mix of migration and non-migration files)
        username: Current username to add/preserve
        background_tasks: Background tasks for async operations
        migration_context: Migration context with file details

    Returns:
        List of tuples: (file_id, original_filename, is_tabular)
    """
    files_to_migrate = migration_context.get("files_to_migrate", [])
    files_to_migrate_details = migration_context.get("files_to_migrate_details", [])
    existing_files_no_migration = migration_context.get(
        "existing_files_no_migration", []
    )

    # Create mapping of file_id to details for migration files
    migration_details_map = {
        detail["file_id"]: detail for detail in files_to_migrate_details
    }

    async def _process_single_migration_file_id(file_id: str) -> tuple:
        """Thin wrapper that routes to the appropriate handler based on context lists."""
        try:
            if file_id in files_to_migrate:
                return await _handle_migration_file_id(
                    file_id=file_id,
                    username=username,
                    migration_details_map=migration_details_map,
                )
            if file_id in existing_files_no_migration:
                return await _handle_existing_no_migration_file_id(
                    file_id=file_id,
                    username=username,
                    background_tasks=background_tasks,
                )
            error_msg = f"File {file_id} not found in migration context"
            logging.error(error_msg)
            return ("error", file_id, error_msg)
        except Exception as e:
            logging.error(f"Error processing migration file ID {file_id}: {str(e)}")
            return ("error", file_id, f"Error processing file ID {file_id}: {str(e)}")

    # Process all file IDs in parallel
    try:
        results = await asyncio.gather(
            *[_process_single_migration_file_id(file_id) for file_id in file_ids],
            return_exceptions=True,
        )

        # Handle results similar to existing function
        processed_results = []
        error_details = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_details.append({"file_id": file_ids[i], "error": str(result)})
            elif isinstance(result, tuple) and len(result) == 3:
                if result[0] == "error":
                    error_details.append({"file_id": result[1], "error": result[2]})
                else:
                    processed_results.append(result)
            else:
                processed_results.append(result)

        # If there are errors, raise an HTTPException
        if error_details:
            error_message = (
                f"Migration processing failed for {len(error_details)} file(s)"
            )
            detailed_errors = {
                "message": error_message,
                "errors": error_details,
                "successfully_processed": [result[0] for result in processed_results],
            }
            logging.error(f"Error in migration processing: {error_message}")
            raise EmbeddingCreationError(error_message, details=detailed_errors)

        return processed_results

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unexpected error in migration processing: {str(e)}")
        raise EmbeddingCreationError(
            f"Migration processing error: {str(e)}",
            details={"error_type": "migration_failure"},
        )


async def validate_existing_file_ids(
    parsed_existing_file_ids: List[str], configs: dict
):
    """
    Validate that all existing file IDs are valid and accessible.

    Returns:
        JSONResponse with error details if any file IDs are invalid, None if all valid
    """
    from rtl_rag_chatbot_api.chatbot.gcs_handler import GCSHandler

    gcs_handler = GCSHandler(configs)

    invalid_file_ids = []
    for file_id in parsed_existing_file_ids:
        try:
            file_info = gcs_handler.get_file_info(file_id)
            if not file_info:
                invalid_file_ids.append(file_id)
        except Exception:
            invalid_file_ids.append(file_id)

    if invalid_file_ids:
        invalid_count = len(invalid_file_ids)
        total_count = len(parsed_existing_file_ids)
        error_message = (
            f"Found {invalid_count} invalid file ID(s) out of {total_count} total."
        )
        return JSONResponse(
            status_code=400,
            content={
                "detail": {
                    "message": error_message,
                    "errors": [
                        {"file_id": fid, "error": "File not found"}
                        for fid in invalid_file_ids
                    ],
                }
            },
        )

    return None


def combine_upload_results(
    existing_results: List[tuple], new_file_results: List[tuple]
) -> JSONResponse:
    """
    Combine results from existing file IDs and new file uploads into a unified response.

    Args:
        existing_results: List of tuples from existing file ID processing
        new_file_results: List of tuples from new file processing

    Returns:
        JSONResponse with combined results
    """
    # Combine all results (existing first to preserve expected ordering)
    all_results = existing_results + new_file_results

    if not all_results:
        return JSONResponse(
            status_code=400,
            content={"message": "No valid files or file IDs were processed."},
        )

    # Deduplicate while preserving order and keeping the first occurrence's metadata
    # Each tuple is (file_id, original_filename, is_tabular)
    unique_map: Dict[str, tuple] = {}
    for file_id, filename, is_tabular in all_results:
        if file_id not in unique_map:
            unique_map[file_id] = (filename, is_tabular)

    # Extract data from deduplicated results
    all_file_ids = list(unique_map.keys())
    all_filenames = [unique_map[fid][0] for fid in all_file_ids]
    all_is_tabular_flags = [unique_map[fid][1] for fid in all_file_ids]

    # Use existing format_upload_response function for consistency
    return format_upload_response(all_file_ids, all_filenames, all_is_tabular_flags)


async def update_metadata_in_background(
    configs, file_id, updated_metadata, error_context=""
):
    """Asynchronous function to update file_info.json in the background."""
    try:
        # Initialize GCS handler within the task
        background_gcs = GCSHandler(configs)
        background_gcs.update_file_info(file_id, updated_metadata)
    except Exception as e:
        logging.error(
            f"Error updating metadata in background for {file_id} "
            f"(context: {error_context}): {str(e)}"
        )


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

        # Flatten new_usernames in case it's a nested list (fixes URL processing issue)
        flat_new_usernames = []
        if isinstance(new_usernames, list):
            for item in new_usernames:
                if isinstance(item, list):
                    # If item is a list, extend with its contents
                    flat_new_usernames.extend(item)
                else:
                    # If item is a string, append it directly
                    flat_new_usernames.append(item)
        else:
            # If new_usernames is not a list, treat it as a single username
            flat_new_usernames = [new_usernames]

        # Add any new usernames that aren't in the merged list
        for username in flat_new_usernames:
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
        # Ensure migrated flag is present (preserve existing value)
        if "migrated" not in file_metadata:
            file_metadata["migrated"] = False
        # Merge provided username_list into existing metadata (preserve and deduplicate)
        if username_list:
            current_usernames = file_metadata.get("username", [])
            if not isinstance(current_usernames, list):
                current_usernames = [current_usernames] if current_usernames else []
            for u in username_list:
                if u not in current_usernames:
                    current_usernames.append(u)
            file_metadata["username"] = current_usernames
    else:
        logging.info(f"Creating new metadata for {file_id}")
        file_metadata = {
            "file_id": file_id,
            "username": username_list if username_list else [],
            "embeddings_status": "in_progress",
            "migrated": False,  # New files are not migrated
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
    background_tasks: BackgroundTasks = None,
):
    """
    Background task for creating embeddings.
    With the decoupled approach, local embeddings are created first (blocking),
    marked as 'ready_for_chat', and then cloud operations happen in the background (non-blocking).

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

        # Update metadata status to in_progress and ensure the file is created
        file_metadata["embeddings_status"] = "in_progress"
        await update_metadata_in_background(
            configs, file_id, file_metadata, "before embedding creation"
        )

        # Log details about the metadata
        file_hash = file_metadata.get("file_hash") if file_metadata else None
        logging.info(f"Using file-specific metadata for {file_id}: {file_metadata}")
        logging.info(f"File hash for {file_id}: {file_hash}")

        # CRITICAL SECTION: Create local embeddings (blocking operation)
        # This must be completed before returning to allow chat to begin
        is_image = file_metadata.get("is_image", False) if file_metadata else False
        embedding_result = await embedding_handler.create_embeddings(
            file_id=file_id,
            temp_file_path=temp_file_path,
            is_image=is_image,
            file_metadata=file_metadata,
        )

        # If embeddings were created successfully, trigger background upload
        if embedding_result["status"] == "ready_for_chat":
            # Store file hash in database if the feature is enabled
            if configs.use_file_hash_db and file_metadata:
                file_hash = file_metadata.get("file_hash")
                original_filename = file_metadata.get("original_filename")
                if file_hash:
                    try:
                        # Import database function and store hash directly
                        from rtl_rag_chatbot_api.common.db import (
                            delete_file_info_by_file_id,
                            insert_file_info_record,
                        )

                        with get_db_session() as db_session:
                            # Always delete old entry for this file_id before insert (covers migration)
                            try:
                                del_result = delete_file_info_by_file_id(
                                    db_session, file_id
                                )
                                if del_result.get("status") == "success":
                                    deleted_count = del_result.get("deleted_count", 0)
                                    logging.info(
                                        f"Deleted {deleted_count} old FileInfo records for file_id: {file_id}"
                                    )
                            except Exception as del_err:
                                logging.warning(
                                    f"Failed to delete existing FileInfo for {file_id}: {del_err}"
                                )

                            result = insert_file_info_record(
                                db_session,
                                file_id,
                                file_hash,
                                original_filename,
                                None,  # Use configurable default
                                configs,
                            )
                            if result["status"] == "success":
                                logging.info(
                                    f"Successfully stored file hash in database for file_id: {file_id}"
                                )
                            else:
                                logging.error(
                                    f"Failed to store file hash in database: {result['message']}"
                                )
                    except Exception as hash_store_error:
                        logging.error(
                            f"Failed to store file hash in database: {str(hash_store_error)}"
                        )

            # Start the GCS upload as a non-blocking operation using background_tasks if available
            if background_tasks:
                # Use FastAPI's BackgroundTasks for proper non-blocking execution
                background_tasks.add_task(embedding_handler.upload_embeddings, file_id)
                logging.info(
                    f"Scheduled background upload for file_id: {file_id} using BackgroundTasks"
                )
            else:
                # Fallback to creating a new task for backward compatibility
                asyncio.create_task(embedding_handler.upload_embeddings(file_id))
                logging.info(
                    f"Triggered background upload for file_id: {file_id} using asyncio.create_task"
                )

            # The username is already included in the initial metadata write.
            # A separate update is redundant for new files.
            # if username_list:
            #     asyncio.create_task(
            #         update_usernames_in_background(configs, file_id, username_list)
            #     )
        else:
            # Handle errors in embedding creation
            logging.error(
                f"Error creating embeddings for {file_id}: {embedding_result.get('message', 'Unknown error')}"
            )
            # Run cleanup in background to avoid blocking
            if configs.use_file_hash_db:
                asyncio.create_task(
                    run_cleanup_after_error(configs, SessionLocal, file_id)
                )
            else:
                # When database is disabled, just clean up ChromaDB and GCS
                from rtl_rag_chatbot_api.common.cleanup_coordinator import (
                    CleanupCoordinator,
                )

                cleanup_coordinator = CleanupCoordinator(configs, None, gcs_handler)
                asyncio.create_task(
                    asyncio.to_thread(
                        cleanup_coordinator.cleanup_chroma_instance,
                        file_id,
                        include_gcs=True,
                    )
                )

        return embedding_result

    except Exception as e:
        logging.error(f"Error in create_embeddings_background for {file_id}: {str(e)}")
        # Run cleanup in background to avoid blocking
        if configs.use_file_hash_db:
            asyncio.create_task(run_cleanup_after_error(configs, SessionLocal, file_id))
        else:
            # When database is disabled, just clean up ChromaDB and GCS
            from rtl_rag_chatbot_api.common.cleanup_coordinator import (
                CleanupCoordinator,
            )

            cleanup_coordinator = CleanupCoordinator(configs, None, gcs_handler)
            asyncio.create_task(
                asyncio.to_thread(
                    cleanup_coordinator.cleanup_chroma_instance,
                    file_id,
                    include_gcs=True,
                )
            )
        return {
            "status": "error",
            "message": f"Failed to create embeddings: {str(e)}",
            "file_id": file_id,
        }


@app.post("/embeddings/check", response_model=Dict[str, Any])
async def check_embeddings(
    request: EmbeddingsCheckRequest, current_user=Depends(get_current_user)
):
    """
    Check if embeddings exist for the specified files and model choice.

    Args:
        request (EmbeddingsCheckRequest): Request containing file_ids (list) and model_choice
        current_user: Authenticated user information

    Returns:
        Dict containing:
            - results (List[Dict]): List of results for each file
            - summary (Dict): Summary statistics
    """
    try:
        # Check if embeddings exist for all files
        results = await _check_file_embeddings(
            file_ids=request.file_ids, model_choice=request.model_choice
        )

        # Create summary statistics
        all_exist = all(r["embeddings_exist"] for r in results)
        all_legacy = all(r["model_type"] == "azure" for r in results)
        all_new = all(r["model_type"] != "azure" for r in results)
        has_one_legacy_model = any(r["model_type"] == "azure" for r in results)

        total_files = len(results)

        if len(results) > 1 and not all_legacy and has_one_legacy_model:
            results = list(
                map(
                    lambda r: {**r, "embeddings_exist": False}
                    if r["model_type"] == "azure"
                    else r,
                    results,
                )
            )

        existing_files = sum(1 for r in results if r["embeddings_exist"])

        # AIP-923: this is a hotfix, so that the client can process the return
        # Currently the client accepts status 400 and a list of all embeddings with status
        # if existing_files != total_files:
        # looks like at least one is missing
        # return JSONResponse(status_code=400, content=results)

        return {
            "results": results,
            "summary": {
                "total_files": total_files,
                "files_with_embeddings": existing_files,
                "files_missing_embeddings": total_files - existing_files,
                # AIP-1066, https://rtldata.atlassian.net/browse/AIP-1066
                # if there are multiple files, we require all to be non-legacy.
                # if there is one single file, we allow legacy model_type "azure"
                "all_files_ready": (
                    (all_legacy or all_new or (all_exist and not has_one_legacy_model))
                    if total_files > 1
                    else all_exist
                ),
                "model_choice": request.model_choice,
            },
        }

    except HTTPException:
        # Re-raise HTTPExceptions (including those from _check_file_embeddings)
        raise
    except Exception as e:
        logging.error(f"Error checking embeddings: {str(e)}")
        raise EmbeddingCreationError(
            f"Error checking embeddings: {str(e)}",
            details={"error_type": "embedding_check_failed"},
        )


@app.get("/embeddings/status/{file_id}", response_model=Dict[str, Any])
async def get_embedding_status(file_id: str, current_user=Depends(get_current_user)):
    """
    Get the current status of embeddings for a specific file.
    The frontend can poll this endpoint to know when embeddings are ready for chat.

    Now also checks for old embeddings and triggers auto-migration if needed.

    Args:
        file_id: The ID of the file to check
        current_user: Authenticated user information

    Returns:
        Dict containing:
            - status: Current status of embeddings ("not_started", "in_progress", "ready_for_chat", or "completed")
            - can_chat: Whether chat is available with this file
            - file_id: The checked file ID
            - message: Human-readable message about the current status
            - migrated: Whether the file was migrated (optional)
    """
    try:
        # ===== AUTO-MIGRATION CHECK FOR EMBEDDING STATUS =====
        from rtl_rag_chatbot_api.chatbot.auto_migration_service import (
            AutoMigrationService,
        )

        auto_migration_service = AutoMigrationService(
            configs, gcs_handler, SessionLocal
        )

        # Check if file needs migration and migrate if necessary
        migration_result = await auto_migration_service.check_and_migrate_if_needed(
            file_id=file_id,
            embedding_handler=embedding_handler,
        )

        migrated = migration_result.get("migrated", False)
        if migrated:
            logging.info(
                f"Auto-migrated {file_id} during status check: "
                f"{migration_result.get('migration_result', {})}"
            )

        # Check local file_info.json first
        local_info_path = os.path.join("./chroma_db", file_id, "file_info.json")

        if os.path.exists(local_info_path):
            with open(local_info_path, "r") as f:
                file_info = json.load(f)

            status = file_info.get("embeddings_status", "not_started")
            can_chat = status in ["ready_for_chat", "completed"]

            response = {
                "status": status,
                "can_chat": can_chat,
                "file_id": file_id,
                "message": "Ready for chat" if can_chat else "Embeddings not ready yet",
            }

            if migrated:
                response["migrated"] = True
                response["embedding_type"] = migration_result.get(
                    "embedding_type", "azure-3-large"
                )

            return response

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
        raise EmbeddingCreationError(
            f"Error checking embedding status: {str(e)}",
            details={"error_type": "embedding_status_check_failed"},
        )


def check_db_existence(
    file_id: str,
    query: Query,
    configs: dict,
    initialized_models: dict,
    temperature: float,
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
            # For backwards compatibility in check_db_existence, pass empty all_file_infos
            model = TabularDataHandler(
                configs,
                file_id=query.file_id,
                model_choice=query.model_choice,
                all_file_infos={},  # Empty dict for backwards compatibility
                temperature=temperature,
            )
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
    temperature: float,
):
    """
    Initialize RAG model based on query parameters.

    Args:
        query: Query parameters including model choice
        configs: Application configurations
        gcs_handler: GCS handler instance
        file_info: File information from GCS
        temperature: Temperature value for model randomness

    Returns:
        Initialized model instance
    """
    chroma_path = f"./chroma_db/{query.file_id}"
    is_gemini = query.model_choice.lower() in ["gemini-2.5-flash", "gemini-2.5-pro"]
    embedding_type = "azure"
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
        # For GeminiHandler, create instance with all necessary parameters including file info
        all_file_infos = {query.file_id: file_info}  # Create the expected structure
        model = GeminiHandler(
            configs=configs,
            gcs_handler=gcs_handler,
            model_choice=query.model_choice,
            file_id=query.file_id,
            all_file_infos=all_file_infos,  # Pass file info for context
            user_id=query.user_id,
            temperature=temperature,
        )
        # Note: GeminiHandler with constructor params doesn't need separate initialize() call
    elif query.model_choice in ["claude-sonnet-4@20250514", "claude-sonnet-4-5"]:
        # Anthropic (Vertex) handler
        from rtl_rag_chatbot_api.chatbot.anthropic_handler import AnthropicHandler

        all_file_infos = {query.file_id: file_info}
        model = AnthropicHandler(
            configs=configs,
            gcs_handler=gcs_handler,
            model_choice=query.model_choice,
            file_id=query.file_id,
            all_file_infos=all_file_infos,
            user_id=query.user_id,
            temperature=temperature,
        )
    else:
        # For AzureChatbot, all initialization happens in the constructor
        # Pass all required parameters directly, including file info for context
        all_file_infos = {query.file_id: file_info}  # Create the expected structure
        model = Chatbot(
            configs=configs,
            gcs_handler=gcs_handler,
            model_choice=query.model_choice,
            file_id=query.file_id,
            all_file_infos=all_file_infos,  # Pass file info for context
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
    temperature: float,
) -> JSONResponse:
    """
    Generate visualization configuration based on the response.

    Args:
        response: The response from the model
        query: The original query object
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
            try:
                if query.model_choice.startswith("gemini"):
                    response = get_gemini_non_rag_response(
                        configs,
                        current_question,
                        query.model_choice,
                        temperature,
                        max_tokens=4096,
                    )
                else:
                    response = get_azure_non_rag_response(
                        configs, current_question, max_tokens=4096
                    )
            except GeminiSafetyFilterError as e:
                # If safety filter blocks visualization, return error response
                logging.warning(
                    f"Visualization generation blocked by safety filter: {str(e)}"
                )
                raise HTTPException(
                    status_code=422, detail=f"Cannot generate visualization: {str(e)}"
                )

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
    except HTTPException:
        # Re-raise HTTPExceptions (including our safety filter ones)
        raise
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
    temperature: float,
    is_tabular: Optional[bool] = None,  # New parameter with default None
) -> Tuple[Any, Optional[bool]]:  # Returns (model, determined_is_tabular)
    """
    Helper function to initialize and cache chat models.

    This function has been refactored to reduce complexity by breaking down
    the logic into smaller, focused helper functions with clear responsibilities.
    """
    if is_multi_file:
        # Handle multi-file mode
        model, determined_is_tabular = await _initialize_multi_file_model(
            query, configs, gcs_handler, all_file_infos, temperature
        )
        logging.info(f"Model initialized for multi-file: {query.file_ids}")
        logging.info(f"  User: {query.user_id}, Model: {query.model_choice}")
        logging.info(f"  Is Tabular: {determined_is_tabular}")

    elif query.file_id:
        # Handle single file mode
        model, determined_is_tabular = await _initialize_single_file_model(
            query, configs, gcs_handler, all_file_infos, is_tabular, temperature
        )

    else:
        # Guard clause for invalid state
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

    return model, determined_is_tabular


async def _log_session_info(query: Query) -> None:
    """Log session information from the query.

    Args:
        query: Query containing mandatory session_id
    """
    logging.info(f"Processing chat request for session_id: {query.session_id}")


async def _process_single_file(query: Query, gcs_handler: GCSHandler) -> Dict[str, Any]:
    """
    Process information for a single file query.

    Args:
        query: Query containing file_id
        gcs_handler: GCS handler for file operations

    Returns:
        Dictionary containing processed single file information
    """
    file_id_logging = f"file_id: {query.file_id}"
    logging.info(f"Single-file chat request for {file_id_logging}")
    file_info_single = gcs_handler.get_file_info(query.file_id)

    logging.info(
        f"Processing single file {query.file_id} for session {query.session_id}"
    )

    is_tabular = False
    if file_info_single:
        is_tabular = file_info_single.get("is_tabular", False)
        logging.info(f"File {query.file_id} is_tabular from metadata: {is_tabular}")

    if not is_tabular:
        db_path = f"./chroma_db/{query.file_id}/tabular_data.db"
        is_tabular = os.path.exists(db_path)
        if is_tabular:
            logging.info(
                f"File {query.file_id} identified as tabular from local DB file"
            )

    if not file_info_single:
        if is_tabular:
            file_info_single = {
                "file_id": query.file_id,
                "is_tabular": True,
                "embeddings_status": "completed",
            }
        else:
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

    # No DB enrichment here; keep chat hot path lean. Upload/check handle it.

    all_file_infos = {query.file_id: file_info_single}
    model_key = f"{query.file_id}_{query.user_id}_{query.model_choice}"

    return {
        "all_file_infos": all_file_infos,
        "model_key": model_key,
        "file_id_logging": file_id_logging,
        "is_tabular": is_tabular,
    }


async def _check_file_embeddings(
    file_ids: List[str], model_choice: str
) -> List[Dict[str, Any]]:
    """Check if embeddings exist for multiple files and model choice.

    Args:
        file_ids: List of file IDs to check
        model_choice: The model choice to check embeddings for

    Returns:
        List of dictionaries containing embeddings check results
    """
    if not file_ids:
        raise HTTPException(
            status_code=400,
            detail="No file IDs provided for embedding check.",
        )

    # Check embeddings for all files in parallel
    tasks = [
        embedding_handler.check_embeddings_exist(file_id, model_choice)
        for file_id in file_ids
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and collect any errors
    processed_results = []
    errors = []

    for i, result in enumerate(results):
        file_id = file_ids[i]
        if isinstance(result, Exception):
            errors.append(f"Error checking file {file_id}: {str(result)}")
            processed_results.append(
                {
                    "file_id": file_id,
                    "file_name": None,
                    "embeddings_exist": False,
                    "error": str(result),
                    "model_type": "azure",
                    "status": "error",
                }
            )
        else:
            processed_results.append(result)
            if not result["embeddings_exist"]:
                errors.append(f"Embeddings not found for file {file_id}")

    return processed_results


async def _check_file_embeddings_safe(
    file_ids: List[str], model_choice: str
) -> List[Dict[str, Any]]:
    """Check if embeddings exist for multiple files and model choice without raising exceptions.

    This is a safe version of _check_file_embeddings that returns the result without raising HTTPException.

    Args:
        file_ids: List of file IDs to check
        model_choice: The model choice to check embeddings for

    Returns:
        List of dictionaries containing embeddings check results
    """
    try:
        if not file_ids:
            return []

        # Check embeddings for all files in parallel
        tasks = [
            embedding_handler.check_embeddings_exist(file_id, model_choice)
            for file_id in file_ids
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions gracefully
        processed_results = []
        for i, result in enumerate(results):
            file_id = file_ids[i]
            if isinstance(result, Exception):
                logging.error(
                    f"Error checking embeddings for file {file_id}: {str(result)}"
                )
                processed_results.append(
                    {
                        "file_id": file_id,
                        "embeddings_exist": False,
                        "error": str(result),
                        "model_type": "azure",
                        "status": "error",
                    }
                )
            else:
                processed_results.append(result)

        return processed_results

    except Exception as e:
        logging.error(f"Error in _check_file_embeddings_safe: {str(e)}")
        # Return list with error for each file
        return [
            {
                "file_id": file_id,
                "embeddings_exist": False,
                "error": str(e),
                "model_type": "azure",
                "status": "error",
            }
            for file_id in file_ids
        ]


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
    logging.info(f"Processing files for session {query.session_id}: {query.file_ids}")

    if not query.file_ids:
        raise HTTPException(
            status_code=404,
            detail="No files specified for this chat session. Please upload files first.",
        )

    # Skip redundant embeddings existence checks here; initialization will ensure local cache

    # Check each file to see if it's tabular
    tabular_files, non_tabular_files = await _classify_file_types(query.file_ids)

    # Get file info for all files
    all_file_infos = await _get_file_info_multi(query.file_ids, gcs_handler)

    # No DB enrichment in chat path; upload/check handle embedding_type.

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
    temperature: float,
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
    # If the model returned a table-like structure [headers, *rows], format it
    if isinstance(response, list):
        try:
            if response and isinstance(response[0], (list, tuple)):
                return format_table_response(response)
        except Exception:
            pass

    if generate_visualization:
        return handle_visualization(response, query, is_tabular, configs, temperature)

    final_response_data = {
        "response": str(response),
        "is_table": is_tabular if is_tabular is not None else False,
    }

    if is_multi_file and query.file_ids:
        final_response_data["sources"] = query.file_ids
    elif query.file_id:
        final_response_data["sources"] = [query.file_id]

    return final_response_data


def _filter_context_from_file_info(file_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filters a file_info dictionary to include only the necessary keys for context
    and essential application logic, based on the file type.

    Args:
        file_info: The original file_info dictionary.

    Returns:
        A new, filtered dictionary.
    """
    if not file_info:
        return {}

    context_info = {}

    # --- Essential keys for application logic ---
    for key in ["is_tabular", "embeddings_status", "file_id", "embedding_type"]:
        if key in file_info:
            context_info[key] = file_info[key]

    # --- Context-specific keys based on user requirements ---

    # `original_filename` is required for all types.
    if "original_filename" in file_info:
        context_info["original_filename"] = file_info["original_filename"]

    # For URL-based files, add the `url`.
    if "url" in file_info:
        context_info["url"] = file_info["url"]

    # For tabular files, add the `database_summary`.
    if file_info.get("is_tabular") and "database_summary" in file_info:
        context_info["database_summary"] = file_info["database_summary"]

    return context_info


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
    is_multi_file = bool(query.file_ids and len(query.file_ids) > 1)

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
        is_tabular = single_file_info.get(
            "is_tabular", False
        )  # Get is_tabular from the processed file info

        logging.info(
            f"Single-file chat processing complete for {file_id_logging}, is_tabular={is_tabular}"
        )
    else:
        # Neither file_id nor file_ids provided
        raise HTTPException(
            status_code=400,
            detail="Either file_id (for single chat) or file_ids (for multi-chat) must be provided.",
        )

    # Filter the file info for each file to only include context-relevant data
    filtered_all_file_infos = {
        file_id: _filter_context_from_file_info(file_info)
        for file_id, file_info in all_file_infos.items()
    }

    # Return consolidated result with consistent structure
    return {
        "is_multi_file": is_multi_file,
        "all_file_infos": filtered_all_file_infos,  # Use the filtered info
        "model_key": model_key,
        "file_id_logging": file_id_logging,
        "is_tabular": is_tabular,
        "generate_visualization": generate_visualization,
    }


def _get_default_temperature(model_choice: str) -> float:
    """
    Get the default temperature based on the model choice.

    Args:
        model_choice: The chosen model name

    Returns:
        Default temperature value (OpenAI: 0.5, Gemini: 0.8)
    """
    if model_choice.lower() in ["gemini-2.5-flash", "gemini-2.5-pro"]:
        return 0.8  # Higher temperature for Gemini models for more creativity
    else:
        return 0.5  # Lower temperature for OpenAI models for better coherence


async def _detect_visualization_need(
    question: str, configs: dict, temperature: float = 0.8
) -> bool:
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
        try:
            logging.info("Detecting if visualization is needed..")
            # vis_detection_response = get_gemini_non_rag_response(
            #     configs, question_for_detection, "gemini-2.5-flash", temperature
            # )
            vis_detection_response = get_azure_non_rag_response(
                configs, question_for_detection, model_choice="gpt_4_1_nano"
            )
            if (
                vis_detection_response.lower() == "true"
                or "true" in vis_detection_response.lower()
            ):
                generate_visualization = True
        except Exception as e:
            # For any errors with Azure OpenAI, default to False
            logging.error(f"Error in visualization detection: {str(e)}")
            generate_visualization = False

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

        # Set temperature - use provided value or model-specific default
        if query.temperature is not None:
            temperature = query.temperature
        else:
            temperature = _get_default_temperature(query.model_choice)
        logging.info(f"Using temperature {temperature} for model {query.model_choice}")

        # Use config flag for chart generation
        if not configs.generate_visualization:
            # If flag is explicitly set to False, disable visualization
            generate_visualization = False
        else:
            # Otherwise, use the existing detection logic
            generate_visualization = await _detect_visualization_need(
                current_actual_question, configs, temperature
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

        # Do not perform DB enrichment of embedding_type during chat; upload/check handle this

        model_info = initialized_models.get(model_key)
        model = model_info["model"] if model_info else None

        try:
            if not model:
                # Call the helper function to initialize the model
                # Pass the is_tabular flag from _process_file_info
                model, is_tabular = await _initialize_chat_model(
                    query=query,
                    configs=configs,
                    gcs_handler=gcs_handler,
                    initialized_models=initialized_models,
                    model_key=model_key,
                    is_multi_file=is_multi_file,
                    all_file_infos=all_file_infos,
                    file_id_logging=file_id_logging,
                    temperature=temperature,
                    is_tabular=file_data[
                        "is_tabular"
                    ],  # Pass is_tabular flag from file_data
                )

            if generate_visualization and not is_tabular:
                question_to_model = current_actual_question + VISUALISATION_PROMPT
            else:
                question_to_model = current_actual_question

            # Different history handling for tabular vs non-tabular data
            if is_tabular:
                # For tabular data, pass the full conversation array
                # The TabularDataHandler will resolve contextual references internally
                if len(query.text) > 1:
                    # Pass full array including history
                    response = model.get_answer(query.text)
                else:
                    # Single question, pass as-is
                    response = model.get_answer(question_to_model)
            else:
                # For PDF/document chat, use traditional string concatenation
                if len(query.text) > 1:
                    previous_messages = "\n".join(
                        [f"Previous message: {msg}" for msg in query.text[:-1]]
                    )
                    chat_context = (
                        f"{previous_messages}\nCurrent question: {question_to_model}"
                    )
                else:
                    chat_context = question_to_model

                # Todo: If multiple file and tabular collect all responses
                # and give to non rag and return answer.
                response = model.get_answer(chat_context)

            return await _format_chat_response(
                response=response,
                query=query,
                generate_visualization=generate_visualization,
                is_tabular=is_tabular,
                is_multi_file=is_multi_file,
                configs=configs,
                temperature=temperature,
            )

        finally:
            if isinstance(model, TabularDataHandler):
                model.cleanup()

    except HTTPException:
        raise
    except GeminiSafetyFilterError as e:
        # Handle Gemini safety filter errors with structured error
        logging.warning(
            f"Gemini safety filter blocked response for {file_id_logging}: {str(e)}"
        )
        raise SafetyFilterError(
            f"Content blocked by safety filters: {str(e)}",
            details={"file_id": file_id_logging},
        )
    except Exception as e:
        logging.error(
            f"Error in chat endpoint: {str(e)} for {file_id_logging}", exc_info=True
        )
        # Raise structured error instead of generic HTTPException
        from rtl_rag_chatbot_api.common.errors import BaseAppError, ErrorRegistry

        raise BaseAppError(
            ErrorRegistry.ERROR_LLM_GENERATION_FAILED,
            f"Chat request failed: {str(e)}",
            details={"file_id": file_id_logging},
        )


@app.get("/available-models")
async def get_available_models(current_user=Depends(get_current_user)):
    """
    Endpoint to retrieve a list of available models including Azure LLM models, Gemini 2.5 models,
    and image generation models.
    """
    # Get available models from config
    azure_models = list(configs.azure_llm.models.keys())
    gemini_models = ["gemini-2.5-flash", "gemini-2.5-pro"]
    anthropic_models = ["claude-sonnet-4@20250514", "claude-sonnet-4-5"]
    # Add individual image models and the combined option
    image_models = [
        "dall-e-3",
        configs.vertexai_imagen.model_name,
        "imagen-4.0-ultra-generate-001",
        "imagen-4.0-generate-001",
        "Dalle + Imagen",
    ]

    # Return combined list with model categories
    return {
        "models": azure_models + gemini_models + anthropic_models + image_models,
        "model_types": {
            "text": azure_models + gemini_models + anthropic_models,
            "image": image_models,
        },
    }


@app.post("/file/cleanup")
async def manual_cleanup(
    request: CleanupRequest = Body(default=CleanupRequest()),
    current_user=Depends(get_current_user),
):
    """Endpoint to manually trigger cleanup."""
    if not configs.use_file_hash_db:
        return {"status": "Cleanup skipped - database is disabled"}

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
                    if configs.use_file_hash_db:
                        cleanup_coordinator = CleanupCoordinator(
                            configs, SessionLocal, gcs_handler
                        )
                        cleanup_coordinator.cleanup_chroma_instance(
                            file_id, include_gcs=request.include_gcs
                        )
                    else:
                        # When database is disabled, just clean up ChromaDB and GCS
                        cleanup_coordinator = CleanupCoordinator(
                            configs, None, gcs_handler
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
                if configs.use_file_hash_db:
                    cleanup_coordinator = CleanupCoordinator(
                        configs, SessionLocal, gcs_handler
                    )
                else:
                    # When database is disabled, just clean up ChromaDB and GCS
                    cleanup_coordinator = CleanupCoordinator(configs, None, gcs_handler)

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
    Get a streaming response from a Gemini model without RAG context.

    Args:
        request (ChatRequest): Request body containing:
            - prompt (str): The prompt to send to the model
            - model (str): The Gemini model to use. Must be either "gemini-2.5-flash" or "gemini-2.5-pro".
            - temperature (float, optional): The sampling temperature. Defaults to 0.8.
        current_user: Authenticated user information

    Returns:
        StreamingResponse: A streaming response with the model's output
    """
    try:
        # Validate model choice
        if request.model not in ["gemini-2.5-flash", "gemini-2.5-pro"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid model choice. Use 'gemini-2.5-flash' or 'gemini-2.5-pro'.",
            )

        model_handler = ModelHandler(configs, gcs_handler)
        model = model_handler.initialize_model(
            request.model, file_id=None, embedding_type="gemini"
        )
        return StreamingResponse(
            model.get_gemini_response_stream(request.message), media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/chat/anthropic")
async def get_anthropic_response(
    request: ChatRequest, current_user=Depends(get_current_user)
):
    """
    Get a non-RAG response from an Anthropic (Vertex) model.

    Args:
        request (ChatRequest): Request body containing:
            - prompt (str): The prompt to send to the model
            - model (str): The Anthropic model to use. Must be "claude-sonnet-4@20250514" or "claude-sonnet-4-5".
            - temperature (float, optional): The sampling temperature. Defaults to 0.8.
        current_user: Authenticated user information

    Returns:
        PlainTextResponse: A non-streaming response with the model's output
    """
    try:
        # Validate model choice
        if request.model not in ["claude-sonnet-4@20250514", "claude-sonnet-4-5"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid model choice. Use 'claude-sonnet-4@20250514' or 'claude-sonnet-4-5'.",
            )

        from rtl_rag_chatbot_api.chatbot import get_anthropic_non_rag_response

        answer = get_anthropic_non_rag_response(
            config=configs,
            prompt=request.message,
            model_choice=request.model,
            temperature=request.temperature or 0.8,
            max_tokens=4096,
        )
        return PlainTextResponse(answer)
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


# Removed deprecated delete_google_embeddings endpoint as part of unified Azure embeddings approach


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
            - model_choice (str, optional): Model to use ("dall-e-3" or Imagen variant
              such as "imagen-3.0-generate-002"). Default: "dall-e-3".
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
            # Only pass an override when a specific Imagen variant is requested (imagen-...)
            imagen_override = (
                request.model_choice
                if request.model_choice.lower().startswith("imagen-")
                else None
            )
            result = imagen_handler.generate_image(
                prompt=request.prompt,
                size=request.size,
                n=request.n,
                model_name=imagen_override,
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


async def _classify_files_for_model_init(file_ids: List[str]) -> Dict[str, List[str]]:
    """
    Classify files as tabular or non-tabular for model initialization.

    Args:
        file_ids: List of file IDs to classify

    Returns:
        Dictionary with 'tabular' and 'non_tabular' lists
    """
    tabular_file_ids = []
    non_tabular_file_ids = []

    for file_id in file_ids:
        db_path = f"./chroma_db/{file_id}/tabular_data.db"
        if os.path.exists(db_path):
            tabular_file_ids.append(file_id)
        else:
            non_tabular_file_ids.append(file_id)

    return {"tabular": tabular_file_ids, "non_tabular": non_tabular_file_ids}


def _extract_database_summaries(
    file_ids: List[str], all_file_infos: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract database summaries from file info for given file IDs.

    Args:
        file_ids: List of file IDs to extract summaries for
        all_file_infos: All file information

    Returns:
        Dictionary mapping file_id to database_summary
    """
    database_summaries = {}
    for f_id in file_ids:
        if f_id in all_file_infos and "database_summary" in all_file_infos[f_id]:
            database_summaries[f_id] = all_file_infos[f_id]["database_summary"]
            logging.info(f"Found database summary for file_id: {f_id}")
        else:
            logging.warning(f"No database_summary found for file_id: {f_id}")

    return database_summaries


def _create_tabular_data_handler(
    configs: dict,
    file_ids: List[str],
    model_choice: str,
    database_summaries: Dict[str, Any],
    all_file_infos: Dict[str, Any],
    temperature: float,
) -> Any:
    """
    Create TabularDataHandler with error handling and fallback.

    Args:
        configs: Configuration dictionary
        file_ids: List of tabular file IDs
        model_choice: Model choice string
        database_summaries: Database summaries mapping

    Returns:
        TabularDataHandler instance

    Raises:
        Exception: If TabularDataHandler creation fails
    """
    logging.info("=== Creating TabularDataHandler ===")
    logging.info(f"File IDs: {file_ids}")
    logging.info(f"Model choice: {model_choice}")
    logging.info(f"Temperature: {temperature}")
    logging.info(
        f"Database summaries keys: {list(database_summaries.keys()) if database_summaries else 'None'}"
    )
    logging.info(
        f"All file infos keys: {list(all_file_infos.keys()) if all_file_infos else 'None'}"
    )

    # Check if the required directories and files exist before creating TabularDataHandler
    for file_id in file_ids:
        data_dir = f"./chroma_db/{file_id}"
        db_path = os.path.join(data_dir, "tabular_data.db")

        logging.info(f"Checking file_id: {file_id}")
        logging.info(f"  Data directory: {data_dir}")
        logging.info(f"  Database path: {db_path}")

        # Check directory existence
        if not os.path.exists(data_dir):
            logging.error(f"  Data directory does not exist: {data_dir}")
        else:
            logging.info(f"  Data directory exists: {data_dir}")
            # Check directory contents
            try:
                dir_contents = os.listdir(data_dir)
                logging.info(f"  Directory contents: {dir_contents}")
            except Exception as e:
                logging.error(f"  Error listing directory contents: {str(e)}")

        # Check database file existence
        if not os.path.exists(db_path):
            logging.error(f"  Database file does not exist: {db_path}")
        else:
            logging.info(f"  Database file exists: {db_path}")
            # Check file size and permissions
            try:
                file_size = os.path.getsize(db_path)
                file_stat = os.stat(db_path)
                logging.info(f"  Database file size: {file_size} bytes")
                logging.info(f"  Database file permissions: {oct(file_stat.st_mode)}")
            except Exception as e:
                logging.error(f"  Error getting file info: {str(e)}")

    try:
        logging.info("Creating TabularDataHandler instance...")
        handler = TabularDataHandler(
            configs,
            file_id=file_ids[0],  # For backward compatibility
            model_choice=model_choice,
            file_ids=file_ids,
            database_summaries_param=database_summaries if database_summaries else None,
            all_file_infos=all_file_infos,
            temperature=temperature,
        )
        logging.info("TabularDataHandler instance created successfully")
        return handler
    except Exception as e:
        logging.error(f"Failed to create TabularDataHandler: {str(e)}")
        logging.error(f"Exception type: {type(e).__name__}")
        import traceback

        logging.error(f"Full traceback: {traceback.format_exc()}")
        raise


def _create_rag_model_for_multi_file(
    query: Query,
    configs: dict,
    gcs_handler: Any,
    all_file_infos: Dict[str, Any],
    is_gemini: bool,
) -> Any:
    """
    Create appropriate RAG model for multi-file non-tabular content.

    Args:
        query: Query object
        configs: Configuration dictionary
        gcs_handler: GCS handler instance
        all_file_infos: All file information
        is_gemini: Whether this is a Gemini model

    Returns:
        Model instance (GeminiHandler, AnthropicHandler, or Chatbot)
    """
    if is_gemini:
        logging.info(
            f"Using GeminiHandler for multi-file with model: {query.model_choice}"
        )
        return GeminiHandler(
            configs=configs,
            gcs_handler=gcs_handler,
            model_choice=query.model_choice,
            file_ids=query.file_ids,  # type: ignore
            all_file_infos=all_file_infos,
            user_id=query.user_id,
        )
    elif query.model_choice in ["claude-sonnet-4@20250514", "claude-sonnet-4-5"]:
        logging.info(
            f"Using AnthropicHandler for multi-file with model: {query.model_choice}"
        )
        from rtl_rag_chatbot_api.chatbot.anthropic_handler import AnthropicHandler

        return AnthropicHandler(
            configs=configs,
            gcs_handler=gcs_handler,
            model_choice=query.model_choice,
            file_ids=query.file_ids,  # type: ignore
            all_file_infos=all_file_infos,
            user_id=query.user_id,
        )
    else:
        logging.info(
            f"Using AzureChatbot for multi-file with model: {query.model_choice}"
        )
        return Chatbot(
            configs=configs,
            gcs_handler=gcs_handler,
            model_choice=query.model_choice,
            file_ids=query.file_ids,  # type: ignore
            all_file_infos=all_file_infos,
        )


async def _handle_mixed_file_types(
    query: Query,
    configs: dict,
    gcs_handler: Any,
    all_file_infos: Dict[str, Any],
    tabular_file_ids: List[str],
    non_tabular_file_ids: List[str],
) -> Tuple[Any, bool]:
    """
    Handle mixed file types (tabular + non-tabular) in multi-file mode.

    Args:
        query: Query object
        configs: Configuration dictionary
        gcs_handler: GCS handler instance
        all_file_infos: All file information
        tabular_file_ids: List of tabular file IDs
        non_tabular_file_ids: List of non-tabular file IDs

    Returns:
        Tuple of (model, is_tabular_flag)
    """
    logging.info("Mixed file types detected in multi-file chat (tabular + non-tabular)")
    logging.info(f"Tabular files: {tabular_file_ids}")
    logging.info(f"Non-tabular files: {non_tabular_file_ids}")

    # For mixed files, use RAG model as it can handle non-tabular content
    logging.info("Using RAG model for mixed file types")
    model = Chatbot(
        configs=configs,
        gcs_handler=gcs_handler,
        model_choice=query.model_choice,
        file_ids=query.file_ids,  # type: ignore
        all_file_infos=all_file_infos,
    )
    return model, False


async def _handle_all_tabular_files(
    query: Query,
    configs: dict,
    all_file_infos: Dict[str, Any],
    tabular_file_ids: List[str],
    temperature: float,
) -> Tuple[Any, bool]:
    """
    Handle all tabular files in multi-file mode.

    Args:
        query: Query object
        configs: Configuration dictionary
        all_file_infos: All file information
        tabular_file_ids: List of tabular file IDs

    Returns:
        Tuple of (model, is_tabular_flag)
    """
    logging.info(f"All files are tabular in multi-file: {query.file_ids}")
    logging.info("Initializing TabularDataHandler with all tabular files")

    # Guard clause for empty tabular files (should not happen)
    if not tabular_file_ids:
        logging.warning("No tabular files found despite earlier check")
        # This should never happen but we'll create a fallback Chatbot anyway
        # We don't have gcs_handler in this function but it's not used for tabular fallback
        return None, False

    try:
        database_summaries = _extract_database_summaries(
            tabular_file_ids, all_file_infos
        )
        model = _create_tabular_data_handler(
            configs,
            tabular_file_ids,
            query.model_choice,
            database_summaries,
            all_file_infos,
            temperature,
        )
        logging.info(
            f"Successfully initialized TabularDataHandler for {len(tabular_file_ids)} files"
        )
        return model, True

    except Exception as e:
        logging.error(f"Failed to initialize TabularDataHandler: {str(e)}")
        logging.info("Falling back to RAG model for tabular data")
        # Return None to indicate failure - the calling function will handle fallback
        return None, False


async def _handle_all_non_tabular_files(
    query: Query,
    configs: dict,
    gcs_handler: Any,
    all_file_infos: Dict[str, Any],
    non_tabular_file_ids: List[str],
) -> Tuple[Any, bool]:
    """
    Handle all non-tabular files in multi-file mode.

    Args:
        query: Query object
        configs: Configuration dictionary
        gcs_handler: GCS handler instance
        all_file_infos: All file information
        non_tabular_file_ids: List of non-tabular file IDs

    Returns:
        Tuple of (model, is_tabular_flag)
    """
    logging.info(f"All files are non-tabular: {non_tabular_file_ids}")

    # Check if this is a Gemini model
    is_gemini = query.model_choice.lower() in ["gemini-2.5-flash", "gemini-2.5-pro"]

    model = _create_rag_model_for_multi_file(
        query, configs, gcs_handler, all_file_infos, is_gemini
    )
    return model, False


async def _initialize_multi_file_model(
    query: Query,
    configs: dict,
    gcs_handler: Any,
    all_file_infos: Dict[str, Any],
    temperature: float,
) -> Tuple[Any, bool]:
    """
    Initialize model for multi-file mode with proper classification and handling.

    Args:
        query: Query object
        configs: Configuration dictionary
        gcs_handler: GCS handler instance
        all_file_infos: All file information

    Returns:
        Tuple of (model, is_tabular_flag)
    """
    # Ensure local embeddings are available for all files in a multi-file chat
    for file_id in query.file_ids:
        # All models use Azure embeddings in the unified approach
        embedding_type = "azure"

        model_path = os.path.join(
            f"./chroma_db/{file_id}", embedding_type, "chroma.sqlite3"
        )
        if not os.path.exists(model_path):
            logging.info(
                f"Local embeddings not found for {file_id} at {model_path}, downloading from GCS."
            )
            gcs_handler.download_files_from_folder_by_id(file_id)

    # Classify files
    file_classification = await _classify_files_for_model_init(query.file_ids)
    tabular_file_ids = file_classification["tabular"]
    non_tabular_file_ids = file_classification["non_tabular"]

    has_tabular_files = bool(tabular_file_ids)
    has_non_tabular_files = bool(non_tabular_file_ids)

    # Handle different file type combinations
    if has_tabular_files and has_non_tabular_files:
        # Mixed file types
        return await _handle_mixed_file_types(
            query,
            configs,
            gcs_handler,
            all_file_infos,
            tabular_file_ids,
            non_tabular_file_ids,
        )
    elif has_tabular_files and not has_non_tabular_files:
        # All tabular files
        model, is_tabular = await _handle_all_tabular_files(
            query, configs, all_file_infos, tabular_file_ids, temperature
        )
        # If tabular handler failed, fall back to standard Chatbot
        if model is None:
            logging.info(
                "Falling back to standard Chatbot for failed tabular initialization"
            )
            model = Chatbot(
                configs=configs,
                gcs_handler=gcs_handler,
                model_choice=query.model_choice,
                file_ids=query.file_ids,  # type: ignore
                all_file_infos=all_file_infos,
            )
            is_tabular = False
        return model, is_tabular
    else:
        # All non-tabular files
        return await _handle_all_non_tabular_files(
            query, configs, gcs_handler, all_file_infos, non_tabular_file_ids
        )


async def _initialize_single_file_tabular_model(
    query: Query, configs: dict, all_file_infos: Dict[str, Any], temperature: float
) -> Any:
    """
    Initialize TabularDataHandler for single file with proper error handling.

    Args:
        query: Query object
        configs: Configuration dictionary
        all_file_infos: All file information

    Returns:
        TabularDataHandler instance or None if failed
    """
    logging.info(f"Initializing TabularDataHandler for single file: {query.file_id}")

    try:
        # Extract database summary from single_file_info if available
        database_summaries = {}
        single_file_info = all_file_infos.get(query.file_id)
        if single_file_info and "database_summary" in single_file_info:
            database_summaries[query.file_id] = single_file_info["database_summary"]
            logging.info(f"Found database summary for file_id: {query.file_id}")
        else:
            logging.warning(f"No database_summary found for file_id: {query.file_id}")

        model = TabularDataHandler(
            configs,
            file_id=query.file_id,
            model_choice=query.model_choice,
            database_summaries_param=database_summaries if database_summaries else None,
            all_file_infos=all_file_infos,
            temperature=temperature,
        )
        logging.info(f"Successfully initialized TabularDataHandler for {query.file_id}")
        return model

    except Exception as e:
        logging.error(f"Failed to initialize TabularDataHandler: {str(e)}")
        return None


async def _initialize_single_file_model(
    query: Query,
    configs: dict,
    gcs_handler: Any,
    all_file_infos: Dict[str, Any],
    is_tabular: Optional[bool],
    temperature: float,
) -> Tuple[Any, bool]:
    """
    Initialize model for single file mode.

    Args:
        query: Query object
        configs: Configuration dictionary
        gcs_handler: GCS handler instance
        all_file_infos: All file information
        is_tabular: Flag indicating if file is tabular (can be None)

    Returns:
        Tuple of (model, determined_is_tabular)
    """
    # Guard clause for missing is_tabular flag
    if is_tabular is None:
        logging.warning(
            f"is_tabular not provided for {query.file_id}, defaulting to non-tabular"
        )
        is_tabular = False

    # Try to initialize TabularDataHandler for tabular files
    if is_tabular:
        model = await _initialize_single_file_tabular_model(
            query, configs, all_file_infos, temperature
        )
        if model:
            return model, True
        # If TabularDataHandler failed, fall back to RAG model
        is_tabular = False

    # Initialize RAG model for non-tabular files or fallback
    single_file_info_for_init = all_file_infos.get(query.file_id)
    if not single_file_info_for_init:
        raise HTTPException(
            status_code=500,
            detail="Internal error: file_info missing for single file RAG model initialization.",
        )

    logging.info(f"Initializing RAG model for single file: {query.file_id}")
    logging.info(f"  User: {query.user_id}, Model: {query.model_choice}")
    model = initialize_rag_model(
        query, configs, gcs_handler, single_file_info_for_init, temperature
    )
    return model, is_tabular


@app.get("/test-db-connection")
async def test_db_connection():
    """Test the database connection by executing a simple query and return JSON."""
    if not configs.use_file_hash_db:
        return {
            "status": "disabled",
            "message": "Database is disabled - USE_FILE_HASH_DB is set to false",
        }

    try:
        with get_db_session() as db:
            # Execute a simple query to check the connection
            result = db.execute(text("SELECT 1"))
            value = result.scalar()
            return {"status": "success", "result": int(value)}
    except Exception as e:
        logging.error(f"Database connection test failed: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/insert-file-info")
async def insert_file_info(
    request: Request,
    file_id: str = Form(...),
    file_hash: str = Form(...),
    filename: str = Form(None),
    embedding_type: str = Form(None),
    current_user=Depends(get_current_user),
):
    """Insert a new record into the FileInfo table."""
    if not configs.use_file_hash_db:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "Database is disabled - USE_FILE_HASH_DB is set to false",
            },
        )

    from rtl_rag_chatbot_api.common.db import insert_file_info_record

    # Use the context manager properly
    with get_db_session() as db:
        result = insert_file_info_record(
            db, file_id, file_hash, filename, embedding_type, configs
        )

        if result["status"] == "success":
            return JSONResponse(status_code=200, content=result)
        else:
            return JSONResponse(status_code=500, content=result)


@app.delete("/delete-all-file-info")
async def delete_all_file_info(current_user=Depends(get_current_user)):
    """
    Delete all records from the FileInfo table.

    This endpoint will permanently remove all file information records from the database.
    Use with caution as this action cannot be undone.

    Args:
        current_user: Authenticated user information

    Returns:
        dict: Response containing the deletion result
    """
    if not configs.use_file_hash_db:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "Database is disabled - USE_FILE_HASH_DB is set to false",
            },
        )

    from rtl_rag_chatbot_api.common.db import delete_all_file_info_records

    try:
        with get_db_session() as db:
            result = delete_all_file_info_records(db)

            if result["status"] == "success":
                if result["deleted"]:
                    return JSONResponse(
                        status_code=200,
                        content={
                            "message": f"Successfully deleted all {result['deleted_count']} FileInfo records",
                            "deleted_count": result["deleted_count"],
                            "deleted_records": result.get("deleted_records", []),
                        },
                    )
                else:
                    return JSONResponse(
                        status_code=200,
                        content={"message": result["message"], "deleted_count": 0},
                    )
            else:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "Failed to delete FileInfo records",
                        "message": result["message"],
                    },
                )

    except Exception as e:
        logging.error(f"Error deleting all FileInfo records: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": f"An error occurred while deleting FileInfo records: {str(e)}",
            },
        )


def _exclude_uploaded_files_already_handled(
    all_files, detailed_info, available_migration_files
):
    """
    Remove uploaded files from all_files that are already covered by migration logic
    (files scheduled for migration) or recognized as existing with no migration
    (source == 'uploaded_file'). This prevents duplicate preprocessing.
    """
    # Filenames of uploaded files scheduled for migration
    migration_filenames = set()
    for file_detail in available_migration_files:
        if file_detail.get("source") == "uploaded_file":
            original_filename = file_detail.get("original_filename", "")
            if original_filename and original_filename != "N/A":
                migration_filenames.add(original_filename)

    # Filenames of uploaded files that are existing and need no migration
    existing_no_migration_details = detailed_info.get("existing_files_no_migration", [])
    existing_no_migration_uploaded_filenames = set(
        d.get("original_filename")
        for d in existing_no_migration_details
        if d.get("source") == "uploaded_file"
        and d.get("original_filename")
        and d.get("original_filename") != "N/A"
    )

    filenames_to_exclude = migration_filenames.union(
        existing_no_migration_uploaded_filenames
    )

    filtered_files = [
        file for file in all_files if file.filename not in filenames_to_exclude
    ]

    if len(filtered_files) != len(all_files):
        logging.info(
            "Filtered %s uploaded file(s) already handled by migration/no-migration logic",
            len(all_files) - len(filtered_files),
        )

    return filtered_files


async def _prepare_migration_file_for_pipeline(
    file_id: str,
    original_filename: str,
    temp_file_path: str,
    preserved_usernames: list,
    username: str,
    gcs_handler: GCSHandler,
    configs: dict,
    embedding_handler,
) -> tuple[str, str, bool]:
    """Prepare a migration file to follow the standard pipeline.

    - Compute file_hash
    - Ensure encryption uploaded under the SAME file_id
    - Seed file_info with in_progress and essential flags

    Returns: (file_id, original_filename, is_tabular_flag)
    """
    # Combine usernames (duplicates allowed for frequency tracking)
    combined_usernames = (
        preserved_usernames.copy() if isinstance(preserved_usernames, list) else []
    )
    combined_usernames.append(username)

    # Detect tabular/image flags
    import os as _os

    base_name = _os.path.basename(original_filename)
    ext = _os.path.splitext(base_name)[1].lower()
    is_tabular_flag = ext in [
        ".csv",
        ".xlsx",
        ".xls",
        ".db",
        ".sqlite",
        ".sqlite3",
    ]

    # Compute file hash
    file_hash = None
    try:
        import hashlib as _hashlib

        with open(temp_file_path, "rb") as _f:
            file_hash = _hashlib.md5(_f.read()).hexdigest()
    except Exception as _hash_err:
        logging.warning(
            f"Failed to compute file hash for migration file {file_id}: {_hash_err}"
        )

    # Ensure encryption under the same file_id (no-op for tabular)
    try:
        from rtl_rag_chatbot_api.chatbot.utils.file_encryption_manager import (
            FileEncryptionManager,
        )

        _enc_mgr = FileEncryptionManager(gcs_handler)
        await _enc_mgr.ensure_file_encryption(
            file_id=file_id,
            original_filename=base_name,
            temp_file_path=temp_file_path,
            is_tabular=is_tabular_flag,
            is_database=False,
        )
    except Exception as _enc_err:
        logging.warning(
            f"Failed to ensure encryption for migration file {file_id}: {_enc_err}"
        )

    # Note: Do not start embedding creation here. Migration files will be
    # embedded exactly once by the standard parallel document processing flow.

    # DB maintenance: delete old row and insert new
    try:
        from rtl_rag_chatbot_api.common.db import (
            delete_file_info_by_file_id,
            insert_file_info_record,
        )

        if hasattr(configs, "use_file_hash_db") and configs.use_file_hash_db:
            with get_db_session() as db_session:
                try:
                    del_result = delete_file_info_by_file_id(db_session, file_id)
                    if del_result.get("status") == "success":
                        deleted_count = del_result.get("deleted_count", 0)
                        logging.info(
                            f"Deleted {deleted_count} old FileInfo records for file_id: {file_id}"
                        )
                except Exception as del_err:
                    logging.warning(
                        f"Failed to delete existing FileInfo for {file_id}: {del_err}"
                    )
                # insert new
                if file_hash:
                    _ = insert_file_info_record(
                        db_session,
                        file_id,
                        file_hash,
                        base_name,
                        None,  # Use configurable default
                        configs,
                    )
    except Exception as _db_err:
        logging.warning(
            f"Failed to refresh FileInfo DB record for {file_id}: {_db_err}"
        )
    return file_id, original_filename, is_tabular_flag


@app.post("/chat/generate-title")
async def generate_chat_title(
    request: TitleGenerationRequest, current_user=Depends(get_current_user)
):
    """
    Generate an automatically generated title for a chat conversation based on its content.

    This endpoint uses GPT nano model with the TITLE_GENERATION_PROMPT to create
    concise, descriptive titles (around 5 words) that accurately reflect the
    conversation's main topic or task.

    Args:
        request (TitleGenerationRequest): Request body containing:
            - conversation (List[str]): Array of strings alternating between user questions and assistant answers
            - model_choice (Optional[str]): Model to use (defaults to "gpt_4_1_nano")
        current_user: Authenticated user information

    Returns:
        dict: Dictionary containing:
            - title (str): Generated conversation title
            - success (bool): Whether title generation was successful

    Raises:
        HTTPException: If title generation fails or response parsing fails
    """
    try:
        import json

        # Format the conversation with the title generation prompt
        full_prompt = (
            TITLE_GENERATION_PROMPT + "\n\n" + json.dumps(request.conversation)
        )

        logging.info(
            f"Generating title for conversation with {len(request.conversation)}"
            f"messages using {request.model_choice}"
        )

        # Call Azure OpenAI to generate the title
        response = get_azure_non_rag_response(
            configs=configs,
            query=full_prompt,
            model_choice=request.model_choice,
            max_tokens=100,  # Limit tokens for title generation
        )

        # Parse the JSON response
        try:
            parsed_response = json.loads(response.strip())
            title = parsed_response.get("title", "").strip()

            if not title:
                raise ValueError("Empty title in response")

            logging.info(f"Successfully generated title: '{title}'")

            return {"title": title, "success": True}

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {response}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse title generation response: {str(e)}",
            )

    except Exception as e:
        logging.error(f"Error generating chat title: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate chat title: {str(e)}"
        )
