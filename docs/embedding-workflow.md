# Embedding Generation Workflow

## Overview

The embedding generation process in the RAG PDF API has been optimized for both performance and user experience. The system now features fully parallel processing of multiple files with immediate chat availability after local embedding creation, while cloud storage operations run asynchronously in the background. This document details the workflow, states, and components involved in this highly concurrent approach.

### Unified Embedding Approach

As of June 2025, the system has been optimized to use a unified embedding approach:

- All embeddings are generated using only the Azure OpenAI embedding model
- Both Azure and Google Gemini models query the same Azure embeddings
- This approach eliminates duplicate embedding generation and storage
- Ensures consistent vector dimensions (1536) across all models
- Improves performance and reduces API calls

## Embedding Status States

Embeddings for a document can be in one of the following states:

1. **not_started**: Embedding generation has not begun
2. **in_progress**: Embeddings are being generated locally
3. **ready_for_chat**: Local embeddings have been created and are available for chat, but cloud upload may still be in progress
4. **completed**: Embeddings have been fully generated and uploaded to Google Cloud Storage

## Decoupled and Parallel Embedding Workflow

### 1. Document Processing & Initial Status

When new documents are uploaded:

- The system validates and processes all documents concurrently
- Initial file metadata entries are created in parallel
- Embedding status is set to `in_progress` for each file
- Multiple files are distributed across parallel embedding creation tasks
- Concurrent existence checks determine which files need new embeddings vs. username updates

### 2. Parallel Local Embedding Generation

The first phase focuses on generating embeddings locally for immediate use with controlled parallelism:

- Multiple documents are processed concurrently using semaphore-limited task pools
- Text is extracted and chunked from each document in parallel
- Embeddings are created using only Azure OpenAI embedding model for all chat models (including Gemini)
- Embeddings are stored locally in ChromaDB under `./chroma_db/{file_id}/`
- Each local `file_info.json` is updated with status `ready_for_chat` as soon as it completes
- Each document becomes available for chat queries as its embeddings finish, without waiting for others

### 3. Cloud Upload (Asynchronous Background Task)

After local embeddings are ready, the system performs these operations in the background:

- Embeddings are uploaded to Google Cloud Storage asynchronously
- File metadata is updated and synchronized with GCS in parallel
- Username lists are preserved and merged between local and cloud versions without blocking
- Non-critical operations like encrypted file uploads happen fully in background
- Once complete, the embedding status is updated to `completed` in both local and cloud metadata

### 4. Status Checking

The system provides two methods to check embedding status:

- `GET /embeddings/status/{file_id}` - Returns the current status with a human-readable message
- `POST /embeddings/check` - Checks if embeddings exist for a specific file and model

For status checking, the system:
1. First checks the local `file_info.json` file
2. If not found locally, checks Google Cloud Storage
3. Returns status and a flag indicating whether chat is available (`can_chat` is true when status is either `ready_for_chat` or `completed`)

## Key Implementation Components

### ParallelEmbeddingCreator Class

- `create_embeddings_parallel`: Creates embeddings for multiple files concurrently with controlled parallelism
- `_create_embedding_task`: Individual task for creating embeddings for a single file

### EmbeddingHandler Class

- `create_and_upload_embeddings`: Creates embeddings locally and schedules background upload
- `_upload_embeddings_to_gcs_background`: Asynchronous task for cloud storage operations
- `_update_embedding_status_to_completed`: Updates status after successful upload
- `check_embeddings_exist`: Async function to check if embeddings exist for a specific file and model (checks local first)
- `embeddings_exist`: Comprehensive check for both local and cloud embedding existence

### GCSHandler Class

- `update_file_info`: Updates file metadata both locally and on GCS
- `get_file_info`: Retrieves file information with fallback logic
- `update_username_list`: Merges username lists preserving all entries

### API Endpoints

- `/embeddings/status/{file_id}`: GET endpoint for checking embedding status
- `/embeddings/check`: POST endpoint for checking if embeddings exist for a specific file and model

## Frontend Integration

Frontend applications should:

1. Use the status endpoint to poll for embedding availability
2. Enable chat functionality once status is `ready_for_chat` or `completed`
3. Provide appropriate UI feedback based on the embedding status

## Benefits

This parallel and decoupled approach provides several benefits:

1. **Maximum Concurrency**: Multiple files are processed in parallel at every stage
2. **Improved User Experience**: Users can start chatting as soon as local embeddings are ready
3. **Reduced Wait Time**: No need to wait for cloud upload completion or for all files to complete
4. **Fault Tolerance**: Even if cloud upload fails, chat functionality is still available
5. **Scalability**: Multi-worker server setup with controlled concurrency allows handling of many simultaneous uploads
6. **Resource Efficiency**: Asynchronous operations reduce API response times
7. **Optimized Critical Path**: Only embedding creation remains on critical path; all other operations run in background

## Error Handling

The system includes robust error handling for various scenarios:

- Failed embedding generation triggers cleanup of partial artifacts
- Background upload failures are logged but don't affect chat availability
- Status checking handles missing files and corrupted metadata gracefully
- Parallel task failures are isolated and don't affect other files in the batch
- Automatic retries for transient errors in embedding generation
- Resource cleanup happens asynchronously to avoid blocking

## Multi-Worker Configuration

The system supports running with multiple worker processes:

- `start_multi_workers.sh` script launches the app with configurable worker count
- Each worker processes requests independently for true parallelism
- Workers share the same file system for consistent access to embeddings
- AsyncIO concurrency is used within each worker for additional parallelism
