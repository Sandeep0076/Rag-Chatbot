# Embedding Generation Workflow

## Overview

The embedding generation process in the RAG PDF API has been optimized to enable immediate chat availability after local embedding creation, with cloud storage operations running asynchronously in the background. This document details the workflow, states, and components involved in this decoupled approach.

## Embedding Status States

Embeddings for a document can be in one of the following states:

1. **not_started**: Embedding generation has not begun
2. **in_progress**: Embeddings are being generated locally
3. **ready_for_chat**: Local embeddings have been created and are available for chat, but cloud upload may still be in progress
4. **completed**: Embeddings have been fully generated and uploaded to Google Cloud Storage

## Decoupled Embedding Workflow

### 1. Document Processing & Initial Status

When a new document is uploaded:

- The system validates and processes the document
- An initial file metadata entry is created
- Embedding status is set to `in_progress`
- A background task is scheduled for embedding generation

### 2. Local Embedding Generation (Synchronous)

The first phase focuses on generating embeddings locally for immediate use:

- Text is extracted and chunked from the document
- Embeddings are created using Azure OpenAI and/or Google Gemini embedding models
- Embeddings are stored locally in ChromaDB under `./chroma_db/{file_id}/`
- The local `file_info.json` is updated with status `ready_for_chat`
- At this point, the document is available for chat queries

### 3. Cloud Upload (Asynchronous Background Task)

After local embeddings are ready, the system performs these operations in the background:

- Embeddings are uploaded to Google Cloud Storage
- File metadata is updated and synchronized with GCS
- Username lists are preserved and merged between local and cloud versions
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

### EmbeddingHandler Class

- `create_and_upload_embeddings`: Creates embeddings locally and schedules background upload
- `_upload_embeddings_to_gcs_background`: Asynchronous task for cloud storage operations
- `_update_embedding_status_to_completed`: Updates status after successful upload
- `check_embeddings_exist`: Checks if embeddings exist for a specific file and model (checks local first)
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

This decoupled approach provides several benefits:

1. **Improved User Experience**: Users can start chatting as soon as local embeddings are ready
2. **Reduced Wait Time**: No need to wait for cloud upload completion
3. **Fault Tolerance**: Even if cloud upload fails, chat functionality is still available
4. **Scalability**: Background tasks don't block the main API operations
5. **Resource Efficiency**: Asynchronous operations reduce API response times

## Error Handling

The system includes robust error handling for various scenarios:

- Failed embedding generation triggers cleanup of partial artifacts
- Background upload failures are logged but don't affect chat availability
- Status checking handles missing files and corrupted metadata gracefully
