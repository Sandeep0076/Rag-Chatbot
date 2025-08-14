# RAG PDF API - Postman Usage Guide
This guide provides detailed instructions on how to use each endpoint of the RAG PDF API with Postman. The API supports processing and querying various document types including PDFs, images, and tabular data (CSV/Excel) using Retrieval-Augmented Generation (RAG) and SQL querying capabilities.

## Table of Contents

- [Authentication](#authentication)
- [Basic Workflow](#basic-workflow)
- [Endpoints](#endpoints)
  - [File Upload](#file-upload)
  - [Check Embeddings](#check-embeddings)
  - [Check Embedding Status](#check-embedding-status)
  - [Chat with File](#chat-with-file)
  - [Get Available Models](#get-available-models)
  - [Chat with Gemini](#chat-with-gemini)
  - [Get Neighbors](#get-neighbors)
  - [Analyze Image](#analyze-image)
  - [Generate Image](#generate-image)
  - [Generate Combined Images](#generate-combined-images)
  - [Delete Resources](#delete-resources)
  - [Delete All Resources](#delete-all-resources)
  - [Find File by Name](#find-file-by-name)
  - [Manual Cleanup](#manual-cleanup)
- [Session ID Management](#session-id-management)
  - [Session ID Flow](#session-id-flow)
  - [Key Benefits](#key-benefits-of-session-id-system)
  - [Session ID Lifecycle](#session-id-lifecycle)
- [Additional Information](#additional-information)

## Authentication

All endpoints require authentication. The API uses OAuth authentication, which is handled by the `get_current_user` dependency.

In Postman, you need to add an Authorization header to your requests. The specific authentication method depends on your deployment configuration.

## Configuration

### Database Integration
The API supports optional database integration for efficient file hash lookup and metadata tracking:

- **Environment Variable**: `USE_FILE_HASH_DB=true`
- **Purpose**: Enables database storage of file hashes for duplicate detection and faster lookups
- **Benefits**:
  - Prevents duplicate file processing
  - Faster file existence checks
  - Automatic database cleanup when files are deleted
- **Database Table**: `FileInfo` table tracks file_id, file_hash, and creation timestamps
- **Cleanup Behavior**: When enabled, file deletions automatically remove corresponding database records


## Basic Workflow

**Unified Embedding Strategy:** The API now employs a unified embedding strategy. All embeddings are generated using Azure OpenAI models, regardless of the chat model chosen (e.g., GPT series or Gemini). This ensures consistency and efficiency.

The typical workflow for using the API is:

1. Upload a file (PDF, image, CSV, or Excel) using the `/file/upload` endpoint
2. Chat with the content using the `/file/chat` endpoint

## Endpoints

### File Upload

Upload files to create embeddings for subsequent chat queries. The API automatically processes different file types including PDFs, images, CSV files, and SQLite databases.

- **Endpoint URL**: `/file/upload`
- **HTTP Method**: POST
- **Request Headers**:
  - `Authorization`: Your auth token
  - `Content-Type`: `multipart/form-data`
- **Request Parameters**:
  - `file`: Single file upload (optional if using `files` or `existing_file_ids`)
  - `files`: Multiple file upload (optional if using `file` or `existing_file_ids`)
  - `existing_file_ids`: Comma or newline separated list of existing file IDs (optional)
  - `is_image`: Boolean flag for image processing (default: false)
  - `username`: Required username for tracking
  - `urls`: Comma or newline separated URLs for web content processing (optional)
- **Response Format**:
  ```json
  {
    "message": "Files processed successfully",
    "file_ids": ["uuid1", "uuid2", "uuid3"],
    "original_filenames": ["doc1.pdf", "data.csv", "existing_file.pdf"],
    "is_image": false,
    "is_tabular": false,
    "status": "success",
    "multi_file_mode": true,
    "session_id": "uuid-string"
  }
  ```
- **Usage Example**:
  1. Open Postman and create a new request
  2. Set the request method to POST
  3. Enter the URL: `http://your-api-domain/file/upload`
  4. Go to the "Body" tab and select "form-data"
  5. Add the following key-value pairs:
     - Key: `file`, Value: Select a file from your computer (for single file)
     - Key: `files`, Value: Select multiple files from your computer (for multiple files)
     - Key: `existing_file_ids`, Value: `file-id-1, file-id-2, file-id-3` (for existing files)
     - Key: `is_image`, Value: `false` (or `true` if uploading an image)
     - Key: `username`, Value: Enter a username
     - Key: `urls`, Value: Enter URLs separated by commas or newlines (optional)
  6. Click "Send" to upload the file

  **Multiple URL Processing Example**:
  To process multiple URLs simultaneously:
  - Key: `username`, Value: `your-username`
  - Key: `urls`, Value:
    ```
    https://example1.com, https://example2.com, https://example3.com
    ```
    or
    ```
    https://example1.com
    https://example2.com
    https://example3.com
    ```
  - The response will include `file_ids` array with multiple file IDs for multi-file chat

  **Existing File IDs Processing Example**:
  To use existing file IDs with embeddings already created:
  - Key: `username`, Value: `your-username`
  - Key: `existing_file_ids`, Value:
    ```
    uuid-file-1, uuid-file-2, uuid-file-3
    ```
    or
    ```
    uuid-file-1
    uuid-file-2
    uuid-file-3
    ```
  - The system will validate embeddings exist, download them locally if needed, and prepare for chat

  **Mixed Processing Example**:
  To combine new file uploads with existing file IDs:
  - Key: `username`, Value: `your-username`
  - Key: `files`, Value: Select new files to upload
  - Key: `existing_file_ids`, Value: `uuid-file-1, uuid-file-2`
  - Key: `urls`, Value: `https://example.com`
  - All files (new, existing, and URL content) will be processed in parallel

### Check Embeddings

Check if embeddings exist for a specific file and model.

### Check if Embeddings Exist

- **Endpoint URL**: `/embeddings/check`
- **HTTP Method**: POST
- **Request Headers**:
  - `Authorization`: Your auth token
  - `Content-Type`: `application/json`

#### Request Body
```json
{
  "file_ids": ["uuid-string-1", "uuid-string-2", "uuid-string-3"],
  "model_choice": "gpt_4o_mini"
}
```

**Note**: For a single file, use a list with one item: `["uuid-string"]`

#### Response Format
```json
{
  "results": [
    {
      "embeddings_exist": true,
      "model_type": "azure",
      "file_id": "uuid-string-1",
      "status": "ready_for_chat"
    },
    {
      "embeddings_exist": false,
      "model_type": "azure",
      "file_id": "uuid-string-2",
      "status": "not_started"
    },
    {
      "embeddings_exist": true,
      "model_type": "azure",
      "file_id": "uuid-string-3",
      "status": "completed"
    }
  ],
  "summary": {
    "total_files": 3,
    "files_with_embeddings": 2,
    "files_missing_embeddings": 1,
    "all_files_ready": false,
    "model_choice": "gpt_4o_mini"
  }
}
```

#### Usage Example
1. Create a new request in Postman
2. Set the request method to POST
3. Enter the URL: `http://your-api-domain/embeddings/check`
4. Go to the "Body" tab and select "raw" and "JSON"
5. Enter the JSON request body with `file_ids` (list) and `model_choice`
6. Click "Send" to check if embeddings exist for your files

**Examples:**
- Single file: `{"file_ids": ["uuid1"], "model_choice": "gpt_4o_mini"}`
- Multiple files: `{"file_ids": ["uuid1", "uuid2", "uuid3"], "model_choice": "gpt_4o_mini"}`

### Check Embedding Status

Get the current status of embeddings for a specific file. The frontend can poll this endpoint to know when embeddings are ready for chat.

- **Endpoint URL**: `/embeddings/status/{file_id}`
- **HTTP Method**: GET
- **Request Headers**:
  - `Authorization`: Your auth token
- **Path Parameters**:
  - `file_id`: The ID of the file to check
- **Response Format**:
  ```json
  {
    "status": "ready_for_chat",
    "can_chat": true,
    "file_id": "uuid-string",
    "message": "Embeddings are ready for chat. Background upload still in progress."
  }
  ```
- **Possible Status Values**:
  - `not_started`: Embeddings generation has not started
  - `in_progress`: Embeddings are being generated
  - `ready_for_chat`: Local embeddings are ready for chat, but cloud upload may still be in progress
  - `completed`: Embeddings are fully generated and uploaded to cloud storage
- **Usage Example**:
  1. Create a new request in Postman
  2. Set the request method to GET
  3. Enter the URL: `http://your-api-domain/embeddings/status/{your-file-id}`
  4. Click "Send" to check the embedding status

### Chat with File

Process chat queries against document content using specified language models.

- **Endpoint URL**: `/file/chat`
- **HTTP Method**: POST
- **Request Headers**:
  - `Authorization`: Your auth token
  - `Content-Type`: `application/json`
- **Request Body**:
  ```json
  {
    "text": ["What is this document about?"],
    "file_id": "uuid-string",
    "file_ids": ["uuid-string1", "uuid-string2"],
    "model_choice": "gpt_4o_mini",
    "user_id": "user123",
    "session_id": "uuid-string",
    "temperature": 0.7,
    "generate_visualization": false
  }
  ```

- **Request Parameters**:
  - `text` (required): Array of strings containing the conversation history with the current question as the last element
  - `file_id` (optional): Single file ID for single-file chat
  - `file_ids` (optional): Array of file IDs for multi-file chat
  - `model_choice` (required): The language model to use (e.g., "gpt_4o_mini", "gemini-2.5-flash", "gemini-2.5-pro")
  - `user_id` (required): Unique identifier for the user
  - `session_id` (required): Session identifier for tracking conversation context
  - `temperature` (optional): Controls randomness in model responses (range: 0.0 - 2.0)
    - **Default values**:
      - OpenAI models (GPT series): `0.5` (more focused, coherent responses)
      - Gemini models: `0.8` (more creative, diverse responses)
    - **Lower values (0.0-0.3)**: More deterministic, focused responses
    - **Medium values (0.4-0.7)**: Balanced creativity and coherence
    - **Higher values (0.8-2.0)**: More creative, diverse, but potentially less coherent responses
  - `generate_visualization` (optional): Whether to generate data visualizations for applicable queries
- **Response Format**:
  ```json
  {
    "response": "This document is about...",
    "is_table": false,
    "sources": ["uuid-string"]
  }
  ```
  For tabular data, the response may include formatted table data:
  ```json
  {
    "response": "| Column1 | Column2 |\n|---------|---------|...",
    "is_table": true,
    "headers": ["Column1", "Column2", ...],
    "rows": [["value1", "value2"], ...]
  }
  ```
- **Usage Examples**:

  **Basic Example**:
  1. Create a new request in Postman
  2. Set the request method to POST
  3. Enter the URL: `http://your-api-domain/file/chat`
  4. Go to the "Body" tab and select "raw" and "JSON"
  5. Enter the JSON request body with your query
  6. Click "Send" to chat with the document

  **Temperature Usage Examples**:

  *For precise, factual responses (technical documents, legal texts):*
  ```json
  {
    "text": ["What are the exact specifications mentioned in this document?"],
    "file_id": "uuid-string",
    "model_choice": "gpt_4o_mini",
    "user_id": "user123",
    "temperature": 0.2
  }
  ```

  *For balanced responses (general content, summaries):*
  ```json
  {
    "text": ["Summarize the main points of this document"],
    "file_id": "uuid-string",
    "model_choice": "gpt_4o_mini",
    "user_id": "user123",
    "temperature": 0.5
  }
  ```

  *For creative analysis (brainstorming, interpretations):*
  ```json
  {
    "text": ["What are some creative applications of the concepts in this document?"],
    "file_id": "uuid-string",
    "model_choice": "gemini-2.5-flash",
    "user_id": "user123",
    "temperature": 1.0
  }
  ```

### Get Available Models

Retrieve a list of available models including Azure LLM models, Gemini models, and image generation models.

- **Endpoint URL**: `/available-models`
- **HTTP Method**: GET
- **Request Headers**:
  - `Authorization`: Your auth token
- **Response Format**:
  ```json
  {
    "models": ["gpt_4o_mini", "gpt_4o", "gemini-2.5-flash", "gemini-2.5-pro", "dall-e-3", "imagen-3.0", "Dalle + Imagen"],
    "model_types": {
      "text": ["gpt_4o_mini", "gpt_4o", "gemini-2.5-flash", "gemini-2.5-pro"],
      "image": ["dall-e-3", "imagen-3.0", "Dalle + Imagen"]
    }
  }
  ```
- **Usage Example**:
  1. Create a new request in Postman
  2. Set the request method to GET
  3. Enter the URL: `http://your-api-domain/available-models`
  4. Click "Send" to get the list of available models

### Chat with Gemini

Chat with Gemini models (Flash, Pro, 2.5 Flash, or 2.5 Pro) without RAG or file context.

- **Endpoint URL**: `/chat/gemini`
- **HTTP Method**: POST
- **Request Headers**:
  - `Authorization`: Your auth token
  - `Content-Type`: `application/json`
- **Request Body**:
  ```json
  {
    "model": "gemini-2.5-pro",
    "message": "Tell me about artificial intelligence"
  }
  ```
- **Response Format**: Streaming text response
- **Usage Example**:
  1. Create a new request in Postman
  2. Set the request method to POST
  3. Enter the URL: `http://your-api-domain/chat/gemini`
  4. Go to the "Body" tab and select "raw" and "JSON"
  5. Enter the JSON request body with your message
  6. Click "Send" to chat with Gemini

### Get Neighbors

Retrieve nearest neighbors for a given text query from a specific file.

- **Endpoint URL**: `/file/neighbors`
- **HTTP Method**: POST
- **Request Headers**:
  - `Authorization`: Your auth token
  - `Content-Type`: `application/json`
- **Request Body**:
  ```json
  {
    "text": "What is machine learning?",
    "file_id": "uuid-string",
    "n_neighbors": 5
  }
  ```
- **Response Format**:
  ```json
  {
    "neighbors": [
      "Machine learning is a subset of artificial intelligence...",
      "Deep learning algorithms use neural networks...",
      "..."
    ]
  }
  ```
- **Usage Example**:
  1. Create a new request in Postman
  2. Set the request method to POST
  3. Enter the URL: `http://your-api-domain/file/neighbors`
  4. Go to the "Body" tab and select "raw" and "JSON"
  5. Enter the JSON request body with your query
  6. Click "Send" to get nearest neighbors

### Analyze Image

Analyze an uploaded image file and get comprehensive analysis results.

- **Endpoint URL**: `/analyze-image`
- **HTTP Method**: POST
- **Request Headers**:
  - `Authorization`: Your auth token
- **Request Body**:
  - Form data:
    - `file`: The image file to analyze (File)
- **Response Format**:
  ```json
  {
    "message": "Image analysis completed successfully",
    "result_file": "image_analysis_uuid.json",
    "analysis": {
      "gpt4_analysis": "Detailed analysis of the image content..."
    }
  }
  ```
- **Usage Example**:
  1. Create a new request in Postman
  2. Set the request method to POST
  3. Enter the URL: `http://your-api-domain/analyze-image`
  4. Go to the "Body" tab and select "form-data"
  5. Add key: `file`, Value: Select an image file from your computer
  6. Click "Send" to analyze the image

### Generate Image

Generate an image based on a text prompt using DALL-E 3 or Imagen models.

- **Endpoint URL**: `/image/generate`
- **HTTP Method**: POST
- **Request Headers**:
  - `Authorization`: Your auth token
  - `Content-Type`: `application/json`
- **Request Body**:
  ```json
  {
    "prompt": "A futuristic city with flying cars",
    "size": "1024x1024",
    "n": 1,
    "model_choice": "dall-e-3"
  }
  ```
- **Response Format**:
  ```json
  {
    "success": true,
    "image_url": "https://example.com/generated-image.png",
    "prompt": "A futuristic city with flying cars",
    "model": "dall-e-3",
    "size": "1024x1024"
  }
  ```
- **Usage Example**:
  1. Create a new request in Postman
  2. Set the request method to POST
  3. Enter the URL: `http://your-api-domain/image/generate`
  4. Go to the "Body" tab and select "raw" and "JSON"
  5. Enter the JSON request body with your prompt
  6. Click "Send" to generate an image

### Generate Combined Images

Generate images using both DALL-E and Imagen models concurrently with the same prompt.

- **Endpoint URL**: `/image/generate-combined`
- **HTTP Method**: POST
- **Request Headers**:
  - `Authorization`: Your auth token
  - `Content-Type`: `application/json`
- **Request Body**:
  ```json
  {
    "prompt": "A beautiful sunset over mountains",
    "size": "1024x1024",
    "n": 1
  }
  ```
- **Response Format**:
  ```json
  {
    "success": true,
    "dalle_result": {
      "success": true,
      "image_url": "https://example.com/dalle-image.png",
      "model": "dall-e-3"
    },
    "imagen_result": {
      "success": true,
      "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
      "model": "imagen-3.0"
    },
    "prompt": "A beautiful sunset over mountains",
    "models": ["dall-e-3", "imagen-3.0"]
  }
  ```
- **Usage Example**:
  1. Create a new request in Postman
  2. Set the request method to POST
  3. Enter the URL: `http://your-api-domain/image/generate-combined`
  4. Go to the "Body" tab and select "raw" and "JSON"
  5. Enter the JSON request body with your prompt
  6. Click "Send" to generate images with both models

### Delete Resources

Delete ChromaDB embeddings and associated resources for one or multiple files based on username. When `use_file_hash_db` is enabled, also removes corresponding database records.

- **Endpoint URL**: `/delete`
- **HTTP Method**: DELETE
- **Request Headers**:
  - `Authorization`: Your auth token
  - `Content-Type`: `application/json`
- **Request Body**:
  ```json
  {
    "file_ids": "uuid-string",
    "include_gcs": false,
    "username": "user123"
  }
  ```
  or for multiple files:
  ```json
  {
    "file_ids": ["uuid-string1", "uuid-string2"],
    "include_gcs": false,
    "username": "user123"
  }
  ```
- **Response Format**:
  For a single file:
  ```json
  {
    "message": "Username removed from file info"
  }
  ```
  For multiple files:
  ```json
  {
    "results": {
      "uuid-string1": "Success: Username removed from file info",
      "uuid-string2": "Success: Embeddings deleted"
    }
  }
  ```
- **Usage Example**:
  1. Create a new request in Postman
  2. Set the request method to DELETE
  3. Enter the URL: `http://your-api-domain/delete`
  4. Go to the "Body" tab and select "raw" and "JSON"
  5. Enter the JSON request body with the file_ids
  6. Click "Send" to delete resources

### Delete All Resources

Delete ChromaDB embeddings and associated resources for one or multiple files (General method). When `use_file_hash_db` is enabled, also removes corresponding database records automatically.

- **Endpoint URL**: `/delete_all_embeddings`
- **HTTP Method**: DELETE
- **Request Headers**:
  - `Authorization`: Your auth token
  - `Content-Type`: `application/json`
- **Request Body**:
  ```json
  {
    "file_ids": "uuid-string",
    "include_gcs": false
  }
  ```
  or for multiple files:
  ```json
  {
    "file_ids": ["uuid-string1", "uuid-string2"],
    "include_gcs": false
  }
  ```
- **Response Format**:
  For a single file:
  ```json
  {
    "message": "ChromaDB embeddings for file_id uuid-string have been deleted successfully"
  }
  ```
  For multiple files:
  ```json
  {
    "results": {
      "uuid-string1": "Success",
      "uuid-string2": "Success"
    }
  }
  ```
- **Usage Example**:
  1. Create a new request in Postman
  2. Set the request method to DELETE
  3. Enter the URL: `http://your-api-domain/delete_all_embeddings`
  4. Go to the "Body" tab and select "raw" and "JSON"
  5. Enter the JSON request body with the file_ids
  6. Click "Send" to delete all resources

### Find File by Name

Find a file_id by searching through all file_info.json files for a matching original_filename.

- **Endpoint URL**: `/find-file-by-name`
- **HTTP Method**: GET
- **Request Headers**:
  - `Authorization`: Your auth token
- **Request Parameters**:
  - `filename`: The original filename to search for
- **Response Format**:
  ```json
  {
    "file_id": "uuid-string",
    "found": true
  }
  ```
- **Usage Example**:
  1. Create a new request in Postman
  2. Set the request method to GET
  3. Enter the URL: `http://your-api-domain/find-file-by-name?filename=example.pdf`
  4. Click "Send" to find the file by name

### Manual Cleanup

Manually trigger cleanup of resources.

- **Endpoint URL**: `/file/cleanup`
- **HTTP Method**: POST
- **Request Headers**:
  - `Authorization`: Your auth token
  - `Content-Type`: `application/json`
- **Request Body**:
  ```json
  {
    "is_manual": true
  }
  ```
- **Response Format**:
  ```json
  {
    "status": "Cleanup completed successfully"
  }
  ```
- **Usage Example**:
  1. Create a new request in Postman
  2. Set the request method to POST
  3. Enter the URL: `http://your-api-domain/file/cleanup`
  4. Go to the "Body" tab and select "raw" and "JSON"
  5. Enter the JSON request body
  6. Click "Send" to trigger manual cleanup

## Session ID Management

The RAG PDF API uses a session-based approach to manage file uploads and chat contexts. Each upload operation generates a unique `session_id` that groups related files together and provides isolation between different upload sessions.

### Session ID Flow

```
1. FILE UPLOAD PROCESS
   ┌─────────────────────────────────────────────────────────────┐
   │ User uploads file(s) → /file/upload                         │
   │ ├── Backend generates session_id = uuid.uuid4()             │
   │ ├── Files processed and stored with session_id              │
   │ └── Response includes session_id in JSON response           │
   └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
2. FRONTEND SESSION STORAGE
   ┌─────────────────────────────────────────────────────────────┐
   │ Frontend receives session_id from upload response           │
   │ ├── Stores session_id in application state                  │
   │ ├── Associates all uploaded files with this session_id      │
   │ ├── Resets file_ids list for clean session state           │
   │ └── Tracks files by session for proper grouping            │
   └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
3. CHAT REQUEST PROCESS
   ┌─────────────────────────────────────────────────────────────┐
   │ User sends chat message → /file/chat                        │
   │ ├── Frontend includes session_id in chat payload            │
   │ ├── Backend logs session_id for request tracking           │
   │ ├── Files are processed within session context             │
   │ └── Response maintains session isolation                    │
   └─────────────────────────────────────────────────────────────┘
```

### Key Benefits of Session ID System

1. **Session Isolation**
   - Different upload sessions don't interfere with each other
   - Clean separation between different conversation contexts
   - Prevents file mix-ups in concurrent usage scenarios

2. **File Grouping and Context**
   - Files uploaded together are logically grouped as a unit
   - Maintains proper context for multi-file conversations
   - Enables coherent multi-document chat experiences

3. **Request Tracking and Debugging**
   - Easy to track which files belong to which conversation
   - Better logging and debugging capabilities for troubleshooting
   - Clear audit trail for file operations and chat requests

4. **Multi-file Support**
   - Enables proper multi-file chat functionality
   - Supports batch processing of related documents
   - Maintains file relationships across API calls

5. **State Management**
   - Clean session boundaries for frontend applications
   - Proper cleanup when starting new conversations
   - Consistent behavior across different client implementations

### Session ID Lifecycle

The session ID follows a well-defined lifecycle from creation to cleanup:

#### 1. **Creation Phase**
- **Trigger**: File upload operation (`/file/upload` endpoint)
- **Generation**: `session_id = str(uuid.uuid4())` - Creates unique UUID
- **Scope**: Applies to all files in the current upload batch
- **Response**: Included in upload response JSON

#### 2. **Storage and Association Phase**
- **Frontend Storage**: Session ID stored in application state
- **File Association**: All uploaded files tagged with session_id
- **Context Creation**: Session becomes the active context for chat operations
- **State Reset**: Previous session data cleared to maintain isolation

#### 3. **Active Usage Phase**
- **Chat Integration**: Session ID included in all chat requests
- **Context Maintenance**: Backend processes requests within session scope
- **File Resolution**: System resolves file references within session context
- **Logging**: Session ID used for request tracking and debugging

#### 4. **Session Transition**
- **New Upload Trigger**: New file upload creates new session
- **State Reset**: Previous session_id replaced with new one
- **File List Reset**: file_ids list cleared and repopulated
- **Clean Transition**: No overlap between old and new sessions

#### 5. **Cleanup and Termination**
- **Manual Reset**: User initiates "New Chat" or similar action
- **Automatic Reset**: New upload automatically terminates previous session
- **Resource Cleanup**: Associated temporary data and state cleared
- **Session End**: Session ID becomes inactive and is no longer used

#### Session ID Data Flow

```json
// Upload Response Example
{
  "file_ids": ["uuid-1", "uuid-2", "uuid-3"],
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "multi_file_mode": true,
  "status": "success"
}

// Chat Request Example
{
  "text": ["What are the main topics in these documents?"],
  "file_ids": ["uuid-1", "uuid-2", "uuid-3"],
  "model_choice": "gpt_4o_mini",
  "user_id": "user123",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

This session-based architecture ensures reliable, isolated, and trackable interactions between clients and the RAG PDF API, providing a robust foundation for both single-file and multi-file document processing workflows.

## Additional Information

### Setting Up Environment Variables in Postman

To make your requests more manageable, you can set up environment variables in Postman:

1. Click on the "Environment" dropdown in the top right corner
2. Click "New" to create a new environment
3. Add variables such as:
   - `base_url`: Your API base URL
   - `auth_token`: Your authentication token
4. Use these variables in your requests like `{{base_url}}/file/upload` and `{{auth_token}}` in the Authorization header

### Error Handling

The API returns appropriate HTTP status codes for different error scenarios:
- 400: Bad Request (invalid input)
- 404: Not Found (resource not found)
- 500: Internal Server Error (server-side error)

Error responses typically include a `detail` field with more information about the error.

### Multi-File Support

The API supports uploading and chatting with multiple files:
- Use the `files` parameter in `/file/upload` to upload multiple files at once
- Use the `file_ids` parameter in `/file/chat` to chat with multiple files
- The `session_id` field helps track related files in a session

#### Advanced Multi-File Features

**Legacy Embedding Migration**: The API automatically detects and migrates legacy embeddings when uploading files with mixed embedding types:
- **Automatic Detection**: System identifies files with different embedding models
- **Parallel Migration**: Legacy files migrated concurrently for optimal performance
- **Data Preservation**: All usernames and metadata preserved during migration
- **Seamless Integration**: Migration happens automatically without user intervention

**Parallel Processing**: Multi-file uploads are processed concurrently:
- **Concurrent Embedding Creation**: Multiple files processed simultaneously
- **Resource Optimization**: Efficient use of system resources with configurable limits
- **Background Operations**: Non-blocking operations for better user experience
- **Scalability**: Designed to handle hundreds of concurrent users efficiently

### Image Generation Models

The API supports multiple image generation models:
- **DALL-E 3**: OpenAI's image generation model
- **Imagen**: Google's Vertex AI image generation model
- **Combined**: Generate images with both models simultaneously for comparison

### Supported File Types

- **Documents**: PDF, TXT, DOC, DOCX
- **Images**: JPG, JPEG, PNG, GIF, BMP
- **Tabular Data**: CSV, XLSX, XLS, DB, SQLite
- **URLs**: Extract content from web pages

### Session ID Flow Diagram
