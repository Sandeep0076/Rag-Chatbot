# Chat with Multiple Files

## Overview

The multi-file chat feature allows users to interact with and query information from multiple documents simultaneously. Instead of uploading and querying individual files separately, users can select multiple files and ask questions that span across all the selected documents. This is particularly useful when information is spread across different documents or when comparing content from multiple sources.

## Technical Architecture

The multi-file chat feature extends the RAG (Retrieval Augmented Generation) system to work with multiple document sources instead of a single document. The system architecture maintains the core RAG components but adds functionality to handle multiple document collections and merge context from different sources.

## Workflow Pipeline

### File Upload Process

1. **Individual File Upload**
   - Each file is uploaded individually through the `/file/upload` endpoint
   - Files can be PDFs, images, or tabular data (CSV/Excel)
   - Files can also be sourced from URLs using the `urls` parameter
   - **Multiple URL Processing**: URLs can be provided in comma-separated or newline-separated format for batch processing
   - Each file receives a unique `file_id`

2. **Parallel Embedding Generation**
   - Multiple files are processed concurrently using parallel embedding creation
   - The system automatically detects which files need embeddings vs. which need username updates
   - **Legacy Embedding Migration**: Automatically detects and migrates legacy embeddings when files have mixed embedding types
   - **Migration Context Management**: Maintains migration context across the processing pipeline for consistency
   - For each file, embeddings are created using only Azure OpenAI embedding model (1536 dimensions)
   - These unified embeddings are used by all models (including Gemini)
   - Embeddings are stored in ChromaDB with collection path: `./chroma_db/{file_id}/azure`
   - Concurrency is controlled via semaphores to prevent resource exhaustion
   - Each file's metadata is stored in a `file_info.json` file containing:
     - Original filename
     - Upload timestamp
     - File type
     - Username(s) associated with the file (merged across users automatically)
     - Embedding status (in_progress → ready_for_chat → completed)
     - Migration status (migrated flag for tracking migration history)

### Multi-File Chat Workflow

1. **Chat Request Initialization**
   - The client sends a request to the `/chat` endpoint with:
     - `text`: List of user messages/queries
     - `file_ids`: List of file IDs to query against (instead of a single `file_id`)
     - `model_choice`: Selected LLM model (e.g., GPT-3.5, GPT-4, etc.)
     - `user_id`: User identifier

2. **Processing File Information** (`_process_file_info` function)
   - The system identifies this as a multi-file request when `file_ids` is provided
   - For each file ID:
     - Verifies file existence and retrieves file metadata
     - Checks if the file is tabular (has a SQLite database) or non-tabular
     - Builds the `all_file_infos` dictionary mapping each file_id to its metadata
   - Determines if visualization is possible (currently not supported for multi-file)
   - Creates a unique model cache key using all file IDs

3. **File Type Classification**
   - The system classifies files as tabular or non-tabular:
     - Tabular files: Have SQLite databases (CSV, Excel, DB sources)
     - Non-tabular files: PDFs, images, text
   - Special handling for mixed file types (some tabular, some non-tabular)
     - When mixed, defaults to non-tabular mode as it's safer

4. **Model Initialization** (`_initialize_chat_model` function)
   - Based on file types, initializes the appropriate model:
     - For all non-tabular or mixed files: Uses standard `Chatbot` (AzureChatbot)
     - For all tabular files: Uses `TabularDataHandler`
   - Caches the initialized model using the model key for future requests

5. **Context Retrieval**
   - For each file in the multi-file list:
     - Queries the ChromaDB collection specific to that file
     - Retrieves relevant document chunks based on semantic similarity
     - Adds source file information to each chunk: `[Source: {file_id}] {document_text}`
     - Aggregates all relevant chunks into a unified context pool

6. **Creating File Context**
   - Generates a list of all available documents with their original filenames
   - Format: `Available documents: [filename (ID: file_id)]`
   - This helps the LLM understand which documents are available to reference

7. **Query Processing**
   - Combines the unified context, file information, and user query
   - Sends to the selected LLM (Azure OpenAI) with an appropriate system prompt
   - LLM generates a response based on information from all provided documents

8. **Response Formatting**
   - Formats the response based on the chosen output format
   - Returns the final response to the client

## Technical Components

### Key Data Structures

#### `all_file_infos` Dictionary
This critical data structure maps file IDs to their metadata, containing:

```python
{
    "file_id_1": {
        "original_filename": "example1.pdf",
        "upload_time": "2023-07-01T12:00:00",
        "file_type": "pdf",
        "username_list": ["user1"]
        # Other metadata
    },
    "file_id_2": {
        "original_filename": "example2.csv",
        "upload_time": "2023-07-02T14:30:00",
        "file_type": "csv",
        "username_list": ["user1"]
        # Other metadata
    }
    # Additional files...
}
```

### Key Functions

1. **`_process_file_info`** - Processes file metadata for multi-file chat requests:
   - Determines if request is multi-file
   - Verifies embeddings exist for all files before proceeding
   - Builds the `all_file_infos` dictionary
   - Classifies files as tabular or non-tabular
   - Creates a unique model key for caching

2. **`_initialize_chat_model`** - Initializes the appropriate chat model:
   - For mixed or non-tabular files: Uses `Chatbot` class
   - For tabular files: Uses `TabularDataHandler` class
   - Handles special cases for mixed file types

3. **`AzureChatbot.get_answer`** - Core method for multi-file RAG:
   - Queries each file's ChromaDB collection separately
   - Adds source file annotations to document chunks
   - Creates a file context section listing available documents
   - Combines all context and sends to LLM

## Use Cases

1. **Research and Analysis**
   - Compare information across multiple documents
   - Synthesize findings from multiple sources
   - Identify contradictions or confirmations across documents

2. **Knowledge Base Integration**
   - Query across multiple knowledge base documents
   - Find connections between different parts of documentation
   - Get comprehensive answers that span multiple manuals or guides

3. **Data Analysis**
   - Query across both tabular data and descriptive documents
   - Get context from PDFs while analyzing data from spreadsheets

## Current Limitations

1. **Visualization Support**
   - Visualization generation is not currently supported for multi-file chat
   - Visualization features will be disabled when multiple files are selected

2. **Mixed File Type Handling**
   - When mixing tabular and non-tabular files, the system defaults to non-tabular mode
   - This may limit some tabular-specific query capabilities

3. **Context Window Limitations**
   - The total context from all files must fit within the LLM's context window
   - Very large multi-file queries may hit token limits

## Implementation Notes

1. **Parallel Processing**
   - The system uses parallel processing at multiple levels:
     - Multiple files are processed concurrently during upload
     - File existence and embedding checks happen in parallel
     - Embedding creation tasks are distributed across a semaphore-controlled task pool
     - Background tasks handle GCS uploads without blocking chat availability

2. **Memory Efficiency**
   - The system uses reference counting to manage ChromaDB resources
   - Background cleanup processes help manage temporary files and resources
   - Resource cleanup happens asynchronously after embedding operations

3. **File Source Annotation**
   - Document chunks are annotated with source file IDs to help trace information
   - Format: `[Source: {file_id}] {document_text}`

4. **Model Caching**
   - Models are cached using a unique key pattern: `"multi_{sorted_file_ids}_{user_id}_{model_choice}"`
   - This prevents redundant model initialization for the same file combinations

5. **Multi-Worker Server Support**
   - The application can be run with multiple worker processes using `start_multi_workers.sh`
   - Each worker processes requests independently while sharing the file system
   - This provides true parallelism beyond just async concurrency
