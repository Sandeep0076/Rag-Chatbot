# Chat with PDF Documentation

## Overview
The Chat with PDF functionality is an enterprise-level Retrieval Augmented Generation (RAG) system that enables users to have interactive conversations with PDF documents. The system uses Azure OpenAI and Google Gemini models with ChromaDB for vector storage.

## Step-by-Step Workflow

### Step 1: User Uploads a PDF

When a user uploads a PDF, the system follows these steps:

1. **Initial File Check**
   - Validates if it's a valid PDF file
   - Calculates file hash (MD5) of the uploaded PDF
   - Sanitizes the filename for security

2. **Hash Check for Existing File**
   - System searches for the calculated hash in the database
   - Two possible paths:

     A) **If Hash Found (Existing PDF)**:
     - Retrieves existing file metadata
     - Downloads existing embeddings from Google Cloud Storage
     - Loads embeddings into ChromaDB
     - Skips to Step 4 (Chat Process)

     B) **If Hash Not Found (New PDF)**:
     - Generates unique file_id
     - Continues to Step 2

### Step 2: Processing New PDF

For new PDFs only:

1. **Storage & Metadata**
   - Uploads PDF file to Google Cloud Storage
   - Creates metadata record with:
     - File hash
     - Original filename
     - Upload timestamp
     - User information

2. **Text Extraction**
   - Processes PDF to extract text
   - Splits text into manageable chunks
   - Prepares chunks for embedding generation

### Step 3: Embedding Generation

For new PDFs only:

1. **Generate Embeddings**
   - Uses Azure OpenAI embedding model
   - Generates vector embeddings for each text chunk
   - Stores embeddings in ChromaDB
   - Uploads embeddings backup to Google Cloud Storage

2. **Status Update**
   - Updates metadata with embedding status
   - Marks document as ready for chat

### Step 4: Chat Process

1. **Chat Initialization**
   - User selects PDF to chat with
   - System loads appropriate embeddings
   - Initializes chosen LLM (Azure GPT or Gemini)
   - Creates new chat session

   **How Embeddings are Loaded During Chat:**

   1. **Initial Request Processing**:
      - The chat endpoint in `app.py` receives a query containing the `file_id` and `model_choice`
      - The system first checks if the file info exists by calling `gcs_handler.get_file_info(query.file_id)`
      - If file info doesn't exist, it checks embeddings status with `gcs_handler.check_embeddings_status(query.file_id)`

   2. **Model Initialization**:
      - The system creates a model key using `f"{query.file_id}_{query.user_id}_{query.model_choice}"`
      - It checks if a model is already initialized for this key in the `initialized_models` dictionary
      - If not, it calls `check_db_existence` for tabular data or `initialize_rag_model` for documents

   3. **Embedding Location Check**:
      - In `initialize_rag_model`, the system first checks for local embeddings in `chroma_path = f"./chroma_db/{query.file_id}"`
      - It determines the embedding type based on model choice (`google` for Gemini models, `azure` for others)
      - It looks for the specific embedding database at `model_path = os.path.join(chroma_path, embedding_type, "chroma.sqlite3")`

   4. **Embedding Download (if needed)**:
      - If local embeddings don't exist, it checks if embeddings are ready in GCS by verifying `file_info.get("embeddings_status") == "completed"`
      - If embeddings are ready in GCS but not locally, it downloads them with `gcs_handler.download_files_from_folder_by_id(query.file_id)`
      - This maintains the same directory structure locally as in GCS

   5. **ChromaDB Collection Initialization**:
      - The appropriate model (Gemini or Azure) is initialized with the file_id, embedding_type, and collection name
      - The model connects to the ChromaDB collection through the `ChromaDBManager` singleton
      - The collection path is constructed as `./chroma_db/{file_id}/{embedding_type}`
      - For user-specific sessions, it uses `{file_id}/{embedding_type}/user_{user_id}` as the instance key

   6. **Embedding Retrieval During Chat**:
      - When the user asks a question, the model calls `query_chroma` in `BaseRAGHandler`
      - This method gets the user's query embedding using the model's `get_embeddings` method
      - It then retrieves the ChromaDB collection using `_get_chroma_collection()`
      - The collection is queried with the embedding to find the most relevant document chunks
      - These chunks are used as context for generating the response

   7. **Optimizations for Performance**:
      - The system supports both "ready_for_chat" and "completed" embedding statuses
      - Local embeddings are prioritized over GCS embeddings for faster access
      - The `ChromaDBManager` singleton ensures efficient reuse of ChromaDB instances
      - User-specific instances are tracked to maintain separate chat contexts

2. **For Each User Query**:
   a. **Query Processing**
      - Sanitizes user input
      - Generates embedding for the query

   b. **Context Retrieval**
      - Searches similar chunks in ChromaDB
      - Retrieves most relevant context
      - Optimizes context length for model

   c. **Answer Generation**
      - Combines query, context, and chat history
      - Sends to LLM for response
      - Returns formatted answer to user

### Step 5: Resource Management

Throughout the process:

1. **Cleanup**
   - Removes temporary files
   - Cleans up unused embeddings
   - Manages chat history

2. **Error Handling**
   - Handles upload failures
   - Manages embedding generation errors
   - Deals with model availability issues
   - Handles token limit exceeded cases

## Limitations

1. **File Limitations**
   - Maximum PDF size: [size limit]
   - Supported PDF versions
   - Text extraction quality dependency

2. **Processing Limitations**
   - Embedding generation time for large files
   - Context window size limits
   - Rate limits for API calls

## Best Practices

1. **PDF Preparation**
   - Use text-based PDFs (not scanned)
   - Keep file size optimized
   - Ensure good text formatting

2. **Querying**
   - Ask specific questions
   - Reference relevant sections
   - Stay within context limits

### 3. Resource Management

1. **Memory Optimization**
   - Implements automatic cleanup of stale resources
   - Uses connection pooling for database operations
   - Manages chat history efficiently

2. **Security Measures**
   - Encrypts sensitive data
   - Implements proper authentication
   - Sanitizes all inputs
   - Uses secure file handling

### 4. Error Handling

The system handles various scenarios:

1. **File-related Errors**
   - Invalid file formats
   - Corrupt PDFs
   - Upload failures
   - Storage issues

2. **Processing Errors**
   - Embedding generation failures
   - Model availability issues
   - Database connection problems

3. **Query Errors**
   - Invalid queries
   - Context retrieval issues
   - Model response failures

## Performance Considerations

1. **Caching**
   - Embeddings are cached for repeated queries
   - Frequently accessed documents are kept in memory
   - Query results are cached where appropriate

2. **Scalability**
   - Handles concurrent users efficiently
   - Implements connection pooling
   - Uses async operations for I/O-bound tasks

3. **Resource Cleanup**
   - Automatic cleanup of temporary files
   - Regular garbage collection
   - Proper resource deallocation

## Limitations

1. **File Size**
   - Maximum PDF 6mb
   - Token limits for context windows
   - Embedding generation time for large documents

2. **Query Processing**
   - Context window limitations
   - Response time variations
   - Model-specific constraints

## Best Practices

1. **Document Preparation**
   - Ensure PDFs are properly formatted
   - Optimize document size
   - Use clear, well-structured content

2. **Query Optimization**
   - Ask clear, specific questions
   - Provide sufficient context
   - Break down complex queries
