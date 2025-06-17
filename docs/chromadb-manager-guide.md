# ChromaDB Manager Implementation Guide

## Overview
`ChromaDBManager` is a singleton class responsible for managing ChromaDB client instances within the RAG application. Its primary goal is to provide thread-safe access to ChromaDB, especially crucial during parallel processing tasks like multi-file embedding generation.

To achieve true thread safety and prevent race conditions, `ChromaDBManager` now utilizes `threading.local()` to store ChromaDB clients. This means each thread gets its own dedicated client instance, ensuring that operations in one thread do not interfere with another. A global creation lock (`_creation_lock`) is used solely to serialize the creation of these thread-local instances if one doesn't already exist for the calling thread.

### Key Features:
- **Thread-Local Storage**: Each thread accesses its own `PersistentClient` instance, keyed by `file_id` and `embedding_type`.
- **Creation Lock**: A `threading.Lock()` ensures that the initial creation of a client for a specific key (if not already present in the thread's local storage) is an atomic operation, preventing multiple threads from creating the same client simultaneously.
- **Instance Caching**: Thread-local clients are cached within each thread's `_thread_local_data.clients` dictionary.
- **Centralized Cleanup**: Provides methods to clean up specific or all cached client instances.

## Thread Safety Mechanisms

### 1. `threading.local()`
`threading.local()` provides thread-local data. This means that data stored in a `threading.local()` object is specific to each thread. `ChromaDBManager` uses this to store its dictionary of ChromaDB clients (`_clients`).

```python
# In ChromaDBManager's __init__ or a similar setup point:
self._thread_local_data = threading.local()

# To access or initialize clients for the current thread:
if not hasattr(self._thread_local_data, "clients"):
    self._thread_local_data.clients = {}

# self._thread_local_data.clients is now a dictionary unique to the current thread.
```
This ensures that when multiple threads (e.g., parallel embedding workers) request a ChromaDB client for the same `file_id`, they each interact with a client instance stored in their own thread's local data, preventing cross-thread contamination.

### 2. `threading.Lock()` for Client Creation (`_creation_lock`)
While `threading.local()` isolates client instances *after* they are created, a `threading.Lock` (`_creation_lock`) is used to protect the *creation process* itself. This lock ensures that if multiple threads simultaneously try to create a client for the same `file_id` and `embedding_type` (and it doesn't exist yet in their local storage), only one thread will actually perform the `chromadb.PersistentClient()` call. Others will wait and then use the already created (or retrieved) instance.

```python
# Simplified client retrieval/creation logic:
instance_key = self._get_instance_key(file_id, embedding_type)

if not hasattr(self._thread_local_data, "clients"):
    self._thread_local_data.clients = {}

if instance_key not in self._thread_local_data.clients:
    with self._creation_lock:  # Global lock for creation
        # Double-check after acquiring lock (another thread might have created it)
        if instance_key not in self._thread_local_data.clients:
            db_path = self._get_db_path(file_id, embedding_type)
            os.makedirs(db_path, exist_ok=True)
            client = chromadb.PersistentClient(path=db_path)
            self._thread_local_data.clients[instance_key] = client
# Return self._thread_local_data.clients[instance_key]
```

## Instance Management Flow

```mermaid
graph TD
    A[Thread requests ChromaDB Instance for file_id, type] --> B{Client for key in current thread's local storage?}
    B -->|Yes| C[Return cached client from thread's local storage]
    B -->|No| D{Acquire Global Creation Lock}
    D --> E{Client for key in current thread's local storage? (double-check)}
    E -->|Yes| F[Release Lock]
    F --> C
    E -->|No| G[Create New PersistentClient]
    G --> H[Store client in current thread's local storage]
    H --> F
```

## Key Operations

### `get_client(file_id: str, embedding_type: str)`
Retrieves or creates a thread-local `PersistentClient` for the given `file_id` and `embedding_type`.

### `get_collection(file_id: str, embedding_type: str, collection_name: str, embedding_function: Any = None)`
Gets or creates a collection from the appropriate thread-local client.

```python
# Example Usage:
chroma_manager = ChromaDBManager()

# In Thread 1:
client1 = chroma_manager.get_client(file_id="doc123", embedding_type="azure")
collection1 = chroma_manager.get_collection(
    file_id="doc123",
    embedding_type="azure",
    collection_name="rag_collection_doc123_azure"
)

# In Thread 2 (concurrently):
client2 = chroma_manager.get_client(file_id="doc456", embedding_type="azure")
collection2 = chroma_manager.get_collection(
    file_id="doc456",
    embedding_type="azure",
    collection_name="rag_collection_doc456_azure"
)

# client1 and client2 might point to different physical ChromaDB instances
# if their file_id or embedding_type results in different db_paths.
# Operations on collection1 and collection2 are isolated at the client level per thread.
```

## Instance Cleanup

`ChromaDBManager` provides methods to manage the lifecycle of cached client instances:

-   **`_clear_thread_local_clients()`**: Clears all client instances cached *within the current thread*. This is useful if a thread is being reused and needs a fresh start, though typically Python threads are not reused in this manner for simple task execution.
-   **`cleanup_instance(file_id: str, embedding_type: str)`**: Attempts to remove a specific client instance associated with `file_id` and `embedding_type` from *all threads' local storage*. This is more complex due to the nature of `threading.local()` and might not be fully effective if threads are actively using the client. It iterates through known threads if tracked, or relies on future calls to `get_client` in those threads to not find the cleaned-up key.
-   **`cleanup_all_instances()`**: This is the most comprehensive cleanup. It iterates through all known thread-local data stores (if the manager tracks them, or by other means if available) and clears their `clients` dictionaries. This is typically called at application shutdown or when a global reset of ChromaDB connections is needed.

The previous time-based automatic cleanup of a globally shared cache is no longer the primary mechanism due to the shift to thread-local storage. Cleanup is now more explicit or tied to thread lifecycle.

## Integration Points

-   **Embedding Handlers**: `EmbeddingHandler` and similar components should use the singleton `ChromaDBManager` instance to get clients and collections, ensuring that parallel embedding tasks operate with thread-isolated ChromaDB resources.
-   **Application Lifecycle**: Consider calling `chroma_manager.cleanup_all_instances()` during application shutdown (e.g., in FastAPI's shutdown event) to release resources, although ChromaDB `PersistentClient` might handle its own resource release on garbage collection.

```python
# Example: Passing ChromaDBManager to handlers
# In app.py or main setup:
chroma_manager_singleton = ChromaDBManager()

embedding_handler = EmbeddingHandler(configs, gcs_handler, chroma_manager=chroma_manager_singleton)
azure_chatbot = AzureChatbot(configs, gcs_handler, chroma_manager=chroma_manager_singleton)

# During application shutdown (FastAPI example)
# app.add_event_handler("shutdown", chroma_manager_singleton.cleanup_all_instances)
```
This ensures all parts of the application use the same manager, which then correctly isolates clients per operational thread.
