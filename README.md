# RAG PDF API

## Description
RAG PDF API is a FastAPI-based application that implements a Retrieval-Augmented Generation (RAG) system for processing and querying PDF documents and images. The system uses Azure OpenAI services, Google Cloud Vertex AI (Gemini), and GPT-4 Vision for various AI functionalities, Google Cloud Storage for file management, and Chroma as the vector database.

## Features
- PDF and image preprocessing and embedding generation
- Vector database creation and management using Chroma
- Chat-based querying of processed documents
- Integration with Azure OpenAI for embeddings and language models
- Integration with Google Cloud Vertex AI (Gemini) for advanced language processing
- GPT-4-Omni integration for image analysis
- Google Cloud Storage for file management
- FastAPI backend with Prometheus metrics
- **Advanced Legacy Embedding Migration System** - Automatically migrates legacy embeddings to ensure consistency
- **Parallel File Processing** - Concurrent processing of multiple files with optimized resource usage
- **Smart File Deduplication** - Hash-based file identification to prevent duplicate processing
- **Multi-File Chat Support** - Query multiple documents simultaneously with unified context
- (Optional) Streamlit-based user interface for easy interaction

## Installation

### Prerequisites
- Python 3.10 or higher
- Poetry for dependency management
- Google Cloud SDK
- Azure account with OpenAI services
- Google Cloud account with Vertex AI enabled

### Setup
1. Install Python 3.10 or higher if not already installed.

2. Install Poetry:
   ```
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. Clone the repository:
   ```
   git clone [repository-url]
   cd rtl-rag-chatbot-api
   ```

4. Configure Poetry:
   ```
   poetry config http-basic.python-packages gitlab-ci-token xxx
   ```

5. Install dependencies:
   ```
   poetry install
   ```

6. Lock dependencies:
   ```
   poetry lock
   ```
   Note: You may need to reset/checkout the poetry.lock file again from the repository if there are conflicts.

7. Add Poetry to your PATH:
   ```
   export PATH="$HOME/.local/bin:$PATH"
   ```
   Consider adding this line to your shell configuration file (e.g., .bashrc, .zshrc) for persistence.

8. Set up environment variables (see Environment Variables section)

9. Install pre-commit hooks:
   ```
   pre-commit install
   ```

## Configuration

### Environment Variables
The application requires several environment variables to be set. These are detailed in the `env-variables.md` file. Key variables include:

- Azure OpenAI Configuration
- Google Cloud Storage Configuration
- Google Cloud Vertex AI Configuration
- Application-specific variables (prefix with `RAG_PDF_API__`)
- **Migration System Variables**:
  - `USE_FILE_HASH_DB`: Enable database-based file hash lookup (recommended)
  - `LEGACY_EMBEDDING_TYPE`: Legacy embedding type (default: "azure")
  - `NEW_EMBEDDING_TYPE`: Current embedding type (default: "azure-03-small")

To set these variables, you can either export them in your shell or create a `.env` file in the project root.

The main configuration file is `configs/app_config.py`. You can override these configurations using the environment variables.

## Usage

### Running the Application
To start the FastAPI server:

```
poetry run start
```

The server will start on `http://0.0.0.0:8080`.

To run the Streamlit interface:

```
streamlit run streamlit_app.py
```

To run the Version logger interface:

```
python version_doc/version_logger.py
```
To run unit tests:

```
pytest tests/test_api_unit.py -n 10
```

### API Endpoints

1. **Health Check**: GET `/internal/healthy`
2. **Readiness Check**: GET `/internal/ready`
3. **Application Info**: GET `/info`
4. **File Upload with Migration**: POST `/file/upload` - Now supports automatic legacy embedding migration
5. **Multi-File Chat**: POST `/file/chat` - Enhanced with multi-file support and migration awareness
6. **Get Nearest Neighbors**: POST `/file/neighbors`
7. **Available Models**: GET `/available-models`
8. **Cleanup Files**: POST `/file/cleanup`
9. **Initialize Model**: POST `/model/initialize`
10. **Analyze Image**: POST `/analyze-image`
11. **Image Generation**: POST `/image/generate` and `/image/generate-combined`
12. **Gemini Chat**: POST `/chat/gemini`
13. **Metrics**: GET `/metrics`
14. **Delete Resources**: DELETE `/delete` and `/delete_all_embeddings`
15. **Database Operations**: GET `/test-db-connection`, POST `/insert-file-info`, DELETE `/delete-all-file-info`
16. **File Search**: GET `/find-file-by-name`

## Architecture

The application is structured as follows:

- `app.py`: Main FastAPI application with enhanced migration support
- `streamlit_app.py`: Streamlit user interface
- `chatbot/migration_handler.py`: **NEW** - Advanced migration system for legacy embeddings
- `chatbot/chatbot_creator.py`: Implements the Chatbot class for RAG functionality
- `chatbot/gcs_handler.py`: Handles interactions with Google Cloud Storage
- `chatbot/gemini_handler.py`: Manages Gemini model interactions
- `chatbot/image_reader.py`: Handles image analysis using GPT-4 Omni
- `chatbot/parallel_embedding_creator.py`: **NEW** - Parallel embedding creation for multiple files
- `common/embeddings.py`: Manages the creation of embeddings
- `common/vector_db_creator.py`: Handles the creation and management of the Chroma vector database
- `common/cleanup_coordinator.py`: **ENHANCED** - Intelligent cleanup with migration awareness

## Migration System

### Overview
The new migration system automatically detects and migrates legacy embeddings to ensure consistency across multi-file uploads. It's designed to handle complex scenarios where users upload files with mixed embedding types.

### Key Features

#### 🔄 **Automatic Migration Detection**
- Detects mixed embedding types during multi-file upload
- No manual intervention required
- Works seamlessly with existing upload flow

#### 🚀 **Parallel Processing**
- **Concurrent Migration**: Multiple files processed simultaneously
- **Resource Optimization**: Efficient use of system resources
- **Background Tasks**: Non-blocking operations for better user experience

#### 🎯 **Smart Decision Making**
- **Hash-based Detection**: Files identified by MD5 hash for accurate deduplication
- **Context-Aware Processing**: Different handling for migration vs. new files
- **Fallback Mechanisms**: Graceful handling of edge cases

#### 📊 **Migration Scenarios**

| Scenario | File Types | Action |
|----------|------------|---------|
| **All New Files** | 3 new PDFs | Process normally with `azure-03-small` |
| **All Current** | 2 `azure-03-small` + 1 new | Skip existing, process new file |
| **Mixed Types** | 1 `azure` + 1 `azure-03-small` + 1 new | **Migrate legacy file**, skip current, process new |
| **All Legacy** | 3 `azure` files | Skip migration (all same type) |

### Migration Flow

```python
# 1. Upload Detection
if len(all_files) > 1:
    # Trigger migration analysis
    is_multi_file_scenario, plan = await plan_upload_with_migration(
        all_files, parsed_existing_file_ids, configs
    )

# 2. Migration Planning
# System analyzes files and creates migration plan
# - Identifies files needing migration
# - Categorizes existing vs. new files
# - Determines optimal processing order

# 3. Parallel Execution
# - Legacy files migrated concurrently
# - New files processed in parallel
# - All operations optimized for speed

# 4. Context Preservation
# - Usernames preserved across migration
# - File IDs maintained for consistency
# - Metadata enhanced with migration flags
```

### Configuration

```bash
# Enable database lookup for file hashes (recommended)
USE_FILE_HASH_DB=true

# Legacy and current embedding types
LEGACY_EMBEDDING_TYPE=azure
NEW_EMBEDDING_TYPE=azure-03-small
```

## Multi-File Support

### Enhanced Chat Capabilities
- **Unified Context**: Query multiple documents simultaneously
- **Smart File Classification**: Automatic detection of tabular vs. document files
- **Mixed File Types**: Handle combinations of PDFs, images, and tabular data
- **Session Management**: Maintain context across multiple file types

### File Processing Modes
1. **Single File**: Traditional single-file processing
2. **Multi-File**: Enhanced processing with migration awareness
3. **Mixed Types**: Intelligent handling of different file formats
4. **Migration Mode**: Automatic legacy embedding updates

## Performance Optimizations

### Parallel Processing
- **Concurrent Embedding Creation**: Multiple files processed simultaneously
- **Background Uploads**: Non-blocking GCS operations
- **Resource Pooling**: Efficient use of system resources
- **Semaphore Control**: Configurable concurrency limits

### Memory Management
- **Streaming Operations**: Large files processed in chunks
- **Efficient Cleanup**: Automatic resource cleanup after processing
- **Hash-based Deduplication**: Prevents duplicate file processing
- **Optimized Storage**: Smart use of temporary and permanent storage

## Development

### Running Tests
To run the test suite:

```
make test
```

### Migration System Testing
```python
# Test migration plan generation
async def test_migration_planning():
    handler = MigrationHandler(configs)
    plan = await handler.plan_upload_with_migration(files, existing_ids, configs)
    assert plan["action"] in ["normal", "fallthrough", "error"]

# Test parallel processing
async def test_parallel_migration():
    # Verify concurrent file processing
    # Check resource usage optimization
    # Validate migration consistency
```

## Deployment

The project includes Helm charts for Kubernetes deployment. Deployment configurations can be found in the `helm` directory.

### Migration System Deployment
- **Database Requirements**: PostgreSQL recommended for file hash storage
- **Resource Allocation**: Ensure sufficient memory for parallel processing
- **Monitoring**: Track migration success rates and performance metrics

## Monitoring and Metrics

### Key Metrics
- **Migration Success Rate**: Percentage of successful migrations
- **Processing Time**: Time to complete multi-file uploads
- **Resource Usage**: Memory and CPU utilization during parallel processing
- **Error Rates**: Migration and processing failure statistics

### Logging
```python
# Migration detection
logging.info("Mixed embedding types detected - migrating legacy embeddings")

# Parallel processing
logging.info(f"Starting parallel processing for {len(files)} files")

# Migration completion
logging.info(f"Successfully migrated {migrated_count} files")
```

## Troubleshooting

### Common Issues

1. **Migration Not Triggered**
   - Check if files have different embedding types
   - Verify hash calculation is working correctly
   - Check database/GCS lookup configuration

2. **Performance Issues**
   - Monitor resource usage during parallel processing
   - Check semaphore limits and adjust if needed
   - Verify background task configuration

3. **File Processing Errors**
   - Check file format compatibility
   - Verify embedding model availability
   - Review error logs for specific failure reasons

### Debug Commands
```python
# Check migration plan
plan = await plan_upload_with_migration(files, existing_ids, configs)
print(f"Action: {plan['action']}")
print(f"Migration context: {plan.get('migration_context')}")

# Monitor parallel processing
logging.info(f"Active tasks: {len(asyncio.all_tasks())}")
logging.info(f"Semaphore count: {file_processing_semaphore._value}")
```

## Future Enhancements

1. **Batch Migration**: Migrate all legacy files in the system
2. **Migration Rollback**: Ability to revert migrations if needed
3. **Progress Tracking**: Real-time progress updates for large migrations
4. **Migration Scheduling**: Background migration of legacy files
5. **Analytics Dashboard**: Detailed migration statistics and reporting
6. **Advanced File Types**: Support for additional document formats
7. **Performance Tuning**: Dynamic concurrency adjustment based on system load

## Contact

[Sandeep Pathania] - [AI-Products Team]
