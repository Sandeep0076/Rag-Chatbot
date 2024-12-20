# [RAG Application] Rules

Every time you modify or add to the codebase, explicitly follow these rules and document any deviations.

## Project Context
- Enterprise-level RAG (Retrieval Augmented Generation) application
- Supports multiple document types (PDF, Images, CSV/Excel)
- Handles concurrent users and requests
- Uses multiple LLM providers (Azure OpenAI, Google Gemini)
- Memory-efficient with automatic resource cleanup
- Vector storage using ChromaDB for embeddings

## Code Style and Structure
- Write clear, performant Python code with proper type hints
- Use asyncio for concurrent operations where applicable
- Implement singleton patterns for shared resources
- Prefer functional composition over inheritance
- Use descriptive variable names that indicate state/purpose
- Structure repository files as follows:
```
rtl_rag_chatbot_api/
├── chatbot/
    ├── chatbot_creator.py    # Core RAG implementation
    ├── csv_handler.py        # Tabular data processing
    ├── embedding_handler.py   # Embedding management
    ├── file_handler.py       # File operations
    ├── gemini_handler.py     # Gemini model integration
    ├── model_handler.py      # Model management
├── common/
    ├── base_handler.py       # Shared functionality
    ├── chroma_manager.py     # Vector DB management
    ├── cleanup_coordinator.py # Resource cleanup
    ├── models.py            # Pydantic models
├── configs/
    ├── app_config.py        # Configuration management
├── oauth/
    └── get_current_user.py  # Authentication
```

## Chat wit PDF related code files:
-app.py
- chatbot_creator.py
-file_handler.py
-gemini_handler.py
-gcs_handler.py
-model_handler.py
-base_handler.py

## Chat wit CSV,EXcel or DB related code files:
-app.py
-csv_handler.py
-file_handler.py
-prompt_handler.py
-prepare_sqlitedb_from_csv_xlsx.py

## Image related code files:
-app.py
-file_handler.py
-image_reader.py

## Tech Stack
- FastAPI
- ChromaDB
- Azure OpenAI
- Google Gemini
- SQLAlchemy
- APScheduler
- Pydantic
- Uvicorn

## Naming Conventions
- Use snake_case for all Python files and functions
- Use PascalCase for classes
- Use UPPERCASE for constants
- Use descriptive prefixes for handlers (e.g., BaseHandler, FileHandler)
- Include type hints in function signatures

## Type Hints and Validation
- Use Pydantic for data validation and serialization
- Define explicit return types for all functions
- Use type hints for all function parameters
- Implement proper interface definitions using abstract base classes
- Use TypeVar for generic types where appropriate

## Resource Management
- Implement proper cleanup mechanisms for all resources
- Use context managers for database connections
- Implement singleton pattern for shared resources
- Handle background tasks efficiently with BackgroundTasks
- Use connection pooling for databases

## State Management
- Use ChromaDB for vector storage chromadb manager
- Implement proper session management
- Handle concurrent access with thread-safe implementations
- Cleanup stale resources automatically
- Use efficient caching strategies

## Error Handling and Logging
- Implement comprehensive error handling
- Use proper logging levels (DEBUG, INFO, ERROR)
- Include traceback information for debugging
- Handle model-specific errors appropriately
- Implement proper error messages for client feedback

## Performance Optimization
- Use batch processing for embeddings
- Implement connection pooling
- Use async operations for I/O-bound tasks
- Implement proper cleanup of resources
- Monitor memory usage

## Security
- Implement proper authentication
- Sanitize all inputs
- Use environment variables for sensitive data
- Implement proper CORS policies
- Handle file uploads securely

## API Design
- Use proper HTTP methods
- Implement proper response models
- Use appropriate status codes
- Document all endpoints
- Handle rate limiting appropriately

## Memory Management
- Implement automatic cleanup of stale resources
- Use proper garbage collection strategies
- Monitor memory usage
- Clean up temporary files
- Implement resource pooling
- Use cleanup_coordinator.py

## Testing
- Write unit tests for core functionality
- Implement integration tests
- Test concurrent access scenarios
- Test memory leaks
- Test error scenarios

## Documentation
- Maintain clear docstrings
- Document configuration requirements
- Keep API documentation up to date
- Document deployment requirements
- Include performance considerations

## Development Workflow
Commit Message Prefixes:
- "fix:" for bug fixes
- "feat:" for new features
- "perf:" for performance improvements
- "docs:" for documentation
- "refactor:" for code restructuring
- "test:" for adding tests
- "chore:" for maintenance
- use like AIP-228 (feat): add delete endpoint to delete pdf and embeddings

Rules:
- Review code for memory leaks
- Test concurrent access
- Document configuration changes
- Update requirements.txt
- Monitor performance metrics

## Deployment
- Use proper environment variables
- Implement health checks
- Monitor resource usage
- Handle graceful shutdowns
- Implement proper logging

## Monitoring
- Track memory usage
- Monitor concurrent connections
- Log error rates
- Track model performance
- Monitor file processing times
