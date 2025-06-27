# RTL-RAG-CHABOT-API

## Description
RTL RAG CHABOT API

## Links
This is an FastAPI project.
See https://fastapi.tiangolo.com/

## How to run locally:

1. Clone the repository

2. Create a [personal access token](https://gitlab.com/-/profile/personal_access_tokens).

3. Configure the remote repository
```
poetry config repositories.python-packages https://gitlab.com/api/v4/projects/33281928/packages/pypi/simple/
```

4. Configure your token
```
poetry config http-basic.python-packages <gitlab-token-name> <gitlab-token>
```

5. Install pre-commit
```
pre-commit install
```

6. Install the package:  
```
make install
```

7. Run project locally
```
make serve
```

8. Test project locally
```
make serve # starts application at http://127.0.0.1:8080

make e2e # create request to local api endpoint and prints response
```

9. Run Tests
```
make test
```

## Documentation

### API Usage
- [API Usage Guide](README_API_USAGE.md) - Comprehensive guide for using the API endpoints

### Chat Features
- [Chat with PDF Documents](README_CHAT_WITH_PDF.md) - Guide for chatting with PDF files
- [Chat with Multiple Files](README_CHAT_WITH_MULTIPLE_FILES.md) - Multi-file chat functionality
- [Chat with Tabular Data](README_CHAT_WITH_TABULAR.md) - Working with CSV/Excel files

### Model Integration
- [Gemini 2.5 Models Integration](gemini-2.5-integration.md) - NEW: Guide for using Gemini 2.5 Flash and Pro models

### Technical Workflows
- [Embedding Workflow](embedding-workflow.md) - Document embedding process
- [Async Parallel Embedding](async-parallel-embedding-guide.md) - Parallel processing guide
- [ChromaDB Manager](chromadb-manager-guide.md) - Vector database management
- [CSV Workflow](csv-workflow-readme.md) - Tabular data processing
- [Cleanup Workflow](cleanup-workflow.md) - Resource cleanup procedures

### Image Generation
- [Image Generation API](image_generation_api.md) - API for generating images
- [Image Generation Guide](image_generation.md) - Using DALL-E and Imagen models
