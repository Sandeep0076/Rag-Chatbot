# Quick Test Reference Guide

## 🚀 Quick Start

### 1. Setup Environment
```bash
cd /Users/pathania/Rag/rag-pdf-api
poetry shell
```

### 2. Set Environment Variables
```bash
# Copy and paste this entire block:
export TITLE="RAG PDF API Test" && \
export DESCRIPTION="Testing Multi-File Upload" && \
export GCP_PROJECT="test-project" && \
export BUCKET_NAME="test-bucket" && \
export SYSTEM_PROMPT_PLAIN_LLM="Test prompt" && \
export SYSTEM_PROMPT_RAG_LLM="Test RAG prompt" && \
export VECTOR_DB_COLLECTION_NAME="test-collection" && \
export IMAGE_FILE_PATH="/tmp/test.jpg" && \
export INFO_TEXT="Test info" && \
export AZURE_EMBEDDING_API_KEY="test-embedding-key" && \
export AZURE_EMBEDDING_ENDPOINT="https://test-embedding.openai.azure.com/" && \
export AZURE_EMBEDDING_API_VERSION="2023-05-15" && \
export AZURE_EMBEDDING_DEPLOYMENT="test-deployment" && \
export AZURE_EMBEDDING_MODEL_NAME="text-embedding-ada-002" && \
export GEMINI_API_KEY="test-gemini-key" && \
export GEMINI_PROJECT_ID="test-project" && \
export AZURE_DALLE_3_API_KEY="test-dalle3-key" && \
export AZURE_DALLE_3_ENDPOINT="https://test-dalle3.openai.azure.com/" && \
export AZURE_DALLE_3_API_VERSION="2024-02-01" && \
export AZURE_DALLE_3_DEPLOYMENT="test-dalle3-deployment"
```

## 🧪 Common Test Commands

### Run All Tests with Coverage
```bash
poetry run pytest tests/test_api.py tests/test_api_advanced.py --cov=rtl_rag_chatbot_api --cov-report=html
```

### Run Individual Test Files
```bash
# End-to-end tests only
poetry run pytest tests/test_api.py -v

# Advanced error handling tests only  
poetry run pytest tests/test_api_advanced.py -v
```

### Run Specific Test Classes
```bash
# Main pipeline tests
poetry run pytest tests/test_api.py::TestEndToEndPipeline -v

# Error handling tests
poetry run pytest tests/test_api_advanced.py::TestAdvancedMultiFileScenarios -v
```

### Run Individual Tests
```bash
# Single file upload
poetry run pytest tests/test_api.py::TestEndToEndPipeline::test_1_single_file_upload -v

# Multi-file upload  
poetry run pytest tests/test_api.py::TestEndToEndPipeline::test_4_multi_file_upload -v

# Image processing
poetry run pytest tests/test_api.py::test_chat_with_single_image -v

# Invalid file handling
poetry run pytest tests/test_api_advanced.py::TestAdvancedMultiFileScenarios::test_invalid_existing_file_ids -v
```

## 📊 Coverage Reports

### View HTML Coverage Report
```bash
# After running tests with --cov-report=html
open htmlcov/index.html
```

### Coverage for Specific Module
```bash
poetry run pytest tests/test_api*.py --cov=rtl_rag_chatbot_api.app --cov-report=term
```

## 🔧 Debugging Tests

### Run with Verbose Output
```bash
poetry run pytest tests/test_api.py -v -s
```

### Run Single Test with Debug
```bash
poetry run pytest tests/test_api.py::test_chat_with_csv -v -s --tb=short
```

### Run Tests with Specific Pattern
```bash
# All image-related tests
poetry run pytest -k "image" -v

# All upload tests
poetry run pytest -k "upload" -v

# All delete tests
poetry run pytest -k "delete" -v
```

## ⚡ Performance

### Run Tests in Parallel (if pytest-xdist installed)
```bash
poetry run pytest tests/test_api*.py -n auto --cov=rtl_rag_chatbot_api
```

### Run with Timeout
```bash
poetry run pytest tests/test_api.py --timeout=600  # 10 minute timeout
```

## 🎯 Expected Results

### Success Indicators
- ✅ **All tests pass** in `test_api.py` and `test_api_advanced.py`
- ✅ **~8-10 minute** execution time
- ✅ **45% overall coverage** (5,808 total statements, 3,202 covered)
- ✅ **Key component coverage:**
  - `app.py`: 64%
  - `chatbot_creator.py`: 83%
  - `chroma_manager.py`: 84%
  - `file_handler.py`: 59%
  - `gcs_handler.py`: 56%

### Coverage Details
- **Total Statements**: 5,808
- **Lines Covered**: 3,202
- **Lines Missing**: 2,606

### Components Needing Attention
- ❌ Modules with low or 0% coverage, such as `image_reader.py` (17%), `website_handler.py` (34%), and `sharepoint_handler.py` (0%).

### Common Issues
- ❌ **Environment errors**: Check all env vars are set
- ❌ **Import errors**: Use `poetry run` or activate shell
- ❌ **Timeout errors**: Tests may take longer on slower systems
- ❌ **File not found**: Ensure you're in correct directory

## 📁 Test Files Overview

### `tests/test_api.py` (End-to-End Tests)
- ✅ Single file upload & chat
- ✅ Multi-file upload & chat  
- ✅ File type processing (PDF, CSV, Excel, Image, SQLite, URL)
- ✅ Image generation (DALL-E 3, Imagen 3)
- ✅ Delete operations

### `tests/test_api_advanced.py` (Error Handling)
- ✅ Invalid file formats
- ✅ Empty files
- ✅ Invalid file IDs
- ✅ Concurrent uploads
- ✅ System failures
- ✅ Resource management

---

For detailed documentation, see: [Multi-File Upload Feature Testing Guide](testing-multi-file-upload-guide.md)
