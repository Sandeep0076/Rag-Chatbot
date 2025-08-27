# Multi-File Upload Feature Testing Guide

## 📋 Overview

This document provides comprehensive documentation for the end-to-end unit tests created for the RAG PDF API's multi-file upload feature. The testing suite validates file upload workflows, error handling, edge cases, and provides code coverage analysis.

## 🎯 Testing Objectives

### Primary Goals
1. **End-to-End Validation**: Test complete workflows from file upload to chat functionality
2. **Multi-File Support**: Validate concurrent processing of multiple files
3. **Error Handling**: Ensure robust handling of edge cases and failures
4. **Code Coverage**: Assess coverage of multi-file upload components
5. **Integration Testing**: Verify proper interaction between system components

### Coverage Focus Areas
- **Core Application Layer** (`rtl_rag_chatbot_api/app.py`)
- **File Processing Pipeline** (`rtl_rag_chatbot_api/chatbot/file_handler.py`)
- **Storage Management** (`rtl_rag_chatbot_api/chatbot/gcs_handler.py`)
- **Embedding Creation** (`rtl_rag_chatbot_api/chatbot/embedding_handler.py`)
- **Data Processing** (CSV, Excel, Image, Database handlers)

## 📁 Test File Structure

```
tests/
├── test_api.py              # End-to-end workflow tests
├── test_api_advanced.py     # Error handling & edge cases
├── mock_files/              # Test data files
│   ├── mock_file1.pdf
│   ├── mock_file2.pdf
│   ├── mock_file.csv
│   ├── mock_file.png
│   ├── mock_file2.jpg
│   ├── mock_file.sqlite
│   └── mock_file.xlsx
└── __init__.py
```

## 🧪 Test Categories

### 1. End-to-End Workflow Tests (`test_api.py`)

#### **TestEndToEndPipeline Class**
Stateful integration tests that pass data between test methods:

**Test Flow:**
```
Upload → Process → Chat → Multi-Upload → Multi-Chat → Delete
```

**Individual Tests:**
- `test_1_single_file_upload()` - Upload single PDF, verify processing
- `test_2_check_embeddings()` - Validate embedding creation
- `test_3_single_file_chat()` - Chat with uploaded file
- `test_4_multi_file_upload()` - Upload additional file with existing ID
- `test_5_multi_file_chat()` - Chat with multiple files
- `test_6_delete_single_file()` - Delete individual file
- `test_7_delete_multi_files()` - Delete multiple files

#### **File Type Specific Tests**

**CSV Processing:**
- `test_chat_with_csv()` - Upload CSV, query tabular data
- `test_chat_with_csv_visualization()` - Generate charts from CSV data

**Document Processing:**
- `test_chat_with_doc_gemini()` - Process text files with Gemini model

**Image Processing:**
- `test_chat_with_single_image()` - Upload and analyze single image
- `test_chat_with_multiple_images()` - Process multiple images concurrently

**Database Processing:**
- `test_chat_with_database()` - Upload SQLite database, query data
- `test_chat_with_excel()` - Process Excel files

**URL Processing:**
- `test_chat_with_single_url()` - Extract content from single URL
- `test_chat_with_multiple_urls()` - Process multiple URLs concurrently

**Image Generation:**
- `test_generate_dalle3_images()` - Generate images using DALL-E 3
- `test_generate_imagen3_images()` - Generate images using Imagen 3

### 3. Migration System Tests

#### **TestMigrationSystem Class**
Tests for the legacy embedding migration system:

**Migration Detection Tests:**
- `test_migration_detection_mixed_types()` - Detect mixed embedding types
- `test_migration_detection_all_current()` - Skip migration when all files current
- `test_migration_detection_all_legacy()` - Handle all legacy files appropriately

**Migration Execution Tests:**
- `test_migration_execution_parallel()` - Execute migration with parallel processing
- `test_migration_context_preservation()` - Preserve migration context across pipeline
- `test_migration_username_preservation()` - Preserve all usernames during migration
- `test_migration_file_id_consistency()` - Maintain file ID consistency

**Migration Error Handling Tests:**
- `test_migration_missing_files_error()` - Handle missing files for migration
- `test_migration_partial_failure()` - Handle partial migration failures
- `test_migration_rollback_on_error()` - Rollback changes on migration failure

**Migration Performance Tests:**
- `test_migration_parallel_performance()` - Measure parallel processing performance
- `test_migration_resource_usage()` - Monitor resource usage during migration
- `test_migration_concurrent_users()` - Test migration with multiple concurrent users

### 2. Advanced Error Handling Tests (`test_api_advanced.py`)

#### **TestAdvancedMultiFileScenarios Class**
Tests for error conditions, edge cases, and complex scenarios:

**Error Handling Tests:**
- `test_invalid_file_format_error_handling()` - Invalid file formats
- `test_empty_file_upload()` - Empty file handling
- `test_invalid_existing_file_ids()` - Invalid file ID rejection
- `test_partial_failure_existing_and_new_files()` - Mixed valid/invalid scenarios

**Edge Case Tests:**
- `test_multiple_same_type_files_upload()` - Multiple files of same type
- `test_concurrent_upload_same_file()` - Concurrent uploads by different users
- `test_large_filename_handling()` - Very long filename truncation
- `test_url_upload_with_invalid_urls()` - Invalid URL handling

**System Integration Tests:**
- `test_gcs_upload_failure_handling()` - GCS service failure simulation
- `test_session_consistency_multi_upload()` - Session ID consistency
- `test_delete_nonexistent_files()` - Delete non-existent files
- `test_embedding_check_invalid_file_id()` - Invalid embedding checks
- `test_status_check_invalid_file_id()` - Invalid status queries
- `test_upload_without_username()` - Missing required parameters
- `test_chat_with_invalid_session_combination()` - Session validation

#### **TestResourceManagementAndCleanup Class**
Resource management and cleanup validation:

- `test_cleanup_after_failed_upload()` - Resource cleanup verification
- `test_memory_usage_multiple_large_files()` - Memory management testing

## 🛠 How to Run Tests

### Prerequisites

1. **Environment Setup:**
```bash
# Navigate to project directory
cd /Users/pathania/Rag/rag-pdf-api

# Activate poetry environment
poetry shell

#export variables
```

### Running Tests

#### **Run All End-to-End Tests:**
```bash
poetry run pytest tests/test_api.py -v --cov=rtl_rag_chatbot_api --cov-report=html
```

#### **Run Combined Coverage Analysis:**
```bash
# Option 1: Run both files together
poetry run pytest tests/test_api*.py --cov=rtl_rag_chatbot_api --cov-report=term --cov-report=html

# Option 2: Run separately with coverage append
poetry run pytest tests/test_api.py --cov=rtl_rag_chatbot_api --cov-report=html --cov-append
poetry run pytest tests/test_api_advanced.py --cov=rtl_rag_chatbot_api --cov-report=html --cov-append
```

#### **Run Specific Test Classes:**
```bash
# End-to-end pipeline only
poetry run pytest tests/test_api.py::TestEndToEndPipeline -v

# Error handling only
poetry run pytest tests/test_api_advanced.py::TestAdvancedMultiFileScenarios -v

# Resource management only
poetry run pytest tests/test_api_advanced.py::TestResourceManagementAndCleanup -v
```

#### **Run Individual Tests:**
```bash
# Single file upload test
poetry run pytest tests/test_api.py::TestEndToEndPipeline::test_1_single_file_upload -v

# Multi-file upload test
poetry run pytest tests/test_api.py::TestEndToEndPipeline::test_4_multi_file_upload -v

# Image generation test
poetry run pytest tests/test_api.py::test_generate_dalle3_images -v
```

## 📊 Coverage Analysis

### Current Coverage Results (as of last run)

**Overall System Coverage:**
- **Total Coverage**: 45%
- **Total Statements**: 5,808
- **Covered Statements**: 3,202
- **Missing Statements**: 2,606

**Detailed Module Coverage:**

| Module | Coverage | Statements | Covered | Missing |
|--------|----------|------------|---------|---------|
| `app.py` | 64% | 1112 | 717 | 395 |
| `chatbot_creator.py` | 83% | 137 | 114 | 23 |
| `file_handler.py` | 59% | 458 | 270 | 188 |
| `gcs_handler.py` | 56% | 286 | 160 | 126 |
| `embedding_handler.py` | 49% | 450 | 221 | 229 |
| `gemini_handler.py` | 47% | 266 | 125 | 141 |
| `csv_handler.py` | 42% | 433 | 180 | 253 |
| `dalle_handler.py` | 77% | 26 | 20 | 6 |
| `imagen_handler.py` | 43% | 146 | 63 | 83 |
| `combined_image_handler.py` | 44% | 27 | 12 | 15 |
| `base_handler.py` | 43% | 276 | 120 | 156 |
| `chroma_manager.py` | 84% | 67 | 56 | 11 |
| `prepare_sqlitedb...` | 61% | 256 | 155 | 101 |
| `image_reader.py` | 17% | 95 | 16 | 79 |
| `website_handler.py` | 34% | 300 | 103 | 197 |
| **Other Utils & Common** | ~75% | 105 | 79 | 26 |
| **Modules with 0% Coverage** | 0% | ~850 | 0 | ~850 |

**Key Takeaways:**
- The overall test coverage is **45%**
- Core components like `app.py` (64%), `chatbot_creator.py` (83%), and `chroma_manager.py` (84%) have good coverage.
- Handlers for major functionalities like `file_handler.py` (59%), `gcs_handler.py` (56%), and `embedding_handler.py` (49%) are partially tested but have significant room for improvement.
- Modules like `sharepoint_handler.py`, `confluence_handler.py`, and several workflow/common scripts still have 0% coverage and should be prioritized in the next testing phase.



### Coverage Reporting

**Generate Coverage Reports:**
```bash
# Run tests with coverage
poetry run pytest tests/test_api*.py --cov=rtl_rag_chatbot_api --cov-report=html --cov-report=term

# View detailed HTML report
open htmlcov/index.html
```

**Current Test Statistics:**
- **Total Test Files**: 2 (`test_api.py`, `test_api_advanced.py`)
- **Total Test Methods**: 35+ individual tests
- **Execution Time**: ~8-10 minutes for full suite
- **Last Updated**: 2025-07-07 13:31 +0200


## 📈 Success Metrics

### Test Execution Results
- **Total Tests**: 35 (18 end-to-end + 17 advanced)
- **Pass Rate**: 100% (when environment properly configured)
- **Average Execution Time**: 8-10 minutes for full suite
- **Coverage Improvement**: +10-15% over baseline

### Validation Criteria
✅ **File Upload Pipeline**: All file types process correctly  
✅ **Multi-File Handling**: Concurrent uploads work properly  
✅ **Error Handling**: Graceful failure management  
✅ **Delete Operations**: Proper resource cleanup  
✅ **Session Management**: Consistent session tracking  
✅ **Integration**: External service interaction  
