# Error Handling Guide

## Overview

This RAG PDF API implements a comprehensive, standardized error handling system that returns structured error responses with consistent fields to enable reliable frontend integration and debugging.

## Error Response Format

All API errors follow this standardized format:

```json
{
  "status": "error",
  "code": 2005,
  "key": "ERROR_PDF_PARSING_FAILED",
  "error_code": 2005,
  "error_key": "ERROR_PDF_PARSING_FAILED",
  "message": "Unable to extract text from this PDF. The file might be scanned, corrupted, or password-protected.",
  "details": {
    "file_path": "/path/to/file.pdf",
    "method": "ocr"
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always `"error"` for error responses |
| `code` | integer | Numeric error code (e.g., 2005) |
| `key` | string | Stable error key for UI translation/messaging (e.g., "ERROR_PDF_PARSING_FAILED") |
| `error_code` | integer | Backward compatibility - same as `code` |
| `error_key` | string | Backward compatibility - same as `key` |
| `message` | string | Human-readable error message |
| `details` | object | Optional contextual information (file_id, file_path, etc.) |

## Error Code Categories

Error codes are organized by category using thousands place:

### 1xxx - General & Authentication

| Code | Key | HTTP Status | Description |
|------|-----|-------------|-------------|
| 1000 | ERROR_UNKNOWN | 500 | Unknown or unexpected error |
| 1001 | ERROR_API_KEY_MISSING | 401 | API key is missing from request |
| 1002 | ERROR_API_KEY_INVALID | 401 | API key is invalid or expired |
| 1003 | ERROR_RATE_LIMIT_EXCEEDED | 429 | Too many requests, rate limit exceeded |
| 1004 | ERROR_SERVER_UNAVAILABLE | 503 | Server is temporarily unavailable |
| 1005 | ERROR_INSUFFICIENT_PERMISSIONS | 403 | User lacks necessary permissions |
| 1010 | ERROR_BAD_REQUEST | 400 | Invalid request format or parameters |

### 2xxx - File & Document

| Code | Key | HTTP Status | Description |
|------|-----|-------------|-------------|
| 2001 | ERROR_FILE_UPLOAD_FAILED | 500 | File upload operation failed |
| 2002 | ERROR_FILE_TYPE_UNSUPPORTED | 400 | File type not supported |
| 2003 | ERROR_FILE_SIZE_EXCEEDED | 413 | File size exceeds maximum allowed |
| 2004 | ERROR_FILE_CORRUPTED | 400 | File appears to be corrupted |
| 2005 | ERROR_PDF_PARSING_FAILED | 400 | Failed to extract text from PDF |
| 2006 | ERROR_CSV_PARSING_FAILED | 400 | Failed to parse CSV file |
| 2007 | ERROR_DOCS_PARSING_FAILED | 400 | Failed to parse document file |
| 2008 | ERROR_DOCUMENT_INDEXING_FAILED | 500 | Failed to index document for search |
| 2009 | ERROR_FILE_NOT_FOUND | 404 | Requested file not found |
| 2010 | ERROR_DOC_TEXT_TOO_SHORT | 400 | Extracted document text is too short |
| 2011 | ERROR_URL_EXTRACTION_FAILED | 400 | Failed to extract content from URL |
| 2012 | ERROR_URL_CONTENT_TOO_SHORT | 400 | URL content is too short/insufficient |
| 2013 | ERROR_TABULAR_INVALID_DATA | 422 | Tabular data is invalid |
| 2014 | ERROR_CSV_NO_TABLES | 400 | No tables found in CSV/Excel file |
| 2015 | ERROR_CSV_ALL_TABLES_EMPTY | 400 | All tables in file are empty |
| 2016 | ERROR_FILE_HASH_FAILED | 500 | Failed to compute file hash |
| 2017 | ERROR_FILE_SAVE_FAILED | 500 | Failed to save file to storage |
| 2018 | ERROR_DOC_TEXT_VALIDATION_FAILED | 400 | Document text validation failed |
| 2019 | ERROR_TXT_EXTRACTION_FAILED | 400 | Failed to extract text from TXT file |

### 3xxx - Image

| Code | Key | HTTP Status | Description |
|------|-----|-------------|-------------|
| 3001 | ERROR_IMAGE_READER_FAILED | 400 | Failed to read/analyze image |
| 3002 | ERROR_IMAGE_FORMAT_UNSUPPORTED | 415 | Image format not supported |
| 3003 | ERROR_IMAGE_CREATION_FAILED | 500 | Image generation failed |
| 3004 | ERROR_IMAGE_CREATION_PROMPT_REJECTED | 422 | Image prompt rejected by safety filters |
| 3005 | ERROR_IMAGE_DIMENSIONS_INVALID | 400 | Invalid image dimensions requested |
| 3006 | ERROR_IMAGE_ANALYSIS_FAILED | 500 | Image analysis operation failed |

### 4xxx - Query & Chat

| Code | Key | HTTP Status | Description |
|------|-----|-------------|-------------|
| 4001 | ERROR_QUERY_INVALID | 400 | Query format or content is invalid |
| 4002 | ERROR_CONTEXT_RETRIEVAL_FAILED | 404 | Failed to retrieve context for query |
| 4003 | ERROR_LLM_GENERATION_FAILED | 500 | LLM generation failed |
| 4004 | ERROR_QUERY_TOO_LONG | 413 | Query exceeds maximum length |
| 4005 | ERROR_NO_SOURCE_SELECTED | 400 | No document source selected |
| 4006 | ERROR_CHART_JSON_INVALID | 400 | Chart JSON format is invalid |
| 4007 | ERROR_CHART_GENERATION_FAILED | 500 | Chart generation failed |
| 4008 | ERROR_SAFETY_FILTER_BLOCKED | 422 | Content blocked by safety filters |
| 4009 | ERROR_MODEL_INITIALIZATION_FAILED | 500 | Failed to initialize model |
| 4010 | ERROR_EMBEDDING_CREATION_FAILED | 500 | Failed to create embeddings |
| 4011 | ERROR_EMBEDDINGS_NOT_FOUND | 404 | Embeddings not found for file |
| 4012 | ERROR_AGENT_EXECUTION_FAILED | 500 | SQL agent execution failed |
| 4013 | ERROR_TITLE_GENERATION_FAILED | 500 | Conversation title generation failed |

### 5xxx - Database & Storage

| Code | Key | HTTP Status | Description |
|------|-----|-------------|-------------|
| 5001 | ERROR_DATABASE_CONNECTION_FAILED | 500 | Database connection failed |
| 5002 | ERROR_DATABASE_QUERY_FAILED | 500 | Database query execution failed |
| 5003 | ERROR_GCS_UPLOAD_FAILED | 500 | Google Cloud Storage upload failed |
| 5004 | ERROR_GCS_DOWNLOAD_FAILED | 500 | Google Cloud Storage download failed |
| 5005 | ERROR_GCS_DELETE_FAILED | 500 | Google Cloud Storage delete failed |
| 5006 | ERROR_ENCRYPTION_FAILED | 500 | Data encryption failed |
| 5007 | ERROR_DECRYPTION_FAILED | 500 | Data decryption failed |

## Usage in Backend

### Raising Errors

```python
from rtl_rag_chatbot_api.common.errors import (
    PdfTextExtractionError,
    CsvInvalidOrEmptyError,
    EmbeddingsNotFoundError
)

# PDF extraction error
if word_count == 0:
    raise PdfTextExtractionError(
        "Unable to extract text from this PDF. The file might be scanned, corrupted, or password-protected.",
        details={"file_path": file_path, "method": "ocr"}
    )

# CSV parsing error
if df is None or df.empty:
    raise CsvInvalidOrEmptyError(
        f"Failed to read CSV file or file is empty: {file_path}",
        details={"file_path": file_path}
    )

# Embeddings not found
if not embeddings_exist:
    raise EmbeddingsNotFoundError(
        f"Embeddings not found for file {file_id}. Please create embeddings first.",
        details={"file_id": file_id}
    )
```

### Error Response Conversion

Errors are automatically caught and converted to JSON responses by the centralized exception handlers registered in `app.py`:

```python
from rtl_rag_chatbot_api.common.errors import register_exception_handlers

app = FastAPI(...)
register_exception_handlers(app)
```

## Usage in Frontend

### Parsing Error Responses

```javascript
// Example error response
const errorResponse = {
  status: "error",
  code: 2005,
  key: "ERROR_PDF_PARSING_FAILED",
  error_code: 2005,
  error_key: "ERROR_PDF_PARSING_FAILED",
  message: "Unable to extract text from this PDF...",
  details: {
    file_path: "/path/to/file.pdf",
    method: "ocr"
  }
};

// Access error information
console.log(`Error Code: ${errorResponse.code}`);
console.log(`Error Key: ${errorResponse.key}`);
console.log(`Message: ${errorResponse.message}`);
```

### Error Key to User Message Mapping

Frontend should maintain a mapping of error keys to user-friendly messages for internationalization:

```javascript
const ERROR_MESSAGES = {
  ERROR_PDF_PARSING_FAILED: {
    en: "We couldn't read this PDF. It might be scanned or password-protected. Please try another file.",
    de: "Wir konnten diese PDF nicht lesen. Sie ist möglicherweise gescannt oder passwortgeschützt. Bitte versuchen Sie eine andere Datei."
  },
  ERROR_CSV_NO_TABLES: {
    en: "No data tables found in this file. Please check if the file contains valid data.",
    de: "Keine Datentabellen in dieser Datei gefunden. Bitte prüfen Sie, ob die Datei gültige Daten enthält."
  },
  ERROR_EMBEDDINGS_NOT_FOUND: {
    en: "This file hasn't been processed yet. Please wait for processing to complete.",
    de: "Diese Datei wurde noch nicht verarbeitet. Bitte warten Sie, bis die Verarbeitung abgeschlossen ist."
  }
  // ... more mappings
};

function getErrorMessage(errorKey, language = 'en') {
  return ERROR_MESSAGES[errorKey]?.[language] || ERROR_MESSAGES[errorKey]?.en || 'An error occurred';
}
```

### Handling Specific Errors

```javascript
async function uploadFile(file) {
  try {
    const response = await fetch('/file/upload', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (data.status === 'error') {
      // Handle specific error types
      switch (data.key) {
        case 'ERROR_PDF_PARSING_FAILED':
          showError('PDF could not be read. Try a different file.');
          break;
        case 'ERROR_FILE_SIZE_EXCEEDED':
          showError('File is too large. Maximum size is 100MB.');
          break;
        case 'ERROR_CSV_NO_TABLES':
          showError('CSV file has no valid data tables.');
          break;
        default:
          showError(data.message || 'An error occurred');
      }
    } else {
      // Success handling
      handleSuccess(data);
    }
  } catch (error) {
    showError('Network error. Please try again.');
  }
}
```

## Common Error Scenarios

### Scenario 1: PDF Upload Fails (Text Extraction Error)

**Backend:**
```python
# In base_handler.py
if word_count == 0:
    raise PdfTextExtractionError(
        "Unable to extract text from this PDF. The file might be scanned, corrupted, or password-protected.",
        details={"file_path": file_path, "method": "ocr"}
    )
```

**API Response:**
```json
{
  "status": "error",
  "code": 2005,
  "key": "ERROR_PDF_PARSING_FAILED",
  "message": "Unable to extract text from this PDF...",
  "details": {
    "file_path": "/temp/abc123.pdf",
    "method": "ocr"
  }
}
```

### Scenario 2: CSV Upload Fails (No Tables)

**Backend:**
```python
# In file_handler.py
if len(table_info) == 0:
    raise CsvNoTablesError(
        f"No tables found in the database for file: {original_filename}. The file may be empty or corrupted.",
        details={"file_id": file_id, "original_filename": original_filename}
    )
```

**API Response:**
```json
{
  "status": "error",
  "code": 2014,
  "key": "ERROR_CSV_NO_TABLES",
  "message": "No tables found in the database for file: data.csv...",
  "details": {
    "file_id": "abc-123",
    "original_filename": "data.csv"
  }
}
```

### Scenario 3: Image Generation Fails (Prompt Rejected)

**Backend:**
```python
# In dalle_handler.py or imagen_handler.py
if "content_policy" in error_msg:
    raise ImagePromptRejectedError(
        "The prompt was rejected by content safety filters",
        details={"prompt": prompt, "model": "dall-e-3"}
    )
```

**API Response:**
```json
{
  "status": "error",
  "code": 3004,
  "key": "ERROR_IMAGE_CREATION_PROMPT_REJECTED",
  "message": "The prompt was rejected by content safety filters",
  "details": {
    "prompt": "...",
    "model": "dall-e-3"
  }
}
```

### Scenario 4: Chat Fails (Embeddings Not Found)

**Backend:**
```python
# In app.py
if not embeddings_exist:
    raise EmbeddingsNotFoundError(
        f"Embeddings not found for file {file_id}. Please create embeddings first.",
        details={"file_id": file_id}
    )
```

**API Response:**
```json
{
  "status": "error",
  "code": 4011,
  "key": "ERROR_EMBEDDINGS_NOT_FOUND",
  "message": "Embeddings not found for file abc-123...",
  "details": {
    "file_id": "abc-123"
  }
}
```

## Testing Error Responses

### Using cURL

```bash
# Test PDF upload with invalid file
curl -X POST http://localhost:8080/file/upload \
  -H "Authorization: Bearer <token>" \
  -F "files=@invalid.pdf" \
  -F "username=testuser"

# Expected response:
# {
#   "status": "error",
#   "code": 2005,
#   "key": "ERROR_PDF_PARSING_FAILED",
#   "message": "Unable to extract text from this PDF..."
# }
```

### Using Python

```python
import requests

response = requests.post(
    'http://localhost:8080/file/upload',
    headers={'Authorization': 'Bearer <token>'},
    files={'files': open('invalid.pdf', 'rb')},
    data={'username': 'testuser'}
)

error_data = response.json()
print(f"Error Code: {error_data['code']}")
print(f"Error Key: {error_data['key']}")
print(f"Message: {error_data['message']}")
```

## Best Practices

### Backend Development

1. **Always use structured errors**: Don't return plain strings or generic exceptions
2. **Include contextual details**: Add file_id, file_path, or other relevant info to `details`
3. **Use appropriate error types**: Choose the error class that best matches the failure
4. **Log errors appropriately**: Use `logging.error()` or `logging.warning()` before raising

### Frontend Development

1. **Check for error status**: Always check `status === "error"` in responses
2. **Use error keys, not messages**: Base UI logic on `key` field, not `message`
3. **Provide user-friendly messages**: Map error keys to localized, user-friendly text
4. **Display contextual help**: For common errors, provide suggestions for resolution
5. **Log full error details**: Log the complete error object for debugging

## Migration Guide

If you have existing code using plain error strings or generic exceptions, here's how to migrate:

### Before (Old Style)

```python
# Old style - DON'T DO THIS
if df is None:
    raise ValueError("CSV file is empty")

return {"error": "Failed to process file"}
```

### After (New Style)

```python
# New style - DO THIS
from rtl_rag_chatbot_api.common.errors import CsvInvalidOrEmptyError

if df is None:
    raise CsvInvalidOrEmptyError(
        "CSV file is empty",
        details={"file_path": file_path}
    )

# Errors are automatically converted to standardized responses
```

## Troubleshooting

### Error Not Being Caught

If your error is not being properly caught and converted:

1. Ensure the exception handler is registered in `app.py`
2. Check that you're raising a `BaseAppError` subclass
3. Verify the error is not being caught and re-raised as a generic exception

### Wrong HTTP Status Code

If the HTTP status code is incorrect:

1. Check the `ErrorSpec` definition in `errors.py`
2. Ensure you're using the correct error class for the situation
3. Verify the exception handler is using `exc.spec.http_status`

### Frontend Not Receiving Structured Error

If the frontend receives a generic error instead of structured error:

1. Check server logs for unhandled exceptions
2. Verify the error is being raised, not just logged
3. Ensure the API response is not being intercepted and modified

## Additional Resources

- Error definitions: `rtl_rag_chatbot_api/common/errors.py`
- Exception handlers: `rtl_rag_chatbot_api/app.py` (see `register_exception_handlers`)
- Error usage examples: Throughout `rtl_rag_chatbot_api/` modules

## Streamlit Frontend Integration

### Error Parsing

The Streamlit frontend automatically parses structured error responses from the API:

```python
def _parse_error_response(upload_response):
    """Parse error response from API and extract meaningful error message with code and key."""
    try:
        error_data = upload_response.json()
        if isinstance(error_data, dict):
            # Check for structured error format (code, key, message)
            code = error_data.get('code') or error_data.get('error_code')
            key = error_data.get('key') or error_data.get('error_key')
            message = error_data.get('message')

            # If we have structured error, format it nicely
            if code and key and message:
                return f"Error {code}: {key} - {message}"

            # Fallback to legacy error handling...
    except Exception as e:
        return upload_response.text
```

### Error Display Format

Errors are displayed in the Streamlit UI with the format:

```
Error {code}: {key} - {message}
```

**Example:**
```
Error 2005: ERROR_PDF_PARSING_FAILED - Unable to extract text from this PDF. The file might be scanned, corrupted, or password-protected.
```

### Handling Different Error Types

The Streamlit app handles errors from different sources:

1. **File Upload Errors**: Displayed when files fail to upload or process
2. **Chat Errors**: Shown when chat requests fail
3. **Image Generation Errors**: Displayed for DALL-E and Imagen failures

**Example - File Upload:**
```python
if upload_response.status_code != 200:
    error_msg = _parse_error_response(upload_response)
    st.error(f"Upload failed: {error_msg}")
```

**Example - Chat:**
```python
if chat_response.status_code != 200:
    error_msg = _parse_error_response(chat_response)
    st.error(f"Chat failed: {error_msg}")
```

**Example - Image Generation:**
```python
error_code = result.get('code') or result.get('error_code')
error_key = result.get('key') or result.get('error_key')
error_msg = result.get('message') or result.get('error', 'Unknown error')

if error_code and error_key:
    st.error(f"DALL-E Error {error_code}: {error_key} - {error_msg}")
else:
    st.error(f"Failed to generate image: {error_msg}")
```

### User-Friendly Error Messages

For end users, see [Streamlit Error Display Examples](./streamlit-error-display-examples.md) which provides:

- Common error scenarios with screenshots
- User-friendly explanations
- Action steps for each error type
- Quick reference for error codes

### Backward Compatibility

The error parser supports both old and new error formats:

- **New format**: Uses `code`, `key`, `message` fields
- **Old format**: Falls back to `error_code`, `error_key`, or legacy `error`/`detail` fields

This ensures existing integrations continue to work while supporting the new structured format.

## Support

For questions or issues with error handling:

1. Check this documentation
2. Review error definitions in `errors.py`
3. Examine existing error usage in the codebase
4. See [Streamlit Error Display Examples](./streamlit-error-display-examples.md) for frontend integration
5. Contact the development team
