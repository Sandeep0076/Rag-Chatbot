# Streamlit Error Display Examples

## Overview

This document provides examples of how structured errors are displayed in the Streamlit frontend interface and guidance on how to interpret them.

## Error Display Format

All errors in the Streamlit interface follow this format:

```
Error {code}: {key} - {message}
```

**Example:**
```
Error 2005: ERROR_PDF_PARSING_FAILED - Unable to extract text from this PDF. The file might be scanned, corrupted, or password-protected.
```

## Common Error Scenarios

### File Upload Errors

#### Scenario 1: PDF Text Extraction Failed

**Display:**
```
Error 2005: ERROR_PDF_PARSING_FAILED - Unable to extract text from this PDF. The file might be scanned, corrupted, or password-protected.
```

**What it means:** The system couldn't read text from your PDF file.

**Actions to take:**
1. Check if your PDF is scanned (image-based) - try using "Chat with Image" instead
2. Verify the PDF isn't password-protected
3. Try re-saving the PDF from its source application
4. Use a different PDF file

#### Scenario 2: Document Text Too Short

**Display:**
```
Error 2010: ERROR_DOC_TEXT_TOO_SHORT - Unable to extract sufficient text from this file (less than 100 characters). Please try using the 'Chat with Image' feature instead.
```

**What it means:** The extracted text is too short to process.

**Actions to take:**
1. Use "Chat with Image" feature for scanned documents
2. Verify the document contains actual text content
3. Try a document with more content

#### Scenario 3: File Type Not Supported

**Display:**
```
Error 2002: ERROR_FILE_TYPE_UNSUPPORTED - File type not supported
```

**What it means:** The file format you uploaded isn't supported.

**Actions to take:**
1. Use supported formats: PDF, CSV, XLSX, TXT, or images (JPG, PNG)
2. Convert your file to a supported format

### CSV/Excel Upload Errors

#### Scenario 4: No Tables Found

**Display:**
```
Error 2014: ERROR_CSV_NO_TABLES - No tables found in the database for file: data.csv. The file may be empty or corrupted.
```

**What it means:** The CSV/Excel file doesn't contain any valid data tables.

**Actions to take:**
1. Ensure the file has proper headers and structure
2. Check that the file isn't empty
3. Verify the file isn't corrupted

#### Scenario 5: All Tables Empty

**Display:**
```
Error 2015: ERROR_CSV_ALL_TABLES_EMPTY - All tables in file are empty
```

**What it means:** The file has table structures but no data rows.

**Actions to take:**
1. Add data rows to your CSV/Excel file
2. Check that the file contains actual data, not just headers

#### Scenario 6: CSV Parsing Failed

**Display:**
```
Error 2006: ERROR_CSV_PARSING_FAILED - Failed to read CSV file or file is empty: /path/to/file.csv
```

**What it means:** The system couldn't parse the CSV file.

**Actions to take:**
1. Check the CSV encoding (use UTF-8)
2. Verify the delimiter (comma, semicolon, etc.)
3. Ensure the CSV structure is valid

### Chat Errors

#### Scenario 7: Embeddings Not Found

**Display:**
```
Error 4011: ERROR_EMBEDDINGS_NOT_FOUND - Embeddings not found for file abc-123. Please create embeddings first.
```

**What it means:** The file hasn't been processed yet or processing failed.

**Actions to take:**
1. Wait 30-60 seconds for file processing to complete
2. Check the file upload status
3. Try re-uploading the file

#### Scenario 8: Safety Filter Blocked

**Display:**
```
Error 4008: ERROR_SAFETY_FILTER_BLOCKED - Content blocked by safety filters
```

**What it means:** The query or response was flagged by content safety filters.

**Actions to take:**
1. Rephrase your question
2. Avoid sensitive or inappropriate content
3. Try a different query

#### Scenario 9: LLM Generation Failed

**Display:**
```
Error 4003: ERROR_LLM_GENERATION_FAILED - Chat request failed: {error details}
```

**What it means:** The AI model encountered an error generating a response.

**Actions to take:**
1. Try asking your question again
2. Simplify your query
3. Check if the document was uploaded correctly

### Image Generation Errors

#### Scenario 10: Image Creation Failed

**Display:**
```
DALL-E Error 3003: ERROR_IMAGE_CREATION_FAILED - Image generation failed
```

**What it means:** The image generation service encountered an error.

**Actions to take:**
1. Try generating the image again
2. Modify your prompt to be more specific
3. Try a different image size or model

#### Scenario 11: Prompt Rejected

**Display:**
```
Imagen Error 3004: ERROR_IMAGE_CREATION_PROMPT_REJECTED - The prompt was rejected by content safety filters
```

**What it means:** Your image prompt was flagged as inappropriate.

**Actions to take:**
1. Modify your prompt to be more appropriate
2. Avoid sensitive or explicit content
3. Try a different description

## Error Code Quick Reference

| Code Range | Category | Examples |
|------------|----------|----------|
| 1000-1999 | System & Auth | API key issues, rate limits |
| 2000-2999 | File & Document | PDF parsing, CSV processing, file uploads |
| 3000-3999 | Image Operations | Image generation, analysis |
| 4000-4999 | Chat & Query | Embeddings, LLM generation, safety filters |
| 5000-5999 | Database & Storage | Database errors, GCS operations |

## Understanding Error Details

Some errors include additional details in the response:

**Example with details:**
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

The `details` field provides additional context about the error, which may be shown in expanded error views or logged for debugging.

## Best Practices for Users

1. **Read the Error Code**: Note the error code for support requests
2. **Understand the Message**: The message explains what went wrong
3. **Take Action**: Follow the suggested actions to resolve the issue
4. **Retry if Temporary**: Some errors are temporary - try again
5. **Contact Support**: If errors persist, provide the error code and message

## Troubleshooting Tips

### For File Upload Issues
- Check file format and size
- Verify file isn't corrupted
- Try a different file
- Check encoding (for CSVs)

### For Chat Issues
- Wait for file processing to complete
- Verify embeddings were created
- Rephrase your question
- Check document content

### For Image Generation Issues
- Modify your prompt
- Try a different size/model
- Avoid inappropriate content
- Retry after a moment

## Getting Help

If you encounter an error that you cannot resolve:

1. Note the full error message including code and key
2. Check this documentation for common solutions
3. Review the [Error Handling Guide](./error-handling-guide.md) for technical details
4. Contact support with the error details

---

Last updated: 2025-01-06
