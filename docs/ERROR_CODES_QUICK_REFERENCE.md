# Error Codes Quick Reference

## 🔍 Quick Lookup Table

| Code | Key | User-Friendly Description | Action Required |
|------|-----|---------------------------|-----------------|
| **2xxx - File & Document** |
| 2001 | ERROR_FILE_UPLOAD_FAILED | File upload failed | Retry upload or check file |
| 2002 | ERROR_FILE_TYPE_UNSUPPORTED | File type not supported | Use PDF, CSV, XLSX, or TXT |
| 2003 | ERROR_FILE_SIZE_EXCEEDED | File too large | Reduce file size |
| 2004 | ERROR_FILE_CORRUPTED | File is corrupted | Use a different file |
| 2005 | ERROR_PDF_PARSING_FAILED | Cannot read PDF text | Use text-based PDF |
| 2006 | ERROR_CSV_PARSING_FAILED | Cannot parse CSV | Check CSV format |
| 2007 | ERROR_DOCS_PARSING_FAILED | Cannot parse document | Check file integrity |
| 2009 | ERROR_FILE_NOT_FOUND | File not found | File may have been deleted |
| 2010 | ERROR_DOC_TEXT_TOO_SHORT | Document too short | Upload longer document |
| 2011 | ERROR_URL_EXTRACTION_FAILED | Cannot access URL | Check URL is valid |
| 2012 | ERROR_URL_CONTENT_TOO_SHORT | URL content insufficient | Try different URL |
| 2013 | ERROR_TABULAR_INVALID_DATA | Invalid table data | Check data format |
| 2014 | ERROR_CSV_NO_TABLES | No tables found | Ensure file has data |
| 2015 | ERROR_CSV_ALL_TABLES_EMPTY | All tables empty | Add data to file |
| 2019 | ERROR_TXT_EXTRACTION_FAILED | Cannot read text file | Check file encoding |
| **3xxx - Image** |
| 3001 | ERROR_IMAGE_READER_FAILED | Cannot read image | Check image file |
| 3002 | ERROR_IMAGE_FORMAT_UNSUPPORTED | Image format not supported | Use JPG, PNG, or GIF |
| 3003 | ERROR_IMAGE_CREATION_FAILED | Image generation failed | Retry or modify prompt |
| 3004 | ERROR_IMAGE_CREATION_PROMPT_REJECTED | Prompt rejected by filters | Modify prompt content |
| 3005 | ERROR_IMAGE_DIMENSIONS_INVALID | Invalid image size | Use standard dimensions |
| 3006 | ERROR_IMAGE_ANALYSIS_FAILED | Image analysis failed | Retry with different image |
| **4xxx - Query & Chat** |
| 4001 | ERROR_QUERY_INVALID | Invalid query | Rephrase your question |
| 4002 | ERROR_CONTEXT_RETRIEVAL_FAILED | Cannot find context | Check file was uploaded |
| 4003 | ERROR_LLM_GENERATION_FAILED | AI response failed | Retry your query |
| 4004 | ERROR_QUERY_TOO_LONG | Query too long | Shorten your question |
| 4005 | ERROR_NO_SOURCE_SELECTED | No document selected | Select a document first |
| 4006 | ERROR_CHART_JSON_INVALID | Invalid chart data | Contact support |
| 4007 | ERROR_CHART_GENERATION_FAILED | Chart creation failed | Retry or simplify request |
| 4008 | ERROR_SAFETY_FILTER_BLOCKED | Content blocked by filters | Rephrase your query |
| 4009 | ERROR_MODEL_INITIALIZATION_FAILED | Model loading failed | Contact support |
| 4010 | ERROR_EMBEDDING_CREATION_FAILED | Embedding creation failed | Retry file upload |
| 4011 | ERROR_EMBEDDINGS_NOT_FOUND | File not processed yet | Wait for processing |
| 4012 | ERROR_AGENT_EXECUTION_FAILED | Query execution failed | Try simpler query |
| 4013 | ERROR_TITLE_GENERATION_FAILED | Title generation failed | Title not created |
| 4014 | ERROR_API_OVERLOADED | API service overloaded | Wait and retry automatically |
| 4015 | ERROR_API_RETRY_EXHAUSTED | All retry attempts failed | Try again later |

## 📝 Common Error Scenarios

### Scenario: "I uploaded a PDF but it failed"

**Most likely errors:**
- **Error 2005**: PDF is scanned/image-based
  - ✅ Solution: Use a text-based PDF or enable OCR
- **Error 2004**: PDF is corrupted
  - ✅ Solution: Try re-saving the PDF or use different file
- **Error 2003**: PDF is too large
  - ✅ Solution: Compress PDF or split into smaller files

### Scenario: "I uploaded a CSV but nothing happened"

**Most likely errors:**
- **Error 2014**: No tables detected
  - ✅ Solution: Ensure CSV has proper headers and structure
- **Error 2015**: Tables are empty
  - ✅ Solution: Add data rows to your CSV
- **Error 2006**: CSV format invalid
  - ✅ Solution: Check encoding (use UTF-8) and delimiters

### Scenario: "I can't chat with my document"

**Most likely errors:**
- **Error 4011**: Embeddings not ready
  - ✅ Solution: Wait 30-60 seconds for processing
- **Error 4002**: Context not found
  - ✅ Solution: Re-upload the document
- **Error 4005**: No source selected
  - ✅ Solution: Select a document from the list

### Scenario: "Image generation didn't work"

**Most likely errors:**
- **Error 3004**: Prompt rejected
  - ✅ Solution: Modify prompt to be more appropriate
- **Error 3003**: Generation failed
  - ✅ Solution: Retry or try simpler prompt

### Scenario: "API overloaded error"

**Most likely errors:**
- **Error 4014**: API service temporarily overloaded
  - ✅ Solution: System automatically retries (3 attempts with delays)
- **Error 4015**: All retry attempts exhausted
  - ✅ Solution: Wait 1-2 minutes and try again

## 🎯 Error Code Ranges

- **1000-1999**: System, authentication, general errors
- **2000-2999**: File handling and document processing
- **3000-3999**: Image operations
- **4000-4999**: Queries, chat, and AI operations
- **5000-5999**: Database and storage operations

## 🔗 Related Documentation

- [Complete Error Handling Guide](./error-handling-guide.md)
- [Streamlit Error Display Examples](./streamlit-error-display-examples.md)
- [API Usage Documentation](./README_API_USAGE.md)

## 💡 Pro Tips

1. **Always note the error code** - It helps support team diagnose issues quickly
2. **Read the error message** - It usually contains specific guidance
3. **Check file format** - Many errors are due to incorrect file types
4. **Wait for processing** - File processing takes 30-60 seconds
5. **Try again** - Some errors are temporary and resolve on retry

---

Last updated: 2025-12-30
