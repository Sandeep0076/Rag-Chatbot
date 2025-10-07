# Default Embedding Type Configuration

## Overview

The default embedding type for new files can now be configured through an environment variable, making it flexible to switch between different embedding models.

## Environment Variable

Set the `DEFAULT_EMBEDDING_TYPE` environment variable to specify the default embedding type for new file uploads.

### Supported Values

- `azure-3-large` (default) - Uses Azure OpenAI text-embedding-3-large model
- `azure` - Uses Azure OpenAI text-embedding-ada-002 model (legacy)

### Examples

```bash
# Use the new text-embedding-3-large model (default)
export DEFAULT_EMBEDDING_TYPE="azure-3-large"

# Use the legacy ada-002 model
export DEFAULT_EMBEDDING_TYPE="azure"
```

## Configuration Files Updated

The following files have been updated to use the configurable default:

1. **configs/app_config.py** - Added `default_embedding_type` field to `ChatbotConfig`
2. **configs/__init__.py** - Added environment variable mapping
3. **rtl_rag_chatbot_api/chatbot/embedding_handler.py** - Uses configurable default for new uploads
4. **rtl_rag_chatbot_api/common/db.py** - Uses configurable default for database records
5. **rtl_rag_chatbot_api/app.py** - Updated API endpoints to use configurable default
6. **rtl_rag_chatbot_api/chatbot/chatbot_creator.py** - Uses configurable default for file processing
7. **rtl_rag_chatbot_api/chatbot/gemini_handler.py** - Uses configurable default for Gemini model

## Usage

1. Set the environment variable in your deployment environment
2. Restart the application
3. New file uploads will use the specified embedding type
4. Existing files will continue to use their original embedding type

## Backward Compatibility

- If the environment variable is not set, the system defaults to `azure-3-large`
- Existing files retain their original embedding type
- The change only affects new file uploads

## Auto-Migration to New Embeddings

The system now includes **automatic migration** from legacy `azure` embeddings to `azure-3-large`:

- **Transparent**: Happens automatically during file upload, chat, or status check
- **Database-First**: Always uses database as single source of truth for embedding type
- **All Scenarios**: Works for single files, multi-files, and all-legacy files
- **No Manual Work**: Users don't need to do anything

For more details, see [Auto-Migration Guide](auto-migration-guide.md)
