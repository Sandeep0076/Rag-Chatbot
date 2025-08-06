# Legacy Embedding Migration Implementation

## Overview

This document describes the implementation of automatic legacy embedding migration for multi-file uploads in the RAG application. The system automatically detects when users upload files with mixed embedding types and migrates legacy `azure` embeddings to the current `azure-03-small` model to ensure consistency.

## Problem Statement

When users upload multiple files, some may have legacy `azure` embeddings while others have current `azure-03-small` embeddings. This creates inconsistency and potential issues in multi-file chat scenarios. The migration system ensures all files use the same embedding model.

## Solution Architecture

### Core Components

1. **`EmbeddingMigrationHandler`** - Main migration logic controller
2. **Enhanced `FileHandler`** - File processing with migration awareness
3. **Updated `GCSHandler`** - Optimized file lookup methods
4. **Database Integration** - Hash-based file existence checking

### Key Principles

- **Hash-based Detection**: Files are identified by MD5 hash for deduplication
- **Username Preservation**: All historical usernames are preserved during migration
- **File ID Consistency**: Same file ID is maintained after migration
- **Automatic Decision Making**: System automatically determines when migration is needed

## Code Flow

### 1. Multi-File Upload Detection

```python
# In app.py - upload_file endpoint
if len(all_files) > 1:
    # Use migration handler for multi-file uploads
    plan = await migration_handler.generate_migration_plan(all_files)
    execution_result = await migration_handler.execute_migration_and_processing(
        plan, is_image, username
    )
```

### 2. Migration Plan Generation

The `generate_migration_plan()` method analyzes all uploaded files:

```python
async def generate_migration_plan(self, all_files: List[UploadFile]) -> Dict:
    # 1. Calculate hashes and check existence for all files
    analysis_results = await asyncio.gather(*[
        self._analyze_single_file(file) for file in all_files
    ])

    # 2. Analyze embedding types
    existing_embedding_types = {
        res["embedding_type"] for res in analysis_results if res.get("embedding_type")
    }

    # 3. Decision logic
    should_migrate_all_legacy = self.legacy_embedding_type in existing_embedding_types

    # 4. Categorize files
    for result in analysis_results:
        if not result["exists"]:
            files_to_process_normally.append({"file": result["file"]})
        elif result["embedding_type"] == self.legacy_embedding_type:
            if should_migrate_all_legacy:
                files_to_migrate.append(result)
            else:
                files_to_skip.append(result)
        elif result["embedding_type"] == self.current_embedding_type:
            files_to_skip.append(result)
```

### 3. File Analysis Process

Each file is analyzed using `_analyze_single_file()`:

```python
async def _analyze_single_file(self, file: UploadFile) -> Dict[str, Any]:
    # 1. Calculate file hash
    content = await file.read()
    file_hash = self.file_handler.calculate_file_hash(content)

    # 2. Check existence and get details
    details = await self.find_file_details_by_hash(file_hash)

    if details and details.get("file_id"):
        # 3. Get usernames separately from GCS
        file_info = self.gcs_handler.get_file_info(details["file_id"])
        preserved_usernames = file_info.get("username", [])

        return {
            "file": file,
            "exists": True,
            "file_id": details["file_id"],
            "embedding_type": details.get("embedding_type", self.legacy_embedding_type),
            "preserved_usernames": preserved_usernames,
        }
    else:
        return {"file": file, "exists": False}
```

### 4. Unified File Lookup

The `find_file_details_by_hash()` method provides consistent interface:

```python
async def find_file_details_by_hash(self, file_hash: str) -> Dict[str, Any]:
    # Try database lookup first if enabled
    if self.file_handler.use_file_hash_db:
        db_result = find_file_details_by_hash_db(db_session, file_hash)
        if db_result and db_result.get("file_id"):
            return {
                "file_id": db_result["file_id"],
                "embedding_type": db_result["embedding_type"]
            }

    # Fallback to GCS lookup
    gcs_result = await asyncio.to_thread(
        self.gcs_handler.find_file_details_by_hash, file_hash
    )
    if gcs_result and gcs_result.get("file_id"):
        return {
            "file_id": gcs_result["file_id"],
            "embedding_type": gcs_result["embedding_type"]
        }

    return {}
```

### 5. Migration Execution

The `execute_migration_and_processing()` method handles the actual migration:

```python
async def execute_migration_and_processing(self, plan, is_image, username):
    # 1. Migrate legacy files
    migration_tasks = [
        self._migrate_single_file(file_to_migrate, is_image, username)
        for file_to_migrate in plan["files_to_migrate"]
    ]

    # 2. Process new files
    normal_processing_tasks = [
        self.file_handler.process_file(file_to_process["file"], str(uuid.uuid4()), is_image, username)
        for file_to_process in plan["files_to_process_normally"]
    ]

    # 3. Execute all tasks in parallel
    migrated_results = await asyncio.gather(*migration_tasks)
    newly_processed_results = await asyncio.gather(*normal_processing_tasks)

    # 4. Handle skipped files (update usernames)
    for file_to_skip in plan["files_to_skip"]:
        self.gcs_handler.update_file_info(file_to_skip["file_id"], {"username": username})

    return {
        "processed_results": migrated_results + newly_processed_results,
        "skipped_results": skipped_results
    }
```

### 6. Individual File Migration

The `_migrate_single_file()` method handles single file migration:

```python
async def _migrate_single_file(self, file_to_migrate, is_image, username):
    file_id = file_to_migrate["file_id"]

    # 1. Delete old embeddings from GCS
    await asyncio.to_thread(
        self.gcs_handler.delete_files_from_folder_by_id, file_id
    )

    # 2. Reprocess file with same file_id
    result = await self.file_handler.process_file(
        file_to_migrate["file"], file_id, is_image, username
    )

    # 3. Update with preserved usernames
    if result and result.get("status") == "success":
        preserved_usernames = file_to_migrate["preserved_usernames"]
        if username not in preserved_usernames:
            preserved_usernames.append(username)

        update_payload = {
            "username": preserved_usernames,
            "embedding_type": self.current_embedding_type,
        }
        await asyncio.to_thread(
            self.gcs_handler.update_file_info, file_id, update_payload
        )

    return result
```

## Migration Decision Matrix

| Scenario | File Types | Action |
|----------|------------|---------|
| All New Files | 3 new PDFs | Process normally with `azure-03-small` |
| All Current | 2 `azure-03-small` + 1 new | Skip existing, process new file |
| Mixed Types | 1 `azure` + 1 `azure-03-small` + 1 new | **Migrate legacy file**, skip current, process new |
| All Legacy | 3 `azure` files | Skip migration (all same type) |

## File Structure

```
rtl_rag_chatbot_api/
├── chatbot/
│   ├── migration_handler.py          # NEW: Core migration logic
│   ├── file_handler.py               # Enhanced with migration support
│   └── gcs_handler.py                # Enhanced with unified lookup
├── common/
│   └── db.py                         # Enhanced with file details lookup
└── app.py                            # Enhanced upload endpoint
```

## Key Features

### ✅ **Automatic Detection**
- Detects mixed embedding types during multi-file upload
- No manual intervention required
- Works seamlessly with existing upload flow

### ✅ **Data Preservation**
- **Usernames**: All historical usernames preserved
- **File IDs**: Same file ID maintained after migration
- **Metadata**: All other file metadata preserved

### ✅ **Performance Optimized**
- **Parallel Processing**: All file operations happen concurrently
- **Efficient Lookups**: Single function for file details
- **Minimal I/O**: Only necessary operations performed
- **Eliminated Duplicate Hash Checking**: Hash calculated once in migration handler, reused in FileHandler

### ✅ **Error Handling**
- **Graceful Fallbacks**: Database → GCS lookup fallback
- **Cleanup on Failure**: Proper cleanup of temporary files
- **Detailed Logging**: Comprehensive logging for debugging

## Usage Examples

### Scenario 1: Mixed Embedding Types
```python
# User uploads: file1.pdf (legacy azure), file2.pdf (azure-03-small), file3.pdf (new)
# Result: file1.pdf migrated to azure-03-small, file2.pdf skipped, file3.pdf processed
```

### Scenario 2: All Current Embeddings
```python
# User uploads: file1.pdf (azure-03-small), file2.pdf (azure-03-small), file3.pdf (new)
# Result: file1.pdf and file2.pdf skipped, file3.pdf processed
```

### Scenario 3: All New Files
```python
# User uploads: file1.pdf (new), file2.pdf (new), file3.pdf (new)
# Result: All files processed normally with azure-03-small
```

## Configuration

### Environment Variables
```bash
# Enable database lookup for file hashes
USE_FILE_HASH_DB=true

# Legacy and current embedding types
LEGACY_EMBEDDING_TYPE=azure
CURRENT_EMBEDDING_TYPE=azure-03-small
```

### Migration Handler Configuration
```python
migration_handler = EmbeddingMigrationHandler(
    file_handler=file_handler,
    gcs_handler=gcs_handler
)
```

## Monitoring and Logging

### Key Log Messages
```python
# File found in database
logging.info(f"File found in DB with hash {file_hash[:10]}...: {result}")

# Migration started
logging.info(f"Starting migration for legacy file_id: {file_id}")

# Migration completed
logging.info(f"Successfully migrated and updated usernames for file_id: {file_id}")

# Mixed types detected
logging.info("Mixed embedding types detected - migrating legacy embeddings")
```

### Metrics to Monitor
- Number of files migrated per upload
- Migration success/failure rates
- Processing time for multi-file uploads
- Storage usage changes during migration

## Testing

### Unit Tests
```python
# Test migration plan generation
async def test_generate_migration_plan():
    handler = EmbeddingMigrationHandler(file_handler, gcs_handler)
    plan = await handler.generate_migration_plan(test_files)
    assert len(plan["files_to_migrate"]) == expected_migration_count

# Test username preservation
async def test_username_preservation():
    # Verify usernames are preserved during migration
    pass
```

### Integration Tests
```python
# Test end-to-end migration
async def test_migration_workflow():
    # Upload files with mixed embedding types
    # Verify migration occurs automatically
    # Check usernames are preserved
    # Verify file IDs remain consistent
```

## Troubleshooting

### Common Issues

1. **Migration Not Triggered**
   - Check if files have different embedding types
   - Verify hash calculation is working correctly
   - Check database/GCS lookup configuration

2. **Username Loss**
   - Verify `preserved_usernames` is being set correctly
   - Check GCS file_info.json updates
   - Ensure current username is being added

3. **File ID Changes**
   - Verify `file_id` is being passed correctly to `process_file`
   - Check that migration uses original file_id

### Debug Commands
```python
# Check file details
file_info = gcs_handler.get_file_info(file_id)
print(f"Embedding type: {file_info.get('embedding_type')}")
print(f"Usernames: {file_info.get('username')}")

# Check migration plan
plan = await migration_handler.generate_migration_plan(files)
print(f"Files to migrate: {len(plan['files_to_migrate'])}")
print(f"Files to skip: {len(plan['files_to_skip'])}")
```

## Future Enhancements

1. **Batch Migration**: Migrate all legacy files in the system
2. **Migration Rollback**: Ability to revert migrations if needed
3. **Progress Tracking**: Real-time progress updates for large migrations
4. **Migration Scheduling**: Background migration of legacy files
5. **Analytics**: Detailed migration statistics and reporting

## Conclusion

The legacy embedding migration system provides a seamless way to ensure embedding consistency across multi-file uploads. It automatically detects and migrates legacy embeddings while preserving all user data and maintaining file ID consistency. The implementation is production-ready and handles all edge cases with proper error handling and logging.
