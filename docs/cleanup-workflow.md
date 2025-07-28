# RAG Application Cleanup System Documentation

## Overview
This document details the automatic cleanup system implemented in the RAG application to manage memory, storage, and database resources efficiently, with configurable cleanup parameters through environment variables.

## Table of Contents
1. [System Components](#system-components)
2. [Setup Process](#setup-process)
3. [Operation Workflow](#operation-workflow)
4. [Cleanup Process](#cleanup-process)




### Default Values and Their Purpose
1. **Cleanup Interval (60 minutes / 1 hours)**
   - This is how often the scheduler wakes up to check
   - Like an alarm clock ringing every hour


2. **Staleness Threshold (240 minutes / 4 hours)**
   - How long a file needs to be untouched to be considered "stale"
   - Files not accessed for 4+ hours are candidates for cleanup

3. **Minimum Cleanup Interval (30 minutes)**
   - Minimum time required between two cleanup attempts
   - Prevents too frequent cleanups even if requested


### Database Integration Configuration
The cleanup system now supports automatic database cleanup when `use_file_hash_db` is enabled:
- **Environment Variable**: `USE_FILE_HASH_DB=true`
- **Behavior**: When enabled, file deletions will also remove corresponding database records
- **Safety**: Database cleanup failures won't block GCS cleanup operations
- **Logging**: Detailed logs track database cleanup success/failure

### Example
Time 0:00 -> Application starts
             ↓

Time 1:00 -> First scheduler wake-up
             Checks last_cleanup time

             If (current_time - last_cleanup) ≥ 30 mins
                → Performs cleanup (including database if enabled)
             Updates last_cleanup timestamp
             ↓

Time 2:00 -> Next scheduler wake-up
             Same process repeats

## System Components

### Core Components
- **CleanupCoordinator**: Main class managing cleanup operations
- **ChromaDBManager**: Manages database connections and cleanup
- **GCSHandler**: Handles GCS operations and database cleanup integration
- **Database Layer**: FileInfo table for tracking file hashes and metadata
- **Scheduler**: Handles automatic cleanup timing
- **Logging System**: Tracks cleanup operations and errors

### Key Files
```plaintext
project/
├── cleanup_coordinator.py    # Main cleanup logic
├── app.py                   # Scheduler and API endpoints
└── chroma_manager.py        # Database management
```


## Operation Workflow

### Regular Operation
1. **File Access Tracking**
   - Monitors file access timestamps
   - Updates usage statistics
   - Tracks active sessions

2. **Resource Monitoring**
   ```plaintext
   ├── ChromaDB instances
   ├── Memory usage
   └── File system status
   ```

### File States
1. **Active Files**
   ```plaintext
   ├── Currently in use
   ├── Recently accessed (< staleness_threshold)
   └── Protected from cleanup
   ```

2. **Inactive Files**
   ```plaintext
   ├── No current usage
   ├── Approaching staleness
   └── Monitored for cleanup
   ```

3. **Stale Files**
   ```plaintext
   ├── Exceeded staleness_threshold
   ├── No active connections
   └── Marked for removal
   ```

##

# DELETE /files Endpoint

## Overview
Deletes files and their associated embeddings from **both local storage and Google Cloud Storage (GCS)**. This endpoint handles batch deletion requests and provides detailed feedback on the success or failure of each deletion operation.

## Request



### Request Body
```json
{
    "file_ids": ["string"]  // Array of file IDs to delete
}
```

## Response

### Success Response
```json
{
    "message": "File deletion completed",
    "deleted_files": ["string"],  // Array of successfully deleted file IDs
    "errors": [                   // Optional array of errors
        {
            "file_id": "string",
            "error": "string"
        }
    ]
}
```



## Deletion Process

1. **ChromaDB Cleanup**
   - Cleans up ChromaDB instances from memory
   - Removes initialized models

2. **Local Storage Cleanup**
   - Deletes the ChromaDB directory for each file
   - Path pattern: `./chroma_db/{file_id}`

3. **GCS Cleanup**
   - Removes all blobs with prefix `file-embeddings/{file_id}/`

4. **Database Cleanup** (NEW)
   - Automatically removes file records from the database when `use_file_hash_db` is enabled
   - Deletes all FileInfo records associated with the file_id
   - Provides detailed logging of deletion results
   - Non-blocking: Database cleanup failures won't prevent GCS cleanup
