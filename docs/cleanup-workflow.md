# RAG Application Cleanup System Documentation

## Overview
This document details the automatic cleanup system implemented in the RAG application to manage memory, storage, and database resources efficiently, with configurable cleanup parameters through environment variables.

## Table of Contents
1. [System Components](#system-components)
2. [Setup Process](#setup-process)
3. [Operation Workflow](#operation-workflow)
4. [Cleanup Process](#cleanup-process)




### Default Values and Their Purpose
1. **Cleanup Interval (240 minutes / 4 hours)**
   - Controls automatic cleanup frequency
   - Balances resource usage and system performance
   - Adjustable for different workload patterns

2. **Staleness Threshold (240 minutes / 4 hours)**
   - Determines file inactivity period
   - Marks files for cleanup after threshold
   - Configurable based on usage patterns

3. **Minimum Cleanup Interval (15 minutes)**
   - Prevents cleanup operation clustering
   - Protects system from cleanup overhead
   - Important for manual cleanup triggers

## System Components

### Core Components
- **CleanupCoordinator**: Main class managing cleanup operations
- **ChromaDBManager**: Manages database connections and cleanup
- **Scheduler**: Handles automatic cleanup timing
- **Logging System**: Tracks cleanup operations and errors

### Key Files
```plaintext
project/
├── cleanup_coordinator.py    # Main cleanup logic
├── app.py                   # Scheduler and API endpoints
└── chroma_manager.py        # Database management
```

## Setup Process

### 1. Initial Configuration
```python
cleanup_coordinator = CleanupCoordinator(configs, SessionLocal)
```
- Loads environment variables
- Establishes database connections
- Initializes logging system

### 2. Scheduler Setup
```python
scheduler.add_job(
    cleanup_coordinator.cleanup,
    trigger="interval",
    minutes=configs.cleanup.cleanup_interval_minutes
)
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

## Cleanup Process

### Trigger Methods
1. **Automatic Cleanup**
   ```python
   if time_since_last_cleanup >= cleanup_interval_minutes:
       initiate_cleanup()
   ```

2. **Manual Cleanup**
   ```http
   POST /file/cleanup
   ```

### Cleanup Steps
1. **Time Verification**
   ```plaintext
   ├── Check last cleanup timestamp
   ├── Verify minimum interval
   └── Check staleness thresholds
   ```

2. **File Analysis**
   ```plaintext
   ├── Scan directories
   ├── Check access times
   └── List stale files
   ```

3. **ChromaDB Cleanup**
   ```plaintext
   ├── Close inactive connections
   ├── Remove database files
   └── Clear memory cache
   ```

4. **Directory Cleanup**
   ```plaintext
   ├── Remove temporary files
   ├── Clean processed data
   └── Delete empty directories
   ```

5. **Logging and Updates**
   ```plaintext
   ├── Record cleanup actions
   ├── Update timestamps
   └── Log statistics
   ```



### Debug Process
1. **Check Logs**
   ```bash
   tail -f cleanup.log
   ```

2. **Manual Testing**
   ```http
   POST /file/cleanup
   GET /cleanup/status
   ```




## Conclusion
The configurable cleanup system ensures efficient resource management while maintaining system stability and data integrity. Regular monitoring, proper configuration, and maintenance ensure optimal performance.
