# CSV Chat Implementation Documentation

## Overview
The application implements a sophisticated chat system for CSV files using a combination of SQLite database storage, natural language processing, and SQL query generation. The system handles both new and existing files efficiently while maintaining concurrent user access and memory optimization.

## Architecture Components
- TabularDataHandler: Core class handling CSV/Excel interactions
- SQLite Database: Stores tabular data efficiently
- LangChain Integration: For SQL query generation
- Azure OpenAI: For natural language understanding

## Workflow

### 1. File Upload Process
a) New File:
- File uploaded via `/file/upload` endpoint
- Stored temporarily in `local_data/`
- Processed by `PrepareSQLFromTabularData` class
- Creates SQLite database in `./chroma_db/{file_id}/`
- Uploads database to GCS for persistence

b) Existing File:
- Checks file hash for duplicates
- Downloads existing SQLite database if not locally present
- Reuses existing database structure

### 2. Query Processing Flow

#### Direct Answer Path
Triggered when:
- Question can be directly mapped to SQL
- Contains SQL-like keywords (SELECT, FIND, LIST, etc.)
- Clear data retrieval intent

Process:
1. Question formatted using `format_question()`
2. SQL agent processes the query
3. Returns structured data response

#### Forced Answer Path
Triggered when:
- Complex questions needing inference
- No direct SQL mapping possible
- Contextual understanding required

Process:
1. Attempts direct answer first
2. If fails, uses `get_forced_answer()`
3. Uses LLM to interpret context and generate response

### 3. Memory Management
- Connection pooling with SQLAlchemy
- Session cleanup after each query
- Automatic resource disposal
- Concurrent user session handling

### 4. Performance Optimizations
- Lazy loading of models
- Connection pooling
- File caching
- Background cleanup tasks
