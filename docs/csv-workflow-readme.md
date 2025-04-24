# CSV Chat Implementation Documentation

## Overview
The application implements a sophisticated chat system for CSV files using a combination of SQLite database storage, natural language processing, and SQL query generation. The system handles both new and existing files efficiently while maintaining concurrent user access and memory optimization.

## Architecture Components
- TabularDataHandler: Core class handling CSV/Excel interactions
- SQLite Database: Stores tabular data efficiently
- LangChain Integration: For SQL query generation
- PromptHandler: For structuring prompts and questions
- Multiple LLM Providers: Azure OpenAI and Google Gemini for natural language understanding

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

#### Request Initiation
- User submits question to `/file/chat` endpoint in app.py
- Request is routed to `TabularDataHandler.ask_question` method
- Method validates inputs and initializes resources if needed

#### Question Transformation
- Raw user question is passed through `format_question()` from prompt_handler.py
- The question is enriched with table schema, metadata, and context
- System generates a more structured query that aligns with the data format

#### Intent Detection & Processing
The system analyzes the formatted question for specific keywords to determine processing path:

##### Direct Answer Path
Triggered when:
- Question contains analytical keywords (SELECT, FIND, LIST, SHOW, CALCULATE, etc.)
- Clear data retrieval intent is detected
- Question can be directly mapped to SQL

Process:
1. The agent (typically a SQL-capable LLM) processes the formatted question
2. The system extracts:
   - Final answer from the agent
   - Intermediate reasoning steps
   - SQL queries generated (if applicable)
3. These components are combined into a comprehensive prompt
4. Model-specific response generation based on selected LLM:
   - Gemini models: Uses `get_gemini_non_rag_response()`
   - Azure models: Uses `get_azure_non_rag_response()`
5. Returns a natural language answer derived from the structured data

##### Alternative Path
Triggered when:
- No analytical keywords are detected
- Question may be conversational or unclear
- The system returns the formatted question itself

#### Error Handling
- Comprehensive error logging throughout the process
- Graceful error propagation to higher-level handlers
- User-friendly error messages for common failures

### 3. Memory Management
- Connection pooling with SQLAlchemy
- Session cleanup after each query
- Automatic resource disposal
- Concurrent user session handling
- Lazy initialization of agents and models to conserve resources

### 4. Performance Optimizations
- Lazy loading of models
- Connection pooling
- File caching
- Background cleanup tasks
- Model selection based on question complexity

### 5. Example Flow

When a user asks: "What is the average salary in the dataset?"

1. The question enters `/file/chat` endpoint
2. It's routed to `TabularDataHandler.ask_question`
3. The question is formatted with table context: "Given the table employee_data with columns [name, position, salary], calculate the average of the salary column"
4. The system detects "CALCULATE" as a keyword and routes to the analytical path
5. The agent processes this, potentially generating SQL: `SELECT AVG(salary) FROM employee_data`
6. The raw result and steps are combined into a comprehensive prompt
7. The appropriate LLM generates a natural language response: "The average salary in the employee dataset is $45,000"
8. This formatted answer is returned to the calling function and ultimately to the user
