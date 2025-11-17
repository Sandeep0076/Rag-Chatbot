@# Chat with Tabular Data (CSV/Excel)

This document outlines the workflow and implementation details for chatting with tabular data (CSV/Excel files) in our RAG application.

## Core Components

### TabularDataHandler
- Main class handling database operations and queries
- Manages SQL query generation and execution
- Handles response formatting and visualization requests
- Integrates with LLM models for natural language processing
- **NEW**: Supports conversation history for contextual follow-up questions
- **NEW**: Supports configurable temperature parameter for fine-tuning response creativity

### PrepareSQLFromTabularData
- Handles conversion of CSV/Excel to SQLite
- Performs data type inference
- Creates optimized database schema
- Manages data insertion and indexing

### FileHandler
- Manages file operations and preprocessing
- Handles file encryption and security
- Performs file hash calculation and validation
- Manages temporary file cleanup

## Features and Capabilities

### Data Format Support
- CSV files
- Excel files (.xlsx)
- Existing SQLite databases
- Structured tabular data

### Query Capabilities
- Natural language to SQL conversion
- Contextual follow-up questions with conversation history
- Complex aggregations and calculations
- Multi-table joins
- Data visualization queries
- Statistical analysis

## Overview

The system allows users to upload CSV or Excel files and interact with the data using natural language queries. The workflow involves:
1. File upload and preprocessing
2. SQLite database creation
3. Query processing and response generation
4. Optional data visualization

## Detailed Workflow

### 1. File Upload and Processing
- User uploads a CSV/Excel file
- System calculates file hash to check for existing processed data
- File is encrypted for security
- If file exists:
  - Use existing SQLite database and metadata
- If new file:
  - Process file and create new SQLite database
  - Store metadata and file information

### 2. Database Preparation
- CSV/Excel data is converted to SQLite database using `PrepareSQLFromTabularData`
- Process includes:
  - Data type inference for columns
  - Table creation with appropriate schema
  - Data insertion with proper encoding
  - Index creation for optimization
  - Table metadata storage
- Database is stored with a unique identifier based on file hash

### 3. Chat Processing

#### Query Types and Response Generation

1. **SQL-Based RAG Responses (Primary Method)**
   - User query is transformed into SQL using LangChain
   - SQL is executed against the SQLite database
   - Results are formatted into natural language using the chosen LLM

2. **Non-RAG Forced Responses (Special Cases)**
   - Used when:
     - Query requires general knowledge
     - Data visualization is requested
     - Query is about database structure/metadata
     - Error handling or clarification needed


### 3. Data Visualization
- Supports generation of:
  - Charts (bar, line, pie)
  - Graphs
  - Statistical summaries
- Visualization requests are handled by specialized prompts
- Charts are generated using Python plotting libraries

## Technical Implementation

### Database Management
```python
SQLite database with SQLAlchemy ORM
- Automatic schema inference
- Optimized indexing
- Connection pooling
```

### Query Processing Pipeline

**Enhanced with Conversation History Support**

1. **History Resolution** (NEW)
   - Analyzes conversation history to detect contextual references
   - Transforms contextual questions into standalone queries
   - Uses gpt-4o-mini for fast, cost-effective processing
   - Preserves original question if no history context is needed

2. **Database Context Enrichment**
   - Adds table schema and metadata to the question
   - Optimizes query structure based on data characteristics

3. **SQL Generation**
   - Converts natural language to SQL using LangChain
   - Applies intelligent query optimization

4. **Database Query Execution**
   - Executes SQL against SQLite database
   - Handles errors and edge cases

5. **Result Formatting**
   - Formats results into natural language responses
   - Applies visualization if requested

### Conversation History Feature

**Overview**

The tabular data chat now maintains conversation context, allowing users to ask follow-up questions without repeating information. The system intelligently resolves contextual references to create clear, standalone questions.

**How It Works**

1. User sends conversation array: `["Previous Q", "Previous A", ..., "Current Q"]`
2. System extracts history and current question
3. `resolve_question_with_history()` analyzes if current question references history
4. If contextual, creates a merged standalone question
5. Proceeds with normal SQL generation and execution

**Processing Model**

- History resolution always uses **gpt-4o-mini** (regardless of query model)
- Fast processing with minimal cost impact
- Happens BEFORE database context enrichment
- Falls back to original question on any errors

**Examples**

**Example 1: Filter Continuation**
```
History:
  User: "Show me sales by region."
  Assistant: "Here's the table of sales totals by region."
  User: "Only for 2023."

Resolved Question: "Show me sales by region for 2023."
```

**Example 2: Column Reference Change**
```
History:
  User: "List top 10 products by revenue."
  Assistant: "Here's the top 10 products by total revenue."
  User: "What about profit?"

Resolved Question: "List top 10 products by profit instead of revenue."
```

**Example 3: Contextual Refinement**
```
History:
  User: "Search markdown_content for 'skills' and return IDs and titles."
  Assistant: "Here is a list of IDs and titles containing 'skills'..."
  User: "Now look at these entries and output which deal with Claude Skills."

Resolved Question: "From the entries whose markdown_content contains 'skills',
identify and list those that specifically discuss Claude Skills,
including their IDs and titles."
```

**Example 4: Standalone Question (No Modification)**
```
History:
  User: "Show me all employees."
  Assistant: "Here are all employee records..."
  User: "What is the average salary across all departments?"

Resolved Question: "What is the average salary across all departments?"
(No modification needed - question is already standalone)
```

**API Usage with History**

```json
{
  "text": [
    "Show me sales by region.",
    "Here's the data showing sales totals by region...",
    "Only for 2023."
  ],
  "file_id": "csv-file-id",
  "model_choice": "gpt_4o_mini",
  "user_id": "user123"
}
```

**Benefits**

- Natural conversational flow like PDF chat
- Reduces need to repeat context in follow-up questions
- Maintains accuracy by resolving ambiguity before SQL generation
- No API changes required (uses existing `text` array parameter)

### Response Types

1. **RAG Responses**
   - Generated using database content
   - Include relevant data context
   - Maintain data accuracy

2. **Forced/Non-RAG Responses**
   - Used for:
     - System messages
     - Error handling
     - General knowledge
     - Data visualization requests

## Limitations and Best Practices

### File Limitations
- Maximum file size: 10MB
- Supported formats: CSV, Excel (.xlsx, .xls. , .db)

### Query Best Practices
- Be specific with column names
- Specify desired aggregations clearly
- Include time ranges for temporal data
- Mention specific visualization types if needed

### Security
- All tabular data is encrypted at rest
- Access control through user authentication
- Secure database connections
- Regular cleanup of temporary files

## Error Handling

Common error scenarios and their handling:
1. File too large → Size limit message
2. Invalid format → Format support message
3. Query timeout → Timeout notification
4. Data type mismatch → Data type guidance
5. Invalid SQL → Query reformulation request

## Temperature Parameter Support

### Overview
The tabular data handler now supports configurable temperature values to control the creativity and style of responses when processing natural language queries against CSV/Excel data.

### Temperature Behavior for Tabular Data

**Default Values:**
- **OpenAI models (GPT series)**: `0.5` - Optimized for accurate data interpretation
- **Gemini models**: `0.8` - Balanced for natural language explanations

**Recommended Usage:**

- **Low Temperature (0.1-0.3)**: For precise data queries requiring exact calculations
  ```json
  {
    "text": ["What is the exact sum of revenue for Q3 2024?"],
    "file_id": "csv-file-id",
    "model_choice": "gpt_4o_mini",
    "temperature": 0.2
  }
  ```

- **Medium Temperature (0.4-0.7)**: For balanced data analysis with explanations
  ```json
  {
    "text": ["Analyze the sales trends and explain the patterns"],
    "file_id": "csv-file-id",
    "model_choice": "gpt_4o_mini",
    "temperature": 0.5
  }
  ```

- **Higher Temperature (0.8-1.2)**: For creative data insights and storytelling
  ```json
  {
    "text": ["Tell a story about what this sales data reveals about customer behavior"],
    "file_id": "csv-file-id",
    "model_choice": "gemini-2.5-flash",
    "temperature": 1.0
  }
  ```

## Usage Examples

### Basic Query Example
```python
query = "What were the total sales in 2024?"

query = "Show me the top 5 products by revenue, including their growth rate"
```
### Visualization Example
```python
# Request for data visualization
query = "Create a bar chart showing monthly sales trends"

# Returns:
# - Generated chart
```
