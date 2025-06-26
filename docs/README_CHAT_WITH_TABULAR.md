@# Chat with Tabular Data (CSV/Excel)

This document outlines the workflow and implementation details for chatting with tabular data (CSV/Excel files) in our RAG application.

## Core Components

### TabularDataHandler
- Main class handling database operations and queries
- Manages SQL query generation and execution
- Handles response formatting and visualization requests
- Integrates with LLM models for natural language processing
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
1. Query preprocessing and validation
2. SQL generation using LangChain
3. Database query execution
4. Result formatting and response generation

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
    "model_choice": "gemini-flash",
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
