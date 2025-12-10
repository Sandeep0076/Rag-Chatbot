# Fix line length violations and add missing blank lines
import json
import logging
import re
from typing import List

from configs.app_config import Config
from rtl_rag_chatbot_api.chatbot.chatbot_creator import get_azure_non_rag_response

configs = Config()

examples = [
    {
        "query": "Find employee Sandeep's salary",
        "answer": "SELECT * FROM employees WHERE LOWER(name) = LOWER('Sandeep')",
    },
    {
        "query": "Show me all employees named John",
        "answer": "SELECT * FROM employees WHERE LOWER(employee_name) LIKE LOWER('%john%')",
    },
    {
        "query": "What's the salary of employee SMITH",
        "answer": "SELECT salary FROM employees WHERE LOWER(last_name) = LOWER('SMITH')",
    },
    {
        "query": "Name cities with population more than 10000",
        "answer": "SELECT DISTINCT city FROM locations WHERE population > 10000",
    },
    {
        "query": "What are the key metrics or statistics in this Excel document?",
        "answer": (
            "SELECT COUNT(*) as total_records, "
            "AVG(numeric_column) as avg_value, "
            "MIN(numeric_column) as min_value, "
            "MAX(numeric_column) as max_value "
            "FROM table_name"
        ),
    },
    {
        "query": "Identify outliers",
        "answer": (
            "SELECT * FROM products WHERE sales > "
            "(SELECT AVG(sales) + 2 * (SELECT SQRT(AVG((sales - sub.avg_sales) * (sales - sub.avg_sales)))) "
            "FROM (SELECT AVG(sales) as avg_sales FROM products) sub)"
        ),
    },
    {
        "query": "how is Data Quality of document",
        "answer": (
            "SELECT column_name, "
            "COUNT(*) as total_rows, "
            "COUNT(column_name) as non_null_rows, "
            "COUNT(*) - COUNT(column_name) as null_rows "
            "FROM table_name GROUP BY 1"
        ),
    },
    # Examples for context-aware optimization
    {
        "query": "orders per day for 2 years",
        "answer": (
            "SELECT DATE_TRUNC('week', order_date) as week, "
            "COUNT(*) as weekly_orders "
            "FROM orders "
            "WHERE order_date >= DATE('now', '-2 years') "
            "GROUP BY week ORDER BY week"
        ),
    },
    {
        "query": "orders per day for 1 month",
        "answer": (
            "SELECT DATE(order_date) as order_day, "
            "COUNT(*) as daily_orders "
            "FROM orders "
            "WHERE order_date >= DATE('now', '-1 month') "
            "GROUP BY order_day ORDER BY order_day"
        ),
    },
    {
        "query": "employees named Mark",
        "answer": "SELECT * FROM employees WHERE LOWER(name) LIKE LOWER('%mark%')",
    },
    {
        "query": ("What is the employee name in the second row of the dataset?"),
        "answer": "SELECT employee_name FROM in_table LIMIT 1 OFFSET 1",
    },
    {
        "query": "Show me the first 3 rows of the dataset",
        "answer": "SELECT * FROM in_table LIMIT 3",
    },
]

# Enhanced prompt with intelligent context analysis
special_prompt = """
Analyze the given database_info and user_question with intelligent context awareness.

**CRITICAL: CONTEXT-AWARE ANALYSIS**
Before generating any response, analyze:
1. Can this be answered from database_info summary directly?
2. What's the actual data span/size in the database?
3. What type of query structure would this naturally produce?
4. **CRITICAL**: Extract and use the actual table names from database_info -
   NEVER use generic placeholders like "your_table_name"
5. **MULTI-FILE UNIFIED SCHEMA**: In multi-file chat, tables are unified and renamed
   to avoid conflicts using the pattern `{{filename}}_{{tablename}}`. Two extra columns
   exist in every unified table for attribution and filtering:
   - `_source_file_id` (original file identifier)
   - `_source_filename` (original filename)
   Use these columns when the user asks to filter by source file or to attribute results.

Database Info: {database_info}
User Question: {user_question}

**DECISION TREE:**

1. **DIRECT_SUMMARY Questions** (Answer from database_info only, NO SQL needed):
   - "What's this file about?" → Summarize database structure and content
   - "Key insights" → Extract key statistics from database_info
   - "How many records/tables?" → Use counts from database_info
   - "What columns/tables/files are available?" → List structure from database_info
   - "How many [records/columns/tables/files]?" → Use counts from database_info

2. **TIME_SERIES Questions** (Check actual date spans):
   - Analyze actual date ranges in database_info
   - If "per day" and date span > 90 days → Suggest weekly aggregation
   - If "per day" and date span > 365 days → Suggest monthly aggregation
   - If "per hour" and date span > 7 days → Suggest daily aggregation
   - Otherwise preserve original granularity

3. **CATEGORICAL_LISTING Questions** (Check actual category counts):
   - Analyze actual category cardinality from database_info
   - If categories > 50 → Show top 20-30 + summary of others
   - If categories < 50 → Show all with full details

4. **FILTERED_SEARCH Questions** (Always preserve detail):
   - "employees named X" → Full details regardless of table size
   - "orders from customer Y" → Complete results for specific filters
   - "top n records", "first 3 rows" → Complete results for specific filters with headers
   - Use case-insensitive comparisons
   - **In multi-file mode**: Support filters by `_source_file_id` or `_source_filename` when user
     specifies file-based constraints (e.g., "from Sales.xlsx only").

5. **SIMPLE_AGGREGATION Questions** (No optimization needed):
   - Single values: count, sum, average, max, min
   - Usually return 1 result regardless of table size

**CONTEXT ANALYSIS INSTRUCTIONS:**
- Extract actual date ranges from database_info temporal columns
- Extract actual category counts from database_info statistics
- Extract table sizes and column information
- **CRITICAL**: Identify the actual table names from database_info.tables or database_info.table_names
- **CRITICAL**: Extract ALL column names from database_info.tables[].columns[] and verify they exist
  before using them in SELECT, WHERE, GROUP BY, ORDER BY, or any SQL clause
- **CRITICAL - SAMPLE DATA USAGE**: database_info contains top_rows[] with ACTUAL sample data from the first 2 rows
  - ALWAYS examine top_rows[] to understand actual data formats and values
  - When filtering by categorical values (e.g., asset names, symbols), check what values exist in top_rows
- Use EXACT values from samples, not assumptions (e.g., use the value shown, not an abbreviation)
  - For timestamp columns, observe the actual format in samples (e.g., "2025-12-03 00:01:00")
  - Match user terms to actual sample values before constructing WHERE clauses
- In multi-file mode, prefer the unified table names in the `{{filename}}_{{tablename}}` format
  and remember `_source_file_id` and `_source_filename` are always available in unified tables
- Make decisions based on REAL data characteristics, not assumptions

**RESPONSE RULES:**
- For DIRECT_SUMMARY: Provide answer directly from database_info
- For SQL-requiring queries: Generate ONLY the SQL statement, no explanations
- Always use case-insensitive comparisons for text searches
- **CRITICAL NAME/TEXT MATCHING**: For name or text searches, ALWAYS use LIKE with wildcards:
  - Example: For "Aaron" use `LOWER(customer_name) LIKE '%aaron%'` NOT `= 'aaron'`
  - Example: For "Smith" use `LOWER(employee_name) LIKE '%smith%'` NOT `= 'smith'`
  - This ensures partial matches work (e.g., "Aaron" matches "Aaron Riggs")
  - Only use exact equality (=) for categorical fields like status codes, IDs, or exact enums
- **CRITICAL VALUE MATCHING FROM SAMPLES**:
  - FIRST, examine top_rows[] in database_info to see actual data values
  - When user asks about an entity/value, check what the samples actually show (full name, code, abbreviation)
  - Use the EXACT value format from samples (case-sensitive where appropriate)
  - Example: If samples show `asset = 'X'`, use that exact value and column
    (not a guessed abbreviation or another column)
  - For ambiguous terms, check BOTH possible columns in samples to determine correct mapping
- **TEMPORAL MATCHING FROM SAMPLES**:
  - Inspect sample timestamps to capture exact formatting (e.g., timezone offsets like "+00:00")
  - Use the exact string format seen in samples for equality filters (e.g., `timestamp = '2025-12-03 00:02:00+00:00'`)
  - If exact-time parsing is ambiguous, add a fallback day-level filter (e.g., BETWEEN day_start and day_end)
  - For precise times, try exact first; if that may fail, also include a narrow window (e.g., same minute or same day)
  - Avoid stripping timezone information; include it if present in samples
- Include aggregation strategy in the query when needed
- Never include disclaimers or technical explanations
- **CRITICAL**: Use the actual table names from database_info - NEVER use "your_table_name" or similar placeholders
- **CRITICAL**: Use ONLY column names that exist in database_info.tables[].columns[].name
  - If the user mentions "job", check if it's "job_type" or similar in the schema
  - Map user terms to actual column names (e.g., "employee" → "employee_name", "job" → "job_type")
  - NEVER invent or guess column names. Only use what exists in database_info
- If multiple tables exist, choose the most appropriate one based on the question context
- **HARD CAP RULE**: Apply LIMIT 25 intelligently based on query intent:
- For specific row requests (e.g., "second row", "row 5", "first 3 rows"):
  Use the exact LIMIT requested, do NOT override with LIMIT 25
- For general queries without specific row limits: Add LIMIT 25 if not present
- If a LIMIT is present and >25, reduce it to 25
- If using LIMIT offset,count, cap count to 25 while preserving the offset
- Preserve OFFSET when present
- Do NOT include markdown, comments, or code fences.

**TABLE NAME EXTRACTION:**
- Look for "table_names" array in database_info
- Look for "tables" array with "name" fields in database_info
- Use the actual table name(s) in your response
- In multi-file mode, table names follow `{{filename}}_{{tablename}}`; use these unified names
- If unsure which table to use, default to the first table in the list

Examples:
{examples}

**OUTPUT FORMAT:**
- If DIRECT_SUMMARY: Provide direct answer from database_info, NO SQL needed
- If SQL needed: Return ONLY the SQL statement, starting with SELECT
- No explanations, no multiple options, no markdown formatting
- Just the clean SQL statement
- **MUST use actual table names from database_info** (prefer unified names if multi-file)
- **MUST follow the intelligent LIMIT rules (respecting the hard cap rule above)**
"""


def analyze_database_context(database_info: dict) -> dict:
    """
    Extract actionable intelligence from database_summary for smart query planning.

    Args:
        database_info: Database structure information from file_info.json

    Returns:
        dict: Processed context including temporal, categorical, and numerical characteristics
    """
    context = {
        "temporal_context": {},
        "categorical_context": {},
        "numerical_context": {},
        "table_summaries": [],
        "has_unified_tables": False,
    }

    try:
        if not isinstance(database_info, dict):
            return context

        tables = []
        if "tables" in database_info:
            tables = database_info["tables"]
        elif isinstance(database_info, dict) and "table_names" in database_info:
            tables = [
                {"name": name, "columns": [], "row_count": 0}
                for name in database_info["table_names"]
            ]

        for table in tables:
            if not isinstance(table, dict):
                continue

            table_name = table.get("name", "unknown")
            columns = table.get("columns", [])
            row_count = table.get("row_count", 0)

            # Detect unified naming convention {filename}_{tablename}
            if "_" in table_name:
                context["has_unified_tables"] = True

            col_names = [c.get("name", "") for c in columns if isinstance(c, dict)]
            has_source_cols = (
                "_source_file_id" in col_names or "_source_filename" in col_names
            )

            # Summarize table
            context["table_summaries"].append(
                {
                    "name": table_name,
                    "row_count": row_count,
                    "columns": col_names,
                    "has_source_columns": has_source_cols,
                }
            )

            # Analyze temporal characteristics
            date_columns = [
                c for c in col_names if any(k in c.lower() for k in ["date", "time"])
            ]
            if date_columns:
                context["temporal_context"][table_name] = {"date_columns": date_columns}

    except Exception as e:
        logging.error(f"Error analyzing database context: {str(e)}")

    return context


def classify_question_intent(
    user_question: str, database_context: dict, model_choice: str = "gpt_4o_mini"
) -> dict:
    """
    Use LLM to classify question intent with database context awareness.

    Args:
        user_question: The user's original question
        database_context: Processed database context information
        model_choice: The model choice to use for processing (default: "gpt_4o_mini")

    Returns:
        dict: Classification result with category, optimization strategy, and reasoning
    """

    classification_prompt = f"""
    Analyze this question against the database context and classify it intelligently:

    Question: "{user_question}"

    Database Context:
    - Tables: {database_context.get('table_summaries', [])}
    - Temporal columns: {database_context.get('temporal_context', {})}
    - Categorical estimates: {database_context.get('categorical_context', {})}

    IMPORTANT: The database_info provided contains actual sample data (top_rows) showing real values.
    Always reference these samples to understand data formats and categorical values.

    Classify into ONE category:

    1. **DIRECT_SUMMARY**: Can be answered from database structure alone
       - Examples: "what's this about", "key insights", "summarize", "column names",  "attributes", "sample Data"
       - SET needs_sql: false (answered from metadata, no SQL execution needed)

    2. **TIME_SERIES**: Requesting data over time periods
       - Check actual date spans and decide aggregation level
       - Examples: "per day", "daily", "hourly", "over time"
       - SET needs_sql: true (requires SQL execution)

    3. **CATEGORICAL_LISTING**: Lists/breakdowns by categories
       - Check actual category counts for optimization decisions
       - Examples: "all customers", "by region", "breakdown by"
       - SET needs_sql: true (requires SQL execution)

    4. **FILTERED_SEARCH**: Searching for specific items
       - Always preserve full detail regardless of table size
       - Examples: "named John", "customer X", "where Y", "first 3 rows"
       - "top n records", "first 3 rows" → Complete results for specific filters
       - SET needs_sql: true (requires SQL execution)

    5. **SIMPLE_AGGREGATION**: Basic statistics
       - Usually single values, no optimization needed
       - Examples: "total", "count", "average", "sum"
       - SET needs_sql: true (requires SQL execution)

    CRITICAL: For DIRECT_SUMMARY, always set needs_sql to false since these can be
    answered from database metadata alone.

    Also use accurate table names from database_info to make the SQL query.
    Based on the ACTUAL database characteristics, return ONLY this JSON:

        "category": "DIRECT_SUMMARY|TIME_SERIES|CATEGORICAL_LISTING|FILTERED_SEARCH|SIMPLE_AGGREGATION",
        "needs_sql": true/false,
        "optimization_strategy": "none|temporal_aggregation|top_n_summary|preserve_detail",
        "reasoning": "brief explanation based on actual data characteristics"
    }}
    """

    try:
        if model_choice.startswith("gemini"):
            from rtl_rag_chatbot_api.chatbot.gemini_handler import (
                get_gemini_non_rag_response,
            )

            response = get_gemini_non_rag_response(
                configs, classification_prompt, model_choice
            )
        elif model_choice.startswith("claude"):
            from rtl_rag_chatbot_api.chatbot.anthropic_handler import (
                get_anthropic_non_rag_response,
            )

            response = get_anthropic_non_rag_response(
                configs, classification_prompt, model_choice
            )
        else:
            response = get_azure_non_rag_response(configs, classification_prompt)

        # Clean the response - remove markdown code blocks if present
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]  # Remove ```json
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]  # Remove ```
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]  # Remove trailing ```
        cleaned_response = cleaned_response.strip()

        # Try to parse JSON response
        parsed = None
        if cleaned_response.startswith("{"):
            try:
                parsed = json.loads(cleaned_response)
            except Exception:
                parsed = None
        if not parsed:
            # Fallback parsing if LLM doesn't return valid JSON
            logging.warning(f"Non-JSON classification response: {response}")
            parsed = {
                "category": "SIMPLE_AGGREGATION",
                "needs_sql": True,
                "optimization_strategy": "none",
                "reasoning": "Fallback classification due to parsing error",
                "language": "en",
            }

        # Override: if user asks for full/overall dataset, force SQL
        try:
            q = (user_question or "").lower()
            full_scope_terms = [
                "overall",
                "entire",
                "all",
                "full",
                "complete",
                # German equivalents
                "gesamt",
                "gesamte",
                "alle",
                "vollständige",
                "vollstaendige",
            ]
            if any(term in q for term in full_scope_terms):
                parsed["needs_sql"] = True
        except Exception:
            pass

        return parsed
    except Exception as e:
        logging.error(f"Error in question classification: {str(e)}")
        return {
            "category": "SIMPLE_AGGREGATION",
            "needs_sql": True,
            "optimization_strategy": "none",
            "reasoning": "Default classification due to error",
            "language": "en",
        }


def answer_from_database_summary(
    user_question: str,
    database_info: dict,
    language: str = "en",
    model_choice: str = "gpt_4o_mini",
) -> str:
    """
    Answer questions directly from database summary without SQL execution.

    Args:
        user_question: The user's question
        database_info: Database structure information
        language: The language to respond in (default: "en")
        model_choice: The model choice to use for processing (default: "gpt_4o_mini")

    Returns:
        str: Direct answer from database summary
    """

    summary_prompt = f"""
    Answer this question directly from the database information provided:

    Question: "{user_question}"

    Database Information: {json.dumps(database_info, indent=2, default=str)}

    Provide a clear, informative answer based ONLY on the database structure and statistics shown above.
    Focus on:
    - Actual answers to the question
    - Table structures and relationships
    - Record counts and data volumes

    Write in a business-friendly tone as if explaining to a stakeholder.
    Do not mention SQL, databases, or technical terms.

    IMPORTANT: Respond in {language.upper()} language to match the user's question language.
    """

    try:
        if model_choice.startswith("gemini"):
            from rtl_rag_chatbot_api.chatbot.gemini_handler import (
                get_gemini_non_rag_response,
            )

            return get_gemini_non_rag_response(configs, summary_prompt, model_choice)
        elif model_choice.startswith("claude"):
            from rtl_rag_chatbot_api.chatbot.anthropic_handler import (
                get_anthropic_non_rag_response,
            )

            return get_anthropic_non_rag_response(configs, summary_prompt, model_choice)
        else:
            return get_azure_non_rag_response(configs, summary_prompt)
    except Exception as e:
        logging.error(f"Error generating summary answer: {str(e)}")
        table_names = database_info.get("table_names", ["various topics"])
        return (
            f"I can see this data contains information about {table_names} "
            f"but cannot provide more details due to a processing error."
        )


def enhance_query_with_context(
    user_question: str,
    classification: dict,
    database_context: dict,
    model_choice: str = "gpt_4o_mini",
    database_info: dict = None,
) -> str:
    """
    Enhance query based on classification and actual data context.

    Args:
        user_question: Original user question
        classification: Question classification result
        database_context: Database context analysis
        model_choice: The model choice to use for processing (default: "gpt_4o_mini")
        database_info: Full database schema including column names

    Returns:
        str: Enhanced query with appropriate optimization
    """

    category = classification.get("category", "SIMPLE_AGGREGATION")
    optimization = classification.get("optimization_strategy", "none")

    # Extract table information and schema from database_info
    schema_info = ""
    if database_info and isinstance(database_info, dict):
        tables = database_info.get("tables", [])
        schema_parts = []
        for table in tables:
            # Include unified table naming hint and source columns if present
            tname = table.get("name", "")
            cols = table.get("columns", [])
            col_names = [c.get("name", "") for c in cols if isinstance(c, dict)]
            has_source_id = "_source_file_id" in col_names
            has_source_name = "_source_filename" in col_names
            suffix = ""
            if has_source_id or has_source_name:
                suffix = " (unified table; has _source_file_id, _source_filename)"
            schema_parts.append(
                f"Table: {tname}{suffix}\nColumns: {', '.join(col_names)}"
            )
        if schema_parts:
            schema_info = "\n\nSchema:\n" + "\n".join(schema_parts)

    # Extract table information from database context
    table_summaries = database_context.get("table_summaries", [])
    table_names = [
        table.get("name", "") for table in table_summaries if table.get("name")
    ]
    table_info_str = (
        f"Available tables: {', '.join(table_names)}"
        if table_names
        else "No table information available"
    )
    if schema_info:
        table_info_str = f"{table_info_str}\n\n{schema_info}\n\n"

    if category == "TIME_SERIES" and optimization == "temporal_aggregation":
        # Check actual temporal context
        temporal_info = database_context.get("temporal_context", {})

        enhancement_prompt = f"""
        Original question: "{user_question}"

        {table_info_str}
        Temporal context: {temporal_info}

        This is a time-series query that may return too many rows. Based on the actual date span in the data,
        enhance this query with appropriate temporal aggregation:

        - If asking for daily data over >3 months: Aggregate by week
        - If asking for daily data over >1 year: Aggregate by month
        - If asking for hourly data over >7 days: Aggregate by day

        Return ONLY the SQL statement, starting with SELECT.
        Use the actual table names from the available tables above.
        No explanations, no markdown formatting, just the SQL statement.
        STRICT: Apply LIMIT intelligently based on query intent:
        - For specific row requests (e.g., "second row", "row 5", "first 3 rows"): Use the exact LIMIT requested
        - For general queries: Add LIMIT 25 if not present
        - If there is an existing LIMIT larger than 25, reduce it to 25
        - If using LIMIT offset,count, cap count to 25 and preserve the offset
        - Preserve OFFSET if present
        """

    elif category == "CATEGORICAL_LISTING" and optimization == "top_n_summary":
        # Check actual categorical context
        categorical_info = database_context.get("categorical_context", {})

        enhancement_prompt = f"""
        Original question: "{user_question}"

        {table_info_str}
        Categorical context: {categorical_info}

        This query may return many categories. Based on the actual category counts in the data,
        enhance this query to show:
        - Top 20-30 most relevant items (by sales, count, or other meaningful metric)
        - Summary statistics for the remaining items

        Return ONLY the SQL statement, starting with SELECT.
        Use the actual table names from the available tables above.
        No explanations, no markdown formatting, just the SQL statement.
        STRICT: Apply LIMIT intelligently based on query intent:
        - For specific row requests (e.g., "second row", "row 5", "first 3 rows"): Use the exact LIMIT requested
        - For general queries: Add LIMIT 25 if not present
        - If there is an existing LIMIT larger than 25, reduce it to 25
        - If using LIMIT offset,count, cap the count to 25 and preserve the offset
        - Preserve OFFSET if present
        """

    else:
        # No enhancement needed or preserve detail
        enhancement_prompt = f"""
        Original question: "{user_question}"

        {table_info_str}

        Convert this to a SQL statement.
        Use case-insensitive comparisons for text searches.
        Preserve full detail as requested.
        Use the actual table names from the available tables above.

        Return ONLY the SQL statement, starting with SELECT.
        No explanations, no markdown formatting, just the SQL statement.
        STRICT: Apply LIMIT intelligently based on query intent:
        - For specific row requests (e.g., "second row", "row 5", "first 3 rows"): Use the exact LIMIT requested
        - For general queries: Add LIMIT 25 if not present
        - If there is an existing LIMIT larger than 25, reduce it to 25
        - If using LIMIT offset,count, cap the count to 25 and preserve the offset
        - Preserve OFFSET if present
"""

    try:
        # Ensure prompt is within 120 characters per line for lint compliance
        prompt = (
            enhancement_prompt
            + "\nSQL OUTPUT RULES:\n"
            + "- Use the accurate table name from context (prefer `{{filename}}_{{tablename}}` in unified mode).\n"
            + "- **CRITICAL**: Use ONLY column names from database_context that exist in the schema.\n"
            + "- Map user terms to actual column names (e.g., 'job' → 'job_type').\n"
            + "- NEVER invent or guess column names.\n"
            + "- **CRITICAL SAMPLE DATA**: Examine top_rows[] in schema to see actual values before filtering:\n"
            + "  - Match user terms (e.g., 'Bitcoin') to actual sample values\n"
            + "  - Use exact column names and value formats shown in samples\n"
            + "  - Don't assume value formats; verify against top_rows[]\n"
            + "- **TEMPORAL VALUES**: Observe sample timestamp formats (including timezone offsets)\n"
            + "  and use exact strings for equality filters.\n"
            + "  - Example: if sample shows '2025-12-03 00:02:00+00:00', use that exact format in WHERE.\n"
            + "  - If time format is ambiguous or too granular, add a safe fallback:\n"
            + "    * Also filter by the same day (BETWEEN day_start and day_end), or\n"
            + "    * Use a narrow window around the requested time (e.g., same minute) when precision is uncertain.\n"
            + "- **SOURCE FILTERS**: When user references a specific file, filter using `_source_filename`\n"
            + "  or `_source_file_id` as appropriate.\n"
            + "- **CRITICAL NAME MATCHING**: For name/text searches, use LIKE with wildcards:\n"
            + "  Example: LOWER(customer_name) LIKE '%aaron%' NOT = 'aaron'\n"
            + "- Return only raw SQL (no markdown or comments).\n"
            + "- Apply LIMIT intelligently: preserve specific row requests, add LIMIT 25 for general queries;\n"
            + "  when OFFSET is present, preserve it.\n"
        )
        if model_choice.startswith("gemini"):
            from rtl_rag_chatbot_api.chatbot.gemini_handler import (
                get_gemini_non_rag_response,
            )

            result = get_gemini_non_rag_response(configs, prompt, model_choice)
        elif model_choice.startswith("claude"):
            from rtl_rag_chatbot_api.chatbot.anthropic_handler import (
                get_anthropic_non_rag_response,
            )

            result = get_anthropic_non_rag_response(configs, prompt, model_choice)
        else:
            result = get_azure_non_rag_response(configs, prompt)
        return result
    except Exception as e:
        logging.error(f"Error enhancing query: {str(e)}")
        return user_question


def resolve_question_with_history(
    conversation_history: List[str],
    current_question: str,
    previous_formatted_question: str = "",
    previous_resolved_question: str = "",
) -> str:
    """
    Resolve contextual references in the current question by analyzing conversation history.
    Transforms questions that reference previous context into standalone questions.

    This function uses llm to intelligently determine if the current question
    needs to be modified based on conversation history. If the question is contextual
    (e.g., "Only for 2023", "What about profit?"), it merges the context to create
    a clear standalone question. If the question is already standalone, it returns
    it unchanged.

    Args:
        conversation_history: List of previous messages in the conversation.
                            Expected format: [msg1, msg2, ..., previous_question, previous_answer]
        current_question: The user's current/latest question that may contain contextual references
        previous_formatted_question: Optional. The formatted SQL or summary of the previous
                                     question (holistic view of constraints like filters/groupings)
        previous_resolved_question: Optional. The previously resolved question (fully
                                    contextualized) actually used for querying.

    Returns:
        str: A standalone question that includes necessary context from history,
             or the original question if no modification is needed

    Examples:
        History: ["Show me sales by region.", "Here's the sales data...", "Only for 2023."]
        Current: "Only for 2023."
        Returns: "Show me sales by region for 2023."

        History: ["List top 10 products by revenue.", "Here are the products...", "What about profit?"]
        Current: "What about profit?"
        Returns: "List top 10 products by profit instead of revenue."

        History: ["Show me all employees."]
        Current: "What is the average salary?"
        Returns: "What is the average salary?" (no modification needed)
    """
    # If there's no history, return the question as-is
    if not conversation_history or len(conversation_history) == 0:
        logging.info("No conversation history, returning original question")
        return current_question

    # If history exists but current question seems standalone (has clear SQL-like intent),
    # we still pass it to LLM for verification but with lower weight
    try:
        # Build conversation context for the LLM
        # Frontend history format (includes current question at the end):
        # - First turn: [Q1]
        # - Second turn: [Q1, A1, Q2]
        # - Third turn: [Q1, A1, Q2, A2, Q3]
        # Backend passes conversation_history WITHOUT the current question, and current_question separately.
        # Therefore to anchor to the previous user question (Qn), use:
        # - if len(conversation_history) == 0: no previous question
        # - if len(conversation_history) == 1: previous question = conversation_history[-1]
        # - if len(conversation_history) >= 2:
        #   previous question = conversation_history[-2],
        #   previous answer = conversation_history[-1]

        messages = conversation_history

        previous_user_question = ""
        previous_assistant_answer = ""

        if len(messages) == 0:
            previous_user_question = ""
            previous_assistant_answer = ""
        elif len(messages) == 1:
            previous_user_question = messages[-1]
            previous_assistant_answer = ""
        else:
            previous_user_question = messages[-2]
            previous_assistant_answer = messages[-1]

        # Prefer the previously resolved question as the anchor if provided
        if previous_resolved_question:
            previous_user_question = previous_resolved_question

        history_parts = []
        if previous_user_question:
            history_parts.append(
                f"Previous question (resolved when available): {previous_user_question}"
            )
        if previous_assistant_answer:
            history_parts.append(f"Previous answer: {previous_assistant_answer}")

        if previous_formatted_question:
            # Strip generic hard-cap LIMIT 25 from context while preserving other LIMIT values
            try:
                cleaned_prev_formatted = str(previous_formatted_question)
                # Remove patterns: "LIMIT 25", optional "OFFSET n", or "LIMIT n, 25"
                cleaned_prev_formatted = re.sub(
                    r"(?is)\s+LIMIT\s+25\b(?:\s+OFFSET\s+\d+)?",
                    "",
                    cleaned_prev_formatted,
                )
                cleaned_prev_formatted = re.sub(
                    r"(?is)\s+LIMIT\s+\d+\s*,\s*25\b",
                    "",
                    cleaned_prev_formatted,
                )
                # Tidy excess whitespace and trailing semicolons/spaces
                cleaned_prev_formatted = cleaned_prev_formatted.strip().rstrip(";")
            except Exception:
                cleaned_prev_formatted = str(previous_formatted_question)

            history_parts.append(
                "Previous formatted question (holistic context): "
                + cleaned_prev_formatted
            )

        history_context = "\n".join(history_parts)

        resolution_prompt = f"""You are a question resolution assistant for a tabular data chat system.
Your task is to determine if the current question EXPLICITLY references previous conversation context.

**Conversation History:**
{history_context}

**Current Question:**
{current_question}

**CRITICAL RULES:**
1. ONLY modify the question if it contains EXPLICIT contextual references like:
   - Continuations: "Only for 2023", "Just Q1", "also", "too"
   - Alternatives: "What about X?", "How about Y?", "instead"
   - Pronouns: "those", "these", "that", "it", "them"
   - Comparisons: "same for", "different from"
   - Follow-ups: "more details", "tell me more", "specifically"

2. If the question is COMPLETE and STANDALONE (has subject, verb, object), return it EXACTLY as is
   - Example: "How many books are there in Fiction genre?" → Return unchanged
   - Example: "Give me top 5 rows of the table" → Return unchanged
   - Example: "What is the average price?" → Return unchanged

3. ONLY add context when the question is INCOMPLETE without history
   - Example: "What about Non Fiction?" → Needs previous context
   - Example: "Only for 2023" → Needs previous context
   - Example: "Those entries" → Needs previous context

4. CONTEXT INHERITANCE (when modifying):
   - When the current question uses references (e.g., "in this", "those", "that"), INHERIT the scope,
     filters, and constraints from the Previous question/answer.
   - Examples of constraints to inherit: population filters (e.g., Unemployed), time windows,
     categories, groupings, joins, and previously established subsets.
   - Do NOT broaden the scope. Preserve the exact subset unless the new question explicitly changes it.
   - **CRITICAL**: Merge filters DIRECTLY into the new question. Do NOT use vague references like
     "out of these rows", "from the previous results", "those shown", etc.
   - Instead, explicitly restate the filters: e.g., "for Work Hours Per Day > 6 AND stress_level > 6"

**Output Format:**
Return ONLY the question (modified or unchanged). No explanations, no markdown, no prefixes.

**Examples:**

Example 1 - MODIFY (has contextual reference "Only"):
History: Previous question: Show me sales by region.
Current: Only for 2023.
Output: Show me sales by region for 2023.

Example 2 - MODIFY (has contextual reference "What about"):
History: Previous question: How many books in Fiction genre?
Current: What about Non Fiction?
Output: How many books are there in Non Fiction genre?

Example 3 - KEEP UNCHANGED (complete standalone question):
History: Previous question: Show me sales by region.
Current: How many books are there in Fiction genre?
Output: How many books are there in Fiction genre?

Example 4 - KEEP UNCHANGED (complete standalone question):
History: Previous question: What's this file about?
Current: Give me top 5 rows of the table.
Output: Give me top 5 rows of the table.

Example 5 - MODIFY (has pronoun "these entries"):
History: Previous question: Search for "skills".
Current: Now look at these entries in detail.
Output: From the entries containing 'skills', show details about those that discuss Claude Skills.

Example 6 - MODIFY (inherit prior filter "Unemployed"):
History: Previous question: Give me the count of different social media platform names among Unemployed users.
Current: In this what is the average stress level for each platform
Output: What is the average stress level for each social platform among Unemployed users?

Example 7 - MODIFY (inherit prior filter and merge with new condition):
History: Previous question (resolved when available): Give data for Work Hours Per Day more than 6 hours
Current: Out of these how many have stress level more than 6
Output: Give data for Work Hours Per Day more than 6 hours who have stress_level greater than 6

Now resolve the current question based on the conversation history provided above.
"""

        resolved_question = get_azure_non_rag_response(
            configs, resolution_prompt, model_choice="gpt_5_mini"
        )

        # Clean up the response (remove any markdown, quotes, or extra whitespace)
        resolved_question = resolved_question.strip()
        if resolved_question.startswith('"') and resolved_question.endswith('"'):
            resolved_question = resolved_question[1:-1]
        if resolved_question.startswith("'") and resolved_question.endswith("'"):
            resolved_question = resolved_question[1:-1]
        logging.info(3 * "--------------------------------")
        logging.info(f"Previous user question: {previous_user_question}")
        logging.info(f"Original question: {current_question}")
        logging.info(f"Resolved question: {resolved_question}")
        logging.info(3 * "--------------------------------")

        return resolved_question

    except Exception as e:
        logging.error(f"Error resolving question with history: {str(e)}")
        # Fallback to original question on error
        return current_question


def format_question(
    database_info: dict,
    user_question: str,
    model_choice: str = "gpt_5_mini",
    user_language: str | None = None,
) -> dict:
    """
    Enhanced question formatting with intelligent context awareness and optimization.

    Args:
        database_info: Information about the database structure (full database_summary)
        user_question: The original user question
        model_choice: The model choice to use for processing (default: "gpt_5_mini")

    Returns:
        dict: Contains 'formatted_question' and 'needs_sql' flag
    """

    try:
        # Step 1: Analyze database context
        database_context = analyze_database_context(database_info)
        logging.info(f"Database context analysis: {database_context}")

        # Step 2: Classify question intent
        classification = classify_question_intent(
            user_question, database_context, model_choice
        )
        logging.info(f"Question classification: {classification}")

        # Step 3: Route based on classification
        needs_sql = classification.get("needs_sql", True)
        # If caller provided an explicit user language (e.g., via lingua),
        # override any language guess coming from the LLM classification.
        if user_language:
            classification["language"] = user_language

        if not needs_sql:
            # Answer directly from database summary (includes DIRECT_SUMMARY category)
            logging.info("Answering directly from database summary")
            language = classification.get("language", "en")
            formatted_question = answer_from_database_summary(
                user_question, database_info, language, model_choice
            )
            return {
                "formatted_question": formatted_question,
                "needs_sql": False,
                "classification": classification,
            }
        else:
            # Need SQL execution with potential optimization
            optimization_strategy = classification.get("optimization_strategy")
            logging.info(
                f"Generating optimized SQL query with strategy: {optimization_strategy}"
            )

            if classification.get("optimization_strategy") != "none":
                # Apply intelligent optimization
                formatted_question = enhance_query_with_context(
                    user_question,
                    classification,
                    database_context,
                    model_choice,
                    database_info,
                )
            else:
                # Standard query formatting
                # CRITICAL: CONTEXT-AWARE ANALYSIS prompt
                formatted_prompt = special_prompt.format(
                    database_info=database_info,
                    user_question=user_question,
                    examples=examples,
                )

                # Use the appropriate model based on model_choice
                if model_choice.startswith("gemini"):
                    from rtl_rag_chatbot_api.chatbot.gemini_handler import (
                        get_gemini_non_rag_response,
                    )

                    formatted_question = get_gemini_non_rag_response(
                        configs, formatted_prompt, model_choice
                    )
                elif model_choice.startswith("claude"):
                    from rtl_rag_chatbot_api.chatbot.anthropic_handler import (
                        get_anthropic_non_rag_response,
                    )

                    formatted_question = get_anthropic_non_rag_response(
                        configs, formatted_prompt, model_choice
                    )
                else:
                    formatted_question = get_azure_non_rag_response(
                        configs=configs, query=formatted_prompt
                    )

            return {
                "formatted_question": formatted_question,
                "needs_sql": True,
                "classification": classification,
            }

    except Exception as e:
        logging.error(f"Error in enhanced format_question: {str(e)}")
        # Fallback to basic formatting
        formatted_prompt = special_prompt.format(
            database_info=database_info,
            user_question=user_question,
            examples=examples,
        )
        if model_choice.startswith("gemini"):
            from rtl_rag_chatbot_api.chatbot.gemini_handler import (
                get_gemini_non_rag_response,
            )

            formatted_question = get_gemini_non_rag_response(
                configs, formatted_prompt, model_choice
            )
        elif model_choice.startswith("claude"):
            from rtl_rag_chatbot_api.chatbot.anthropic_handler import (
                get_anthropic_non_rag_response,
            )

            formatted_question = get_anthropic_non_rag_response(
                configs, formatted_prompt, model_choice
            )
        else:
            formatted_question = get_azure_non_rag_response(configs, formatted_prompt)
        return {
            "formatted_question": formatted_question,
            "needs_sql": True,
            "classification": {
                "category": "FALLBACK",
                # Respect caller-provided user_language if available; otherwise default to 'en'
                "language": user_language or "en",
            },
        }
