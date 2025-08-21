# Fix line length violations and add missing blank lines
import json
import logging

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

Database Info: {database_info}
User Question: {user_question}

**DECISION TREE:**

1. **DIRECT_SUMMARY Questions** (Answer from database_info only, NO SQL needed):
   - "What's this file about?" → Summarize database structure and content
   - "Key insights" → Extract key statistics from database_info
   - "How many records/tables?" → Use counts from database_info
   - "What columns are available?" → List columns from database_info

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

5. **SIMPLE_AGGREGATION Questions** (No optimization needed):
   - Single values: count, sum, average, max, min
   - Usually return 1 result regardless of table size

**CONTEXT ANALYSIS INSTRUCTIONS:**
- Extract actual date ranges from database_info temporal columns
- Extract actual category counts from database_info statistics
- Extract table sizes and column information
- **CRITICAL**: Identify the actual table names from database_info.tables or database_info.table_names
- Make decisions based on REAL data characteristics, not assumptions

**RESPONSE RULES:**
- For DIRECT_SUMMARY: Provide answer directly from database_info
- For SQL-requiring queries: Generate ONLY the SQL statement, no explanations
- Always use case-insensitive comparisons for text searches
- Include aggregation strategy in the query when needed
- Never include disclaimers or technical explanations
- **CRITICAL**: Use the actual table names from database_info - NEVER use "your_table_name" or similar placeholders
- If multiple tables exist, choose the most appropriate one based on the question context
- **HARD CAP RULE**: The final SELECT must end with LIMIT 25.
If a LIMIT is present and >25, reduce it to 25.
If using LIMIT offset,count, cap count to 25 while preserving the offset.
Preserve OFFSET when present. Do NOT include markdown, comments, or code fences.

**TABLE NAME EXTRACTION:**
- Look for "table_names" array in database_info
- Look for "tables" array with "name" fields in database_info
- Use the actual table name(s) in your response
- If unsure which table to use, default to the first table in the list

Examples:
{examples}

**OUTPUT FORMAT:**
- If DIRECT_SUMMARY: Provide direct answer from database_info, NO SQL needed
- If SQL needed: Return ONLY the SQL statement, starting with SELECT
- No explanations, no multiple options, no markdown formatting
- Just the clean SQL statement
- **MUST use actual table names from database_info**
- **MUST end with LIMIT 25 (respecting the hard cap rule above)**
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
    }

    try:
        if not isinstance(database_info, dict):
            return context

        # Handle both direct tables list and nested structure
        tables = []
        if "tables" in database_info:
            tables = database_info["tables"]
        elif isinstance(database_info, dict) and "table_names" in database_info:
            # Create basic table structure from table names
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

            # Analyze temporal characteristics
            date_columns = []
            for col in columns:
                if isinstance(col, dict):
                    col_name = col.get("name", "")
                    col_type = str(col.get("type", "")).lower()
                    date_indicators = ["date", "time", "timestamp"]
                    if any(
                        date_indicator in col_type for date_indicator in date_indicators
                    ):
                        date_columns.append(col_name)

            if date_columns:
                # Try to estimate date span from sample data if available
                sample_data = table.get("top_rows", [])
                if sample_data and len(sample_data) > 1:
                    # Basic date span estimation (could be enhanced)
                    context["temporal_context"][table_name] = {
                        "date_columns": date_columns,
                        "estimated_span_days": max(
                            30, row_count // 10
                        ),  # Rough estimate
                        "granularity": "daily",
                    }

            # Analyze categorical characteristics
            text_columns = []
            for col in columns:
                if isinstance(col, dict):
                    col_name = col.get("name", "")
                    col_type = str(col.get("type", "")).lower()
                    text_types = ["text", "varchar", "char"]
                    if any(text_type in col_type for text_type in text_types):
                        text_columns.append(col_name)

            if text_columns:
                # Estimate category cardinality (rough approximation)
                for col_name in text_columns:
                    estimated_unique = min(
                        row_count, max(10, row_count // 20)
                    )  # Rough estimate
                    context["categorical_context"][
                        f"{table_name}.{col_name}"
                    ] = estimated_unique

            # Store table summary
            context["table_summaries"].append(
                {
                    "name": table_name,
                    "row_count": row_count,
                    "columns": len(columns),
                    "has_dates": bool(date_columns),
                    "has_categories": bool(text_columns),
                }
            )

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

    IMPORTANT: Also detect the language of the user's question and include it in the response.
    Common languages: English (en) and German (de).  also use accurate table name
    from database_info to make sql query
    Based on the ACTUAL database characteristics, return ONLY this JSON:

        "category": "DIRECT_SUMMARY|TIME_SERIES|CATEGORICAL_LISTING|FILTERED_SEARCH|SIMPLE_AGGREGATION",
        "needs_sql": true/false,
        "optimization_strategy": "none|temporal_aggregation|top_n_summary|preserve_detail",
        "reasoning": "brief explanation based on actual data characteristics",
        "language": "language_code (e.g., en, de)"
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
) -> str:
    """
    Enhance query based on classification and actual data context.

    Args:
        user_question: Original user question
        classification: Question classification result
        database_context: Database context analysis
        model_choice: The model choice to use for processing (default: "gpt_4o_mini")

    Returns:
        str: Enhanced query with appropriate optimization
    """

    category = classification.get("category", "SIMPLE_AGGREGATION")
    optimization = classification.get("optimization_strategy", "none")

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
        STRICT: The final SQL must include LIMIT 25.
        If there is an existing LIMIT larger than 25, reduce it to 25.
        If using LIMIT offset,count, cap the count to 25 and preserve the offset.
        Preserve OFFSET if present.
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
        STRICT: The final SQL must include LIMIT 25.
        If there is an existing LIMIT larger than 25, reduce it to 25.
        If using LIMIT offset,count, cap the count to 25 and preserve the offset.
        Preserve OFFSET if present.
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
        STRICT: The final SQL must include LIMIT 25.
        If there is an existing LIMIT larger than 25, reduce it to 25.
        If using LIMIT offset,count, cap the count to 25 and preserve the offset.
        Preserve OFFSET if present.
"""

    try:
        # Ensure prompt is within 120 characters per line for lint compliance
        prompt = (
            enhancement_prompt
            + "\nSQL OUTPUT RULES:\n"
            + "- Use the accurate table name from context.\n"
            + "- Return only raw SQL (no markdown or comments).\n"
            + "- Statement must end with LIMIT 25; if a LIMIT exists >25, reduce to 25;\n"
            + "  when OFFSET is present, preserve it.\n"
        )
        if model_choice.startswith("gemini"):
            from rtl_rag_chatbot_api.chatbot.gemini_handler import (
                get_gemini_non_rag_response,
            )

            result = get_gemini_non_rag_response(configs, prompt, model_choice)
        else:
            result = get_azure_non_rag_response(configs, prompt)
        return result
    except Exception as e:
        logging.error(f"Error enhancing query: {str(e)}")
        return user_question  # Fallback to original question


def format_question(
    database_info: dict, user_question: str, model_choice: str = "gpt_4o_mini"
) -> dict:
    """
    Enhanced question formatting with intelligent context awareness and optimization.

    Args:
        database_info: Information about the database structure (full database_summary)
        user_question: The original user question
        model_choice: The model choice to use for processing (default: "gpt_4o_mini")

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
                    user_question, classification, database_context, model_choice
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
        else:
            formatted_question = get_azure_non_rag_response(configs, formatted_prompt)
        return {
            "formatted_question": formatted_question,
            "needs_sql": True,
            "classification": {"category": "FALLBACK", "language": "en"},
        }
