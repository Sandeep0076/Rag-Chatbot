import logging
from dataclasses import dataclass
from typing import List


@dataclass
class PromptTemplate:
    template: str
    required_variables: List[str]


class PromptBuilder:
    """Handles building and formatting prompts for different model interactions."""

    @staticmethod
    def build_forced_answer_prompt(
        question: str, answer: str, query_type: str = "unknown", language: str = "en"
    ) -> str:
        """Build prompt for forced answer generation with appropriate verbosity based on query type."""
        # Safety check: if context is extremely large, truncate it
        max_context_length = 200000  # ~50k tokens worth of characters

        if len(answer) > max_context_length:
            # Keep the end of the answer as it's more likely to contain the final result
            truncated_answer = (
                "... [Previous content truncated] ...\n" + answer[-max_context_length:]
            )
            logging.warning(
                f"Context truncated from {len(answer)} to {len(truncated_answer)} characters"
            )
        else:
            truncated_answer = answer

        # Determine response style based on query type
        if query_type == "SIMPLE_AGGREGATION":
            # For simple aggregations, be concise and direct
            instructions = (
                "**CONCISE RESPONSE INSTRUCTIONS:**\n"
                "1. You are providing a direct, factual answer\n"
                "2. NEVER include:\n"
                "   - SQL queries, code, or technical syntax\n"
                "   - Business advice, insights, or recommendations\n"
                "   - Long explanations or context beyond what is present in 'FINAL ANSWER'\n"
                "   - Database terminology or technical details\n"
                "3. ALWAYS provide:\n"
                "   - The final answer as stated under 'FINAL ANSWER'\n"
                "     (include its brief explanation if present; do NOT drop it)\n"
                "   - The actual number/value with appropriate units\n"
                "   - Clean, simple formatting\n"
                "4. Format:\n"
                "   - For single values: 'The average shipping cost is $X.XX'\n"
                "   - For counts: 'There are X items'\n"
                "   - For totals: 'The total is $X.XX'\n"
                "   - For rows: 'Include headers and Use markdown tables for structured data: | Column | Column |\n"
                "   |--------|--------|\n"
                "   | Data   | Data   |\n"
                "   - Keep it simple and factual\n"
                "   - **CRITICAL**: If column headers are provided, use EXACTLY those headers\n"
                "   - Look for 'Column Headers:' in the context and use those exact names\n"
                "5. **IMPORTANT**: After formatting your response, count the actual data rows "
                "(excluding headers and separators)\n"
                "   - If you count EXACTLY 25 data rows, append this note at the bottom:\n"
                "     *Results limited to 25 rows for readability.*\n"
                "   - If you count fewer than 25 rows, do NOT add this note\n"
                "   - Only add the note when you are certain there are exactly 25 data rows\n\n"
                "6. Optionally append ONE concise clarification sentence only if it improves clarity.\n"
                "   If not needed, do not append anything.\n"
            )
        elif query_type == "FILTERED_SEARCH":
            # For filtered search queries, ALWAYS show complete data when ≤25 rows
            instructions = (
                "**FILTERED_SEARCH RESPONSE INSTRUCTIONS:**\n"
                "1. You are providing complete search results for specific criteria\n"
                "2. NEVER include:\n"
                "   - SQL queries, code, or technical syntax\n"
                "   - Business advice, insights, or recommendations\n"
                "   - Database terminology or technical details\n"
                "   - Summaries or partial data when full results are available\n"
                "3. ALWAYS provide:\n"
                "   - COMPLETE data for all matching records (up to 25 rows)\n"
                "   - Full table with headers and all data rows\n"
                "   - No summarization or data reduction\n"
                "   - Clean, professional formatting using markdown tables\n"
                "4. Format:\n"
                "   - Use markdown tables: | Column | Column |\n"
                "   |--------|--------|\n"
                "   | Data   | Data   |\n"
                "   - Include ALL rows from the search results\n"
                "   - Show complete data without truncation\n"
                "   - **CRITICAL**: If column headers are provided, use EXACTLY those headers\n"
                "   - Look for 'Column Headers:' in the context and use those exact names\n"
                "5. **MANDATORY**: Show ALL data rows returned by the search query\n"
                "   - Do not summarize or reduce the data\n"
                "   - Display every single row that matches the search criteria\n"
                "   - If the context shows 25 or fewer rows, show ALL of them\n\n"
                "6. **IMPORTANT**: After formatting your response, count the actual data rows "
                "(excluding headers and separators)\n"
                "   - If you count EXACTLY 25 data rows, append this note at the bottom:\n"
                "     *Note: Results are limited to 25 rows for readability.*\n"
                "   - If you count fewer than 25 rows, do NOT add this note\n"
                "   - Only add the note when you are certain there are exactly 25 data rows\n\n"
                "Provide the complete search results in a clean table format with ALL data rows."
            )
        else:
            # For complex queries, provide detailed business analysis
            instructions = (
                "**COMPREHENSIVE RESPONSE INSTRUCTIONS:**\n"
                "1. You are a business analyst providing clean, professional answers\n"
                "   - If rows are given as output show the whole data\n"
                "2. NEVER include:\n"
                "   - SQL queries, code, or technical syntax of any kind\n"
                "   - Database terminology (table, query, SQL, database, etc.)\n"
                "   - Step-by-step technical processes or explanations\n"
                "   - References to how data was retrieved or processed\n"
                "   - Intermediate steps or thought processes\n"
                "   - Technical error messages or system responses\n"
                "3. ALWAYS provide:\n"
                "   - Clean, formatted actual data/results only\n"
                "   - Business-friendly language and explanations\n"
                "   - Proper formatting using markdown tables when appropriate\n"
                "4. For large result sets (>50 rows):\n"
                "   - Provide summary with key insights first\n"
                "   - Show top 25 most relevant/important results in table format\n"
                "5. For large result sets (<26 rows):\n"
                "   - show the whole data including headers in markdown tabular form\n"
                "6. **IMPORTANT**: After formatting your response, count the actual data rows "
                "(excluding headers and separators)\n"
                "   - If you count EXACTLY 25 data rows, append this note at the bottom:\n"
                "     *Note: Results are limited to 25 rows for readability.*\n"
                "   - If you count fewer than 25 rows, do NOT add this note\n"
                "   - Only add the note when you are certain there are exactly 25 data rows\n"
                "7. When showing a subset of rows (>50 rows):\n"
                "   - Begin with an explicit statement like: 'Showing the first N rows (out of total available)'.\n"
                "   - If only a sample is displayed, include: '*Note: Showing a sample of the data.*'\n"
                "   - Do NOT state an incorrect number of rows.\n"
                "     If you cannot determine N exactly, say 'Showing a subset of rows'.\n"
                "8. Format guidelines:\n"
                "   - Use markdown tables for structured data: | Column | Column |\n"
                "   |--------|--------|\n"
                "   | Data   | Data   |\n"
                "   - Use bullet points for simple lists\n"
                "   - Include percentages, totals, and context where relevant\n"
                "   - Provide clean, professional responses with actual data\n"
                "9. **CRITICAL**: If column headers are provided in the context, use EXACTLY those headers\n"
                "   - Do not invent or modify column names\n"
                "   - Use the exact column names as provided\n"
                "   - Ensure table headers match the data structure exactly\n"
                "   - Look for 'Column Headers:' in the context and use those exact names"
            )

        # Add language instruction
        language_instruction = (
            f"IMPORTANT: Respond in {language.upper()} "
            "language to match the user's question language."
        )

        # Check if column headers are mentioned in the context
        header_emphasis = ""
        if "Column Headers:" in truncated_answer:
            header_emphasis = (
                "\n\n**IMPORTANT: COLUMN HEADERS DETECTED**\n"
                "The context contains specific column headers. Use these EXACT names as table headers.\n"
                "Do not invent, modify, or abbreviate column names.\n\n"
            )

        final_answer_priority = (
            "**CRITICAL CONTEXT USAGE RULES:**\n"
            "1. First, read the 'User Question' and analyze the 'FINAL ANSWER' section deeply.\n"
            "2. If and only if the 'FINAL ANSWER' does not fully answer the question (incomplete/ambiguous),\n"
            "   then consult 'INTERMEDIATE_STEPS' to complete the answer.\n"
            "3. If the 'FINAL ANSWER' answers the question but can be made clearer, you may append ONE concise\n"
            "   clarification sentence to improve understanding. If not needed, do not add anything.\n"
            "4. When presenting tabular data, explicitly count the data rows (exclude headers/separators).\n"
            "   Append the note '*Results limited to 25 rows for readability.*' ONLY if the count is EXACTLY 25.\n"
            "   If the count is not exactly 25, do not include this note.\n"
            "5. After presenting any table, determine whether the question expects a direct conclusion.\n"
            "   If yes, append ONE explicit sentence that directly answers (e.g., name the entity, yes/no, trend).\n"
            "   If not necessary, do not add anything.\n"
            "6. For questions about relationship/correlation, state clearly whether the relationship appears\n"
            "   positive, negative, or no clear relationship. Include approximate strength ONLY if present\n"
            "   in the provided context. Do not invent new metrics.\n\n"
        )

        return (
            f"User Question: {question}\n\n"
            f"Query Results and Context:\n"
            f"{truncated_answer}\n\n"
            f"{header_emphasis}"
            f"{language_instruction}\n\n"
            f"{final_answer_priority}"
            f"{instructions}"
        )

    @staticmethod
    def build_sql_response_prompt(
        intermediate_steps: List[str], final_answer: str
    ) -> str:
        """Build prompt for formatting SQL query results with business focus."""
        return (
            "Transform the following technical query results into a clean business response:\n\n"
            f"Technical Process: {str(intermediate_steps)}\n\n"
            f"Raw Results: {str(final_answer)}\n\n"
            "**TRANSFORMATION RULES:**\n"
            "1. Remove all technical details, SQL references, and database terminology\n"
            "2. Present only the business-relevant data and insights\n"
            "3. Format as professional tables using markdown when appropriate\n"
            "4. Include summaries, totals, and key insights\n"
            "5. Use business-friendly language throughout\n"
            "6. For large datasets, provide intelligent summaries with top results\n"
            "7. Focus on actionable information and business value\n"
            "8. Write as if presenting to an executive or business stakeholder\n"
            "9. **CRITICAL**: If column headers are provided, use EXACTLY those headers\n\n"
            "Provide a clean, executive-ready response with actual data and insights."
        )

    @staticmethod
    def contains_sql_content(text: str) -> bool:
        """
        Check if text contains SQL-like content that should be filtered out.

        Args:
            text: The text to check for SQL content

        Returns:
            bool: True if SQL content is detected
        """
        if not text:
            return False

        text_upper = text.upper()

        # SQL keywords and patterns that indicate technical content
        sql_indicators = [
            "SELECT",
            "FROM",
            "WHERE",
            "GROUP BY",
            "ORDER BY",
            "HAVING",
            "COUNT(*)",
            "COUNT(",
            "SUM(",
            "AVG(",
            "MAX(",
            "MIN(",
            "SQL QUERY",
            "DATABASE QUERY",
            "EXECUTE",
            "QUERY WILL",
            "```SQL",
            "CREATE TABLE",
            "INSERT INTO",
            "UPDATE",
            "DELETE FROM",
            "INNER JOIN",
            "LEFT JOIN",
            "RIGHT JOIN",
            "FULL JOIN",
            "DISTINCT",
            "LIMIT",
            "OFFSET",
            "AS (",
            ") AS",
            "CASE WHEN",
            "END AS",
            "UNION",
            "INTERSECT",
        ]

        return any(indicator in text_upper for indicator in sql_indicators)

    @staticmethod
    def estimate_result_size(text: str) -> str:
        """
        Estimate if the result set is large based on content analysis.

        Args:
            text: The text content to analyze

        Returns:
            str: Size estimation ('small', 'medium', 'large')
        """
        if not text:
            return "small"

        # Count potential data rows (lines that look like data)
        lines = text.split("\n")
        data_lines = 0

        for line in lines:
            line = line.strip()
            # Skip empty lines and obvious non-data lines
            if (
                not line
                or line.startswith("Action:")
                or line.startswith("Observation:")
            ):
                continue
            # Count lines that contain data patterns (numbers, commas, typical data separators)
            if any(char in line for char in [",", "|", "\t"]) or any(
                char.isdigit() for char in line
            ):
                data_lines += 1

        if data_lines > 100:
            return "large"
        elif data_lines > 30:
            return "medium"
        else:
            return "small"
