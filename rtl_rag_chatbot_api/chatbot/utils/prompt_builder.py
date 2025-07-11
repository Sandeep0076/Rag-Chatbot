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
        question: str, answer: str, query_type: str = "unknown"
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
                "   - Long explanations or context\n"
                "   - Database terminology or technical details\n"
                "3. ALWAYS provide:\n"
                "   - Just the direct answer to the question\n"
                "   - The actual number/value with appropriate units\n"
                "   - Clean, simple formatting\n"
                "4. Format:\n"
                "   - For single values: 'The average shipping cost is $X.XX'\n"
                "   - For counts: 'There are X items'\n"
                "   - For totals: 'The total is $X.XX'\n"
                "   - For rows: 'Include headers and Use markdown tables for structured data: | Column | Column |\n"
                "   |--------|--------|\n"
                "   | Data   | Data   |\n"
                "   - Keep it simple and factual\n\n"
                "Provide only the direct, factual answer without additional commentary."
            )
        else:
            # For complex queries, provide detailed business analysis
            instructions = (
                "**COMPREHENSIVE RESPONSE INSTRUCTIONS:**\n"
                "1. You are a business analyst providing clean, professional answers\n"
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
                "   - Actionable insights and key findings for complex answers only\n"
                "   - Proper formatting using markdown tables when appropriate\n"
                "4. For large result sets (>50 rows):\n"
                "   - Provide summary with key insights first\n"
                "   - Show top 10-20 most relevant/important results in table format\n"
                "   - Include totals, averages, and aggregated statistics\n"
                "   - Group similar items for better readability\n"
                "5. Format guidelines:\n"
                "   - Use markdown tables for structured data: | Column | Column |\n"
                "   - Use bullet points for simple lists\n"
                "   - Include percentages, totals, and context where relevant\n"
                "Provide a clean, professional response with actual data that directly answers the question."
            )

        return (
            f"User Question: {question}\n\n"
            f"Query Results and Context: {truncated_answer}\n\n"
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
            "8. Write as if presenting to an executive or business stakeholder\n\n"
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
