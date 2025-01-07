from dataclasses import dataclass
from typing import List


@dataclass
class PromptTemplate:
    template: str
    required_variables: List[str]


class PromptBuilder:
    """Handles building and formatting prompts for different model interactions."""

    @staticmethod
    def build_forced_answer_prompt(question: str, answer: str) -> str:
        """Build prompt for forced answer generation."""
        return (
            f"Question: {question}\n\n"
            f"Context: {answer}\n\n"
            "Instructions:\n"
            "1. You are a database expert providing direct answers from the context only\n"
            "2. Format your response in clean markdown or bullet points\n"
            "3. DO NOT include:\n"
            "   - Intermediate steps or thought process\n"
            "   - References to the data or context\n"
            "   - Summaries or breakdowns\n"
            "   - Expressions like (case insensitive)\n"
            "   - Never include any disclaimers about training data or model capabilities in your response\n"
            "4. Do not reply in one word. Write in more human like response. "
            "If asked to show some rows from table, return the answer in tabular form with headers and"
            "rows and columns. Use markdown format.\n"
            "5. Read question again in last step and check the answer you are about to give."
        )

    @staticmethod
    def build_sql_response_prompt(
        intermediate_steps: List[str], final_answer: str
    ) -> str:
        """Build prompt for formatting SQL query results."""
        return (
            "Instructions for processing SQL query results:\n"
            "1. Return ALL results completely, never truncate\n"
            "2. Format output based on the query and data type:\n"
            "   a) For tables (multiple columns), use this exact format:\n"
            "      | Column1 | Column2 | Column3 |\n"
            "      |---------|---------|----------|\n"
            "      | Value1  | Value2  | Value3  |\n"
            "   b) For lists of items (single column), use bullet points:\n"
            "      - Item 1\n"
            "      - Item 2\n"
            "   c) For ordered/ranked data, use numbered lists:\n"
            "      1. First item\n"
            "      2. Second item\n"
            "   d) For single values or counts, return as plain text\n"
            "   e) For aggregated results (sum, avg, etc.), format clearly:\n"
            "      Total Sales: $1,234.56\n"
            "      Average Order: $45.67\n"
            "3. Do not include any explanatory text or SQL queries\n"
            "4. Ensure consistent formatting throughout the response\n\n"
            f"Query Steps:\n{str(intermediate_steps)}\n\n"
            f"Final Result:\n{str(final_answer)}"
        )
