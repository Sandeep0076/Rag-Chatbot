from configs.app_config import Config
from rtl_rag_chatbot_api.chatbot.chatbot_creator import get_azure_non_rag_response

configs = Config()

examples = [
    {
        "query": "Find employee Sandeep's salary",
        "answer": (
            "Find employee details where the name matches 'Sandeep' "
            "case-insensitively (LOWER(name) = LOWER('Sandeep'))"
        ),
    },
    {
        "query": "Show me all employees  named John",
        "answer": "List all people where LOWER(employee_name) LIKE LOWER('%john%')",
    },
    {
        "query": "What's the salary of employee SMITH",
        "answer": "Show salary details where LOWER(last_name) = LOWER('SMITH')",
    },
    {
        "query": "Name cities with population more than 10000",
        "answer": "List all distinct cities that have a population greater than 10,000.",
    },
    {
        "query": "What are the key metrics or statistics in this Excel document?",
        "answer": (
            "How many total records are in the table? + "
            "What are all the distinct values in the [x] column? + "
            "What is the average, minimum, maximum, and sum of the [x] column? + "
            "What is the total [x] for each [y] + "
            "What are the top 5 [Customers] by [Order Value]?"
        ),
    },
    {
        "query": "Identify outliers",
        "answer": "Which [Products] have sales more than two standard deviations above the mean?",
    },
    {
        "query": "how is Data Quality of document ",
        "answer": "Is there any missing or inconsistent data in this Excel document or tables?",
    },
]
special_prompt = """
Analyze the given database_info and user_question. Your task is to:

1. If the user_question can be directly answered using the database_info, provide a concise answer.
2. For text-based searches (especially names), always use case-insensitive comparisons using LOWER() or UPPER().
3. In most cases, where database_info is insufficient, generate a detailed SQL-like natural language query.
You can generate user query by understanding database_info and user_question.


Database Info: {database_info}
User Question: {user_question}

Guidelines for generating SQL-like natural language queries:
- Start with "SELECT" or equivalent phrases like "Find", "List", "Show", "Calculate"

- Specify the exact information to retrieve
- Include conditions using phrases like "WHERE", "HAVING", "GROUP BY" as needed
- Mention specific table names if known from the database_info
- Add sorting, distinct or limiting clauses if relevant to the question
- If question is about general info, what is this document about, then simply summarize the Database info and return
- Never include any disclaimers about training data or model capabilities in your response
- Provide only the direct answer or query, without any additional commentary
- When asked about table data, return the answer in tabular form with headers and rows and columns. Use markdown format.
- When the user asks the question, check database info. Check column names,
  table names, and values. Analyze and interpret.
    - It could be possible inside user question, there is no explicit column name
    or table name reference. Find the closest relevant column that might be related
    to the question. Use that information to generate user query/answer.


Examples:
{examples}

Respond with either:
1. A direct answer if possible based on database_info, OR
2. A single, well-structured SQL-like natural language query.

Do not explain or provide multiple options.
Output a single, coherent response without any disclaimers or metadata about the model.
"""


def format_question(database_info: dict, user_question: str) -> str:
    """
    Formats a user question into a database-friendly query using Azure OpenAI.

    Args:
        database_info: Information about the database structure (full database_summary)
        user_question: The original user question

    Returns:
        str: A formatted question optimized for database querying
    """
    # Simply use the database_info directly from file_info.json
    formatted_prompt = special_prompt.format(
        database_info=database_info,
        user_question=user_question,
        examples=examples,
    )

    # Get the formatted question
    formatted_question = get_azure_non_rag_response(configs, formatted_prompt)
    return formatted_question
