from configs.app_config import Config
from rtl_rag_chatbot_api.chatbot.chatbot_creator import get_azure_non_rag_response

configs = Config()

examples = [
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
2. In most cases, where database_info is insufficient,
 generate a detailed SQL-like natural language query that could answer the user_question.
   This query should be structured similarly to SQL but written in plain English.

Database Info: {database_info}
Table Name: {table_name}
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

Examples:
{examples}

Respond with either:
1. A direct answer if possible based on database_info, OR
2. A single, well-structured SQL-like natural language query.

Do not explain or provide multiple options.
Output a single, coherent response without any disclaimers or metadata about the model.
"""


def format_question(database_info, user_question, table_name):
    formatted_prompt = special_prompt.format(
        database_info=database_info,
        user_question=user_question,
        table_name=table_name,
        examples=examples,
    )
    return get_azure_non_rag_response(configs, formatted_prompt)
