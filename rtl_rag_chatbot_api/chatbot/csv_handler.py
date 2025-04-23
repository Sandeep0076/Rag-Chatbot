import logging
import os
from contextlib import contextmanager
from typing import List, Optional, Union

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import AzureChatOpenAI
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from configs.app_config import Config
from rtl_rag_chatbot_api.chatbot.chatbot_creator import get_azure_non_rag_response
from rtl_rag_chatbot_api.chatbot.gemini_handler import get_gemini_non_rag_response
from rtl_rag_chatbot_api.chatbot.prompt_handler import format_question
from rtl_rag_chatbot_api.chatbot.utils.prompt_builder import PromptBuilder
from rtl_rag_chatbot_api.common.prepare_sqlitedb_from_csv_xlsx import (
    PrepareSQLFromTabularData,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TabularDataHandler:
    """
    Handles processing and querying of tabular data stored in SQLite databases.

    This class provides functionality to initialize SQLite databases from CSV or Excel files,
    and to query these databases using natural language processing techniques.

    Attributes:
        config (Config): Configuration object containing necessary settings.
        file_id (str): Unique identifier for the file being processed.
        model_choice (str): The chosen language model for processing queries.
        data_dir (str): Directory path for storing data files.
        db_name (str): Name of the SQLite database file.
        db_path (str): Full path to the SQLite database file.
        db_url (str): SQLite database URL.
        engine (Engine): SQLAlchemy database engine.
        Session (sessionmaker): SQLAlchemy session maker.
        db (SQLDatabase): SQLDatabase instance for database operations.
        llm (Union[AzureChatOpenAI, ChatVertexAI]): Language model instance for natural language processing.
        agent (Agent): SQL agent for executing database queries.
        table_info (List[dict]): Information about tables in the database.
        table_name (str): Name of the main table in the database.
    """

    def __init__(
        self, config: Config, file_id: str = None, model_choice: str = "gpt_4o_mini"
    ):
        """
        Initializes the TabularDataHandler with the given configuration and file information.

        Args:
            config (Config): Configuration object containing necessary settings.
            file_id (str, optional): Unique identifier for the file being processed. Defaults to None.
            model_choice (str, optional): The chosen language model. Defaults to "gpt_4o_mini".
        """
        self.config = config
        self.file_id = file_id
        self.model_choice = model_choice
        self.data_dir = (
            "rtl_rag_chatbot_api/tabularData/csv_dir"
            if file_id is None
            else f"./chroma_db/{file_id}"
        )
        self.db_name = "tabular_data.db"
        self.db_path = os.path.join(self.data_dir, self.db_name)
        self.db_url = f"sqlite:///{self.db_path}"
        # Configure connection pooling
        self.engine = create_engine(
            self.db_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
        )
        self.Session = sessionmaker(bind=self.engine)
        self.db = SQLDatabase(engine=self.engine)
        self.llm = self._initialize_llm()
        self.agent = None

        # Get database info - this now returns a database_summary structure
        db_info = self.get_table_info()

        # Extract table_info from database_summary for backward compatibility
        if isinstance(db_info, dict) and "tables" in db_info:
            # New format - database_summary structure
            self.table_info = db_info["tables"]
        else:
            # Old format - direct table_info list
            self.table_info = db_info

        # Set table_name from table_info
        if self.table_info and len(self.table_info) > 0:
            self.table_name = self.table_info[0][
                "name"
            ]  # Assuming the first table is the one we want
        else:
            raise ValueError("No tables found in the database")

    def _initialize_llm(self) -> Union[AzureChatOpenAI, ChatVertexAI]:
        """
        Initializes and returns an instance of either AzureChatOpenAI or ChatVertexAI based on model choice.

        Returns:
            Union[AzureChatOpenAI, ChatVertexAI]: An instance of either Azure OpenAI or Vertex AI chat model.

        Raises:
            ValueError: If the configuration for the specified model is not found.
        """
        logging.info(f"Initializing LLM with model choice: {self.model_choice}")

        # Handle Gemini models
        if self.model_choice.startswith("gemini"):
            model_config = self.config.gemini
            if not model_config:
                raise ValueError("Configuration for Gemini model not found")

            # Map model choice to actual model name
            model_mapping = {
                "gemini-flash": model_config.model_flash,
                "gemini-pro": model_config.model_pro,
            }

            model_name = model_mapping.get(self.model_choice)
            if not model_name:
                raise ValueError(
                    f"Invalid Gemini model choice: {self.model_choice}. Available choices: {list(model_mapping.keys())}"
                )

            logging.info(f"Using Gemini model: {model_name}")
            return ChatVertexAI(
                model_name=model_name,
                project=model_config.project,
                location=model_config.location,
                temperature=0.2,
                max_output_tokens=2048,
                top_p=1,
                top_k=40,
            )

        # Handle Azure OpenAI models
        model_config = self.config.azure_llm.models.get(self.model_choice)
        if not model_config:
            available_models = list(self.config.azure_llm.models.keys())
            raise ValueError(
                f"Configuration for model {self.model_choice} not found. Available models: {available_models}"
            )

        logging.info(f"Using Azure OpenAI model: {self.model_choice}")
        return AzureChatOpenAI(
            azure_endpoint=model_config.endpoint,
            azure_deployment=model_config.deployment,
            api_version=model_config.api_version,
            api_key=model_config.api_key,
            model_name=model_config.model_name,
            temperature=0.2,
        )

    @contextmanager
    def get_db_session(self):
        """Context manager for database sessions"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def initialize_database(self, is_new_file: bool = True):
        """Initialize database - either create new or verify existing"""
        os.makedirs(self.data_dir, exist_ok=True)

        if not os.path.exists(self.db_path):
            logging.error(f"Database file not found at: {self.db_path}")
            return False

        return True

    def cleanup(self):
        """Cleanup database connections"""
        if hasattr(self, "engine"):
            self.engine.dispose()

    def __del__(self):
        """Ensure cleanup on object destruction"""
        self.cleanup()

    def prepare_database(self):
        """
        Prepares the SQLite database by processing the input file (CSV, Excel, or SQLite DB).
        For CSV and Excel files, it creates tables in the database.
        For SQLite DB files, it copies and validates the database structure.
        """
        file_path = self.file_path if hasattr(self, "file_path") else self.data_dir
        data_preparer = PrepareSQLFromTabularData(
            file_path=file_path, output_dir=self.data_dir
        )
        success = data_preparer.run_pipeline()

        if not success:
            raise ValueError("Failed to prepare database from input file")

        # Initialize SQL agent after database is prepared
        self._initialize_agent()

    def _clean_sql_query(self, query: str) -> str:
        """
        Cleans SQL query by removing markdown formatting characters.

        Args:
            query (str): The SQL query that might contain markdown formatting.

        Returns:
            str: Cleaned SQL query without markdown formatting.
        """
        # Remove markdown code block delimiters
        if query.startswith("```") and query.endswith("```"):
            # Remove starting and ending backticks
            query = query[3:].strip()
            if query.endswith("```"):
                query = query[:-3].strip()

        # If query starts with 'sql' or other language identifier, remove it
        lines = query.split("\n")
        if lines and lines[0].lower().strip() in ["sql", "mysql", "sqlite"]:
            query = "\n".join(lines[1:]).strip()

        return query

    def _initialize_agent(self):
        """
        Initializes the SQL agent for executing database queries.
        """
        # Create a custom SQLDatabase wrapper that preprocesses queries
        original_run = self.db.run

        def wrapped_run(query, *args, **kwargs):
            # Clean the query before execution
            cleaned_query = self._clean_sql_query(query)
            logging.info(f"Original query: {query}")
            logging.info(f"Cleaned query: {cleaned_query}")
            return original_run(cleaned_query, *args, **kwargs)

        # Replace the run method with our wrapped version
        self.db.run = wrapped_run

        toolkit = SQLDatabaseToolkit(
            db=self.db, llm=self.llm, handle_parsing_errors=True
        )
        self.agent = create_sql_agent(
            llm=self.llm,
            toolkit=toolkit,
            verbose=True,
            handle_parsing_errors=True,
            agent_executor_kwargs={
                "handle_parsing_errors": True,
                "return_intermediate_steps": True,
            },
        )

    def debug_database(self):
        """
        Logs debug information about the database, including table names and sample data.
        """
        try:
            logging.info(
                f"Tables in the database: {[table['name'] for table in self.table_info]}"
            )

            if self.table_name:
                with self.engine.connect() as connection:
                    result = connection.execute(
                        text(f"SELECT * FROM `{self.table_name}` LIMIT 5")
                    )
                    rows = result.fetchall()
                    logging.info(f"First 5 rows of '{self.table_name}' table: {rows}")
            else:
                logging.info("No tables found in the database.")
        except Exception as e:
            logging.error(f"Error debugging database: {str(e)}", exc_info=True)

    def get_table_info(self) -> List[dict]:
        """
        Retrieves detailed information about all tables in the database.

        Returns:
            List[dict]: A list of dictionaries containing table information, including
                        table name, columns, row count, sample data, and column statistics.
        """
        inspector = inspect(self.engine)
        table_info = []
        with self.Session() as session:
            for table_name in inspector.get_table_names():
                columns = inspector.get_columns(table_name)
                row_count = session.execute(
                    text(f'SELECT COUNT(*) FROM "{table_name}"')
                ).scalar()
                sample_data = session.execute(
                    text(f'SELECT * FROM "{table_name}" LIMIT 3')
                ).fetchall()

                column_stats = {}
                for column in columns:
                    if hasattr(column["type"], "python_type") and column[
                        "type"
                    ].python_type in (int, float):
                        stats = session.execute(
                            text(
                                f'SELECT MIN("{column["name"]}"), MAX("{column["name"]}"), '
                                f'AVG("{column["name"]}") FROM "{table_name}"'
                            )
                        ).fetchone()
                        column_stats[column["name"]] = {
                            "min": stats[0],
                            "max": stats[1],
                            "avg": stats[2],
                        }

                table_info.append(
                    {
                        "name": table_name,
                        "columns": [
                            {"name": col["name"], "type": str(col["type"])}
                            for col in columns
                        ],
                        "row_count": row_count,
                        "sample_data": sample_data,
                        "column_stats": column_stats,
                    }
                )
        # Compose a database summary for file_info.json
        database_summary = {
            "table_count": len(table_info),
            "table_names": [t["name"] for t in table_info],
            "tables": [],
        }
        for t in table_info:
            database_summary["tables"].append(
                {
                    "name": t["name"],
                    "columns": t["columns"],
                    "row_count": t["row_count"],
                    "top_rows": [list(row) for row in t["sample_data"]],
                    "column_stats": t["column_stats"],
                }
            )
        return database_summary

    def get_answer(self, question: str) -> str:
        """
        Processes a user's question and returns an answer based on the database content.

        Args:
            question (str): The user's input question.

        Returns:
            str: The answer to the user's question or an error message if processing fails.
        """
        try:
            answer = self.ask_question(question)
            if answer:
                print("Direct answer")
                return answer
            else:
                print("Forced answer")
                return self.get_forced_answer(question, answer)
        except Exception as e:
            logging.error(f"Error in TabularDataHandler get_answer: {str(e)}")
            return f"An error occurred while processing your question: {str(e)}"

    def ask_question(self, question: str) -> Optional[str]:
        """
        Processes a question using natural language processing and database querying.

        Args:
            question (str): The user's input question.

        Returns:
            Optional[str]: The answer to the question, or None if processing fails.
        """
        if not self.agent:
            self._initialize_agent()

        try:
            # Import GCSHandler to access file_info.json
            from rtl_rag_chatbot_api.chatbot.gcs_handler import GCSHandler

            gcs_handler = GCSHandler(self.config)

            # Get database_summary from file_info.json instead of regenerating it
            file_info = gcs_handler.get_file_info(self.file_id)

            # Use database_summary from file_info if available, otherwise fall back to table_info
            if file_info and "database_summary" in file_info:
                db_summary = file_info["database_summary"]
                logging.info(
                    "Using database_summary from file_info.json for file_id: %s",
                    self.file_id,
                )
            else:
                logging.info(
                    "No database_summary found in file_info.json, generating from table_info"
                )

            # Use the format_question function from prompt_handler with database summary
            formatted_question = format_question(db_summary, question)
            logging.info(f"Formatted question: {formatted_question}")

            # Check if the formatted question contains specific keywords
            keywords = ["SELECT", "FIND", "LIST", "SHOW", "CALCULATE"]
            if formatted_question and any(
                keyword in formatted_question.upper() for keyword in keywords
            ):
                # Enhance the query with case-insensitive comparisons
                response = self.agent.invoke({"input": formatted_question})

                # Extract the final answer and intermediate steps
                final_answer = response.get("output", "No final answer found")
                intermediate_steps = response.get("intermediate_steps", [])
                complete_logs = str(intermediate_steps) + "\n" + str(final_answer)
                base_prompt = PromptBuilder.build_forced_answer_prompt(
                    formatted_question, complete_logs
                )

                # Format the response using the appropriate model
                if self.model_choice.startswith("gemini"):
                    return get_gemini_non_rag_response(
                        self.config, base_prompt, self.model_choice
                    )
                else:
                    return get_azure_non_rag_response(self.config, base_prompt)
            else:
                # If no keywords are found, return the formatted question as is
                return formatted_question

            return None
        except Exception as e:
            logging.error(f"An error occurred while processing the question: {str(e)}")
            raise

    def get_forced_answer(self, question: str, answer: str) -> str:
        """
        Attempts to extract an answer from a given text when a direct answer is not available.

        Args:
            question (str): The original question asked by the user.
            answer (str): The text to search for an answer.

        Returns:
            str: An extracted answer or "Cannot find answer" if no suitable answer is found.
        """
        try:
            base_prompt = PromptBuilder.build_forced_answer_prompt(question, answer)
            return get_azure_non_rag_response(self.config, base_prompt)
        except Exception as e:
            logging.error(f"Error in get_forced_answer: {str(e)}")
            return f"An error occurred while processing your question: {str(e)}"

    def interactive_session(self):
        """
        Starts an interactive session for querying the database.
        Allows users to input questions and receive answers based on the database content.
        """
        print("Welcome to the interactive SQL query session.")
        print("Type 'exit' to end the session.")

        while True:
            question = input("\nEnter your question: ").strip()

            if question.lower() == "exit":
                print("Exiting the session. Goodbye!")
                break

            answer = self.get_answer(question)

            if answer:
                print(f"\nAnswer: {answer}")
            else:
                print(
                    "Sorry, I couldn't find an answer to that question. Let me try again"
                )
                return self.get_forced_answer(question, answer)


# def main(data_dir: str):
#     handler = TabularDataHandler(data_dir)
#     handler.prepare_database()
#     table_info = handler.get_table_info()
#     handler.interactive_session()


# if __name__ == "__main__":
#     data_dir = "rtl_rag_chatbot_api/tabularData/csv_dir"
#     main(data_dir)
