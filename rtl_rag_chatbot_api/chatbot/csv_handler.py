import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

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
        file_ids (List[str]): List of file identifiers for multi-file mode.
        file_id (str): Primary file identifier (for backward compatibility).
        is_multi_file (bool): Whether multiple files are being handled.
        model_choice (str): The chosen language model for processing queries.
        engines (dict): Dictionary of SQLAlchemy engines, one per file.
        sessions (dict): Dictionary of sessionmakers, one per file.
        dbs (dict): Dictionary of SQLDatabase instances, one per file.
        llm (Union[AzureChatOpenAI, ChatVertexAI]): Language model instance for natural language processing.
        agents (dict): Dictionary of SQL agents, one per file.
        database_summaries (dict): Information about tables for each database.
        primary_db_info (dict): Information about tables in the primary database.
    """

    def __init__(
        self,
        config: Config,
        file_id: str = None,
        model_choice: str = "gpt_4o_mini",
        file_ids: List[str] = None,
        database_summaries_param: Optional[Dict[str, Any]] = None,
        all_file_infos: Optional[Dict[str, Any]] = None,
        temperature: float = None,
    ):
        """
        Initializes the TabularDataHandler with the given configuration and file information.

        Args:
            config (Config): Configuration object containing necessary settings.
            file_id (str, optional): Unique identifier for the file being processed. Defaults to None.
            model_choice (str, optional): The chosen language model. Defaults to "gpt_4o_mini".
            file_ids (List[str], optional): List of file IDs for multi-file mode. Defaults to None.
            database_summaries_param (Optional[Dict[str, Any]], optional): Pre-loaded
                database summaries keyed by file_id. Defaults to None.
            all_file_infos (Optional[Dict[str, Any]], optional): Complete file information
                including metadata for context. Defaults to None.
        """
        self.config = config
        self.model_choice = model_choice
        self.db_name = "tabular_data.db"
        self.engines = {}
        self.sessions = {}
        self.dbs = {}
        self.agents = {}

        # Set temperature - use provided value or model-specific default
        if temperature is not None:
            self.temperature = temperature
        elif model_choice.lower() in ["gemini-2.5-flash", "gemini-2.5-pro"]:
            self.temperature = 0.8  # Higher temperature for Gemini models
        else:
            self.temperature = 0.5  # Lower temperature for OpenAI models

        # Initialize database_summaries with pre-loaded data if provided
        self.database_summaries = {}
        if database_summaries_param:
            logging.info(
                f"Using pre-loaded database summaries for "
                f"{len(database_summaries_param)} files"
            )
            self.database_summaries = database_summaries_param

        # Store all_file_infos for context in responses
        self.all_file_infos = all_file_infos if all_file_infos else {}

        # Determine if we're in multi-file mode
        self.is_multi_file = bool(file_ids and len(file_ids) > 0)

        if self.is_multi_file:
            # Multi-file mode
            self.file_ids = sorted(list(set(file_ids)))  # Ensure unique and sorted
            # For backward compatibility, set file_id to the first one
            self.file_id = self.file_ids[0] if self.file_ids else None
            logging.info(
                f"TabularDataHandler initialized in multi-file mode with "
                f"{len(self.file_ids)} files"
            )

            # Initialize databases for all files
            for f_id in self.file_ids:
                self._initialize_file_database(f_id)
        else:
            # Single file mode (backward compatible)
            self.file_id = file_id
            self.file_ids = [file_id] if file_id else []
            if file_id:
                self._initialize_file_database(file_id)
            else:
                # Legacy path for tests or default initialization
                self.data_dir = "rtl_rag_chatbot_api/tabularData/csv_dir"
                self.db_path = os.path.join(self.data_dir, self.db_name)
                self.db_url = f"sqlite:///{self.db_path}"
                self.engines["default"] = create_engine(
                    self.db_url,
                    poolclass=QueuePool,
                    pool_size=5,
                    max_overflow=10,
                    pool_timeout=30,
                    pool_recycle=1800,
                )
                self.sessions["default"] = sessionmaker(bind=self.engines["default"])
                self.dbs["default"] = SQLDatabase(engine=self.engines["default"])

        # Initialize LLM for all database operations
        self.llm = self._initialize_llm()

        # For backward compatibility, set these attributes for the primary file
        if self.file_id:
            self.engine = self.engines.get(self.file_id)
            self.Session = self.sessions.get(self.file_id)
            self.db = self.dbs.get(self.file_id)
            self.agent = self.agents.get(self.file_id)

            if self.file_id in self.database_summaries:
                # Set primary DB info and table_info
                self.primary_db_info = self.database_summaries[self.file_id]

                if (
                    isinstance(self.primary_db_info, dict)
                    and "tables" in self.primary_db_info
                ):
                    self.table_info = self.primary_db_info["tables"]
                else:
                    self.table_info = self.primary_db_info

                # Set table_name from table_info
                if self.table_info and len(self.table_info) > 0:
                    self.table_name = self.table_info[0]["name"]
                else:
                    raise ValueError(
                        f"No tables found in the primary database for file_id: {self.file_id}"
                    )
            else:
                raise ValueError(
                    f"No database summary found for file_id: {self.file_id}"
                )

    def _initialize_file_database(self, file_id: str):
        """
        Initialize a single file's database and related components.

        Args:
            file_id (str): The file ID to initialize
        """
        data_dir = f"./chroma_db/{file_id}"
        db_path = os.path.join(data_dir, self.db_name)
        db_url = f"sqlite:///{db_path}"

        # Configure connection pooling
        self.engines[file_id] = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
        )
        self.sessions[file_id] = sessionmaker(bind=self.engines[file_id])
        self.dbs[file_id] = SQLDatabase(engine=self.engines[file_id])

        # Get database info - check if we already have a pre-loaded summary first
        try:
            if file_id in self.database_summaries:
                logging.info(
                    f"Using pre-loaded database summary for file_id: {file_id}"
                )
                # No need to re-extract summary as it's already loaded
            else:
                logging.info(f"Generating database summary for file_id: {file_id}")
                db_info = self._get_table_info_for_file(file_id)
                self.database_summaries[file_id] = db_info

            # Initialize SQL agent after database is prepared
            self._initialize_agent_for_file(file_id)
            logging.info(f"Successfully initialized database for file_id: {file_id}")
        except Exception as e:
            logging.error(
                f"Error initializing database for file_id {file_id}: {str(e)}"
            )

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

            # Map model choice to actual model name - only 2.5 models
            model_mapping = {
                "gemini-2.5-flash": model_config.model_flash,
                "gemini-2.5-pro": model_config.model_pro,
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
                temperature=self.temperature,
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
            temperature=self.temperature,
        )

    @contextmanager
    def get_db_session(self, file_id=None):
        """Context manager for database sessions"""
        # Use the specified file_id or fall back to primary file_id
        current_file_id = file_id or self.file_id

        if current_file_id in self.sessions:
            session = self.sessions[current_file_id]()
        else:
            # Fallback to primary session
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

    def _initialize_agent_for_file(self, file_id: str):
        """Initializes the SQL agent for a specific file's database.

        Args:
            file_id (str): The file ID whose SQL agent to initialize
        """
        if file_id not in self.dbs:
            raise ValueError(f"No database initialized for file_id: {file_id}")

        toolkit = SQLDatabaseToolkit(db=self.dbs[file_id], llm=self.llm)
        self.agents[file_id] = create_sql_agent(
            llm=self.llm, toolkit=toolkit, verbose=True, handle_parsing_errors=True
        )

        # For backward compatibility with the primary file_id
        if file_id == self.file_id:
            self.agent = self.agents[file_id]

    def _initialize_agent(self):
        """Initializes the SQL agent for the primary file (backward compatibility)."""
        if self.file_id:
            self._initialize_agent_for_file(self.file_id)
        # Ensure self.db is available for the primary file_id context
        if hasattr(self, "db") and self.db:
            original_run_method = self.db.run

            # Define the wrapper function with **kwargs to handle additional parameters
            def wrapped_run(command: str, fetch: str = "all", **kwargs) -> str:
                # 'self' of TabularDataHandler is captured from the outer scope
                logging.debug(f"Executing SQL via wrapped_run: {command}")
                try:
                    # Clean the SQL query to remove markdown formatting and other artifacts
                    cleaned_command = self._clean_sql_query(command)
                    # Pass only the parameters that the original method accepts
                    # Ignore any additional kwargs like 'parameters' that might be passed by newer LangChain
                    result = original_run_method(cleaned_command, fetch)
                    logging.debug(f"SQL result (first 100 chars): {str(result)[:100]}")
                    return result
                except Exception as e:
                    logging.error(
                        f"Error in wrapped_run during SQL execution for command '{command}': {str(e)}"
                    )
                    raise

            # Replace the run method with our wrapped version
            self.db.run = wrapped_run
            logging.info(
                f"Patched db.run method for file_id: {self.file_id or 'default'}"
            )
        else:
            logging.warning(
                "Primary self.db not found or not initialized. Cannot patch db.run."
            )

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

    def _get_table_info_for_file(self, file_id: str) -> dict:
        """
        Retrieves detailed information about all tables in a specific database.

        Args:
            file_id (str): The file ID whose database to analyze

        Returns:
            dict: A dictionary containing database summary information
        """
        if file_id not in self.engines:
            raise ValueError(f"No engine initialized for file_id: {file_id}")

        inspector = inspect(self.engines[file_id])
        table_info = []
        with self.sessions[file_id]() as session:
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
        # Compose a database summary
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

    def get_table_info(self) -> Union[List[dict], dict]:
        """
        Retrieves detailed information about all tables in the database(s).

        For multi-file mode, returns information about the primary database.
        For backward compatibility with existing code.

        Returns:
            Union[List[dict], dict]: Information about tables in the database(s)
        """
        if self.is_multi_file:
            if self.file_id in self.database_summaries:
                return self.database_summaries[self.file_id]
            elif self.file_ids:
                # If primary file_id not in summaries but we have other files, use the first available
                for f_id in self.file_ids:
                    if f_id in self.database_summaries:
                        return self.database_summaries[f_id]
                # If no file has a summary yet, try to generate one for the primary
                return self._get_table_info_for_file(self.file_id)

        # Traditional single-file mode
        return self._get_table_info_for_file(self.file_id)

    def get_answer(self, question: str) -> str:
        """
        Processes a user's question and returns an answer based on the database content.
        For multi-file mode, it attempts to determine which database to query or uses a combined approach.

        Args:
            question (str): The user's input question.

        Returns:
            str: The answer to the user's question or an error message if processing fails.
        """
        try:
            if self.is_multi_file and len(self.file_ids) > 1:
                return self._get_multi_file_answer(question)
            else:
                # Standard single-file approach (also works when only one file in multi-file mode)
                answer = self.ask_question(question)
                if answer:
                    logging.info("Direct answer")
                    return answer
                else:
                    logging.info("Forced answer")
                    return self.get_forced_answer(question, answer)
        except Exception as e:
            logging.error(f"Error in TabularDataHandler get_answer: {str(e)}")
            return f"An error occurred while processing your question: {str(e)}"

    def _get_multi_file_answer(self, question: str) -> str:
        """
        Process a question in multi-file mode by querying across all available databases.
        This method handles queries when multiple tabular files are in use.

        Args:
            question (str): The user's input question

        Returns:
            str: The answer combining information from all databases
        """
        try:
            # First check if the question specifically mentions file IDs or table names to
            # determine which database to use for the query
            target_file_id = None

            # 1. First check if question contains table names that match specific databases
            for file_id, summary in self.database_summaries.items():
                if "table_names" in summary:
                    for table_name in summary["table_names"]:
                        if table_name.lower() in question.lower():
                            logging.info(
                                f"Found table '{table_name}' mentioned in question, using file {file_id}"
                            )
                            target_file_id = file_id
                            break
                if target_file_id:  # Break outer loop too if we found a match
                    break

            # 2. If no specific table identified, create a comprehensive database summary
            if not target_file_id:
                # Create a combined database summary for context
                all_tables_summary = self._generate_all_tables_summary()
                logging.info(
                    "No specific table identified in question. Using combined database approach."
                )

                # Try to route the question to the most relevant database based on content
                suggested_file_id = self._suggest_database_for_question(
                    question, all_tables_summary
                )
                if suggested_file_id:
                    target_file_id = suggested_file_id
                    logging.info(
                        f"Selected database {target_file_id} as most relevant for question"
                    )
                else:
                    # Default to the first file when we can't determine which is more relevant
                    target_file_id = self.file_ids[0]
                    logging.info(
                        f"Using default database {target_file_id} for question"
                    )

            # We've determined which database to use, set up the agent for that database
            if target_file_id not in self.agents:
                self._initialize_agent_for_file(target_file_id)

            # Format the question with additional context if needed
            enhanced_question = self._enhance_question_with_context(
                question, target_file_id
            )

            # Run the question on the selected agent
            agent = self.agents[target_file_id]
            if not agent:
                self._initialize_agent_for_file(target_file_id)
                agent = self.agents[target_file_id]

            logging.info(f"Querying database for file_id: {target_file_id}")
            result = agent.run(enhanced_question)

            # Add source information to the response (without file context)
            # Get just the filename for source attribution
            file_info = (
                self.all_file_infos.get(target_file_id, {})
                if self.all_file_infos
                else {}
            )
            original_filename = file_info.get(
                "original_filename", f"File {target_file_id}"
            )

            enhanced_result = f"[Source: {original_filename}] {result}"
            return enhanced_result

        except Exception as e:
            logging.error(f"Error in _get_multi_file_answer: {str(e)}")
            return f"An error occurred while processing your question across multiple databases: {str(e)}"

    def _generate_all_tables_summary(self) -> str:
        """
        Generate a concise summary of all tables across all databases.

        Returns:
            str: Text summary of all available tables and their columns
        """
        summary_parts = []

        for file_id, db_summary in self.database_summaries.items():
            if "tables" in db_summary:
                summary_parts.append(f"Database {file_id}:")
                for table in db_summary["tables"]:
                    columns_info = ", ".join(
                        [f"{col['name']} ({col['type']})" for col in table["columns"]]
                    )
                    summary_parts.append(
                        f"  Table '{table['name']}' ({table['row_count']} rows): {columns_info}"
                    )

        return "\n".join(summary_parts)

    def _suggest_database_for_question(
        self, question: str, all_tables_summary: str
    ) -> Optional[str]:
        """
        Suggest which database is most relevant to the question based on content.
        Uses LLM to make the determination.

        Args:
            question: The user's question
            all_tables_summary: Summary of all tables across all databases

        Returns:
            Optional[str]: The file_id of the most relevant database, or None if undetermined
        """
        try:
            if self.model_choice.startswith("gemini"):
                from rtl_rag_chatbot_api.chatbot.gemini_handler import (
                    get_gemini_non_rag_response,
                )

                prompt = (
                    f"Based on the following database structure:\n\n{all_tables_summary}\n\n"
                    f"Which database would be most appropriate to answer this question: '{question}'?\n"
                    "Respond with just the database ID (like '1fddcdde-c24e-4fef-b656-dc454f701418')."
                )
                response = get_gemini_non_rag_response(
                    self.config, prompt, self.model_choice
                )
            else:
                from rtl_rag_chatbot_api.chatbot.chatbot_creator import (
                    get_azure_non_rag_response,
                )

                prompt = (
                    f"Based on the following database structure:\n\n{all_tables_summary}\n\n"
                    f"Which database would be most appropriate to answer this question: '{question}'?\n"
                    "Respond with just the database ID (like '1fddcdde-c24e-4fef-b656-dc454f701418')."
                )
                response = get_azure_non_rag_response(
                    self.config, prompt, self.model_choice
                )

            # Extract file_id from response
            for file_id in self.file_ids:
                if file_id in response:
                    return file_id
            return None
        except Exception as e:
            logging.error(f"Error in _suggest_database_for_question: {str(e)}")
            return None

    def _enhance_question_with_context(self, question: str, file_id: str) -> str:
        """
        Enhance the user's question with context about the selected database.

        Args:
            question: The user's original question
            file_id: The selected file_id to query against

        Returns:
            str: An enhanced question with context
        """
        if file_id in self.database_summaries:
            db_summary = self.database_summaries[file_id]
            table_names = db_summary.get("table_names", [])

            # Don't modify simple questions about table structure
            if any(
                keyword in question.lower()
                for keyword in [
                    "what tables",
                    "table names",
                    "how many tables",
                    "show tables",
                    "list tables",
                ]
            ):
                return question

            # Format enhanced question with table context
            if table_names and len(table_names) == 1:
                # Single table case - make it easy
                return f"Using the table '{table_names[0]}', {question}"
            elif table_names and len(table_names) > 1:
                tables_str = ", ".join([f"'{t}'" for t in table_names])
                return f"Using the tables {tables_str}, {question}"

        # Default case - just return the original question
        return question

    def _get_file_context_string(self, file_id: str = None) -> str:
        """
        Get file context information for inclusion in responses.

        Args:
            file_id: The file ID to get context for. Uses self.file_id if not provided.

        Returns:
            str: Formatted file context string
        """
        if not self.all_file_infos:
            return ""

        target_file_id = file_id or self.file_id
        if not target_file_id or target_file_id not in self.all_file_infos:
            return ""

        file_info = self.all_file_infos.get(target_file_id, {})
        original_filename = file_info.get("original_filename", "Unknown file")

        if self.is_multi_file:
            # For multi-file, just include filename reference
            return f"[Source: {original_filename}]"
        else:
            # For single file, include complete file information
            import json

            try:
                file_info_str = json.dumps(file_info, indent=2, default=str)
                return f"File Information:\n{file_info_str}\n\n"
            except Exception:
                # Fallback to just filename if JSON serialization fails
                return f"File: {original_filename}\n\n"

    def _get_file_context_for_prompt(self, file_id: str = None) -> str:
        """
        Get file context information for inclusion in AI prompts (not responses).

        Args:
            file_id: The file ID to get context for. Uses self.file_id if not provided.

        Returns:
            str: Formatted file context string for prompt
        """
        if not self.all_file_infos:
            return ""

        target_file_id = file_id or self.file_id
        if not target_file_id or target_file_id not in self.all_file_infos:
            return ""

        file_info = self.all_file_infos.get(target_file_id, {})
        original_filename = file_info.get("original_filename", "Unknown file")

        # Start with filename context
        context_parts = [f"You are working with the file: '{original_filename}'"]

        # Add complete database summary if available
        if "database_summary" in file_info:
            db_summary = file_info["database_summary"]

            # Send the complete database summary as context (no cleaning)
            import json

            try:
                # Format the complete database summary as readable context
                db_summary_str = json.dumps(db_summary, indent=2, default=str)
                context_parts.append(f"Database Summary:\n{db_summary_str}")

            except Exception:
                # Fallback to string representation if JSON fails
                context_parts.append(f"Database Summary:\n{str(db_summary)}")

        context_str = "\n\n".join(context_parts) + "\n\n"
        return context_str

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
                # Create database_summary from table_info
                db_summary = {
                    "table_count": len(self.table_info),
                    "table_names": [t["name"] for t in self.table_info],
                    "tables": self.table_info,
                }

            # Add file context to the question before processing
            file_context = self._get_file_context_for_prompt()
            enhanced_question = (
                f"{file_context}{question}" if file_context else question
            )

            # Use the format_question function from prompt_handler with database summary
            formatted_question = format_question(db_summary, enhanced_question)
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

                # Use intelligent truncation to prevent token overflow
                truncated_context = self._truncate_intermediate_steps(
                    intermediate_steps, final_answer
                )

                base_prompt = PromptBuilder.build_forced_answer_prompt(
                    formatted_question, truncated_context
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

    def _truncate_intermediate_steps(
        self,
        intermediate_steps: list,
        final_answer: str,
        max_context_tokens: int = 50000,
    ) -> str:
        """
        Intelligently truncate intermediate steps to prevent token overflow while preserving key information.

        Args:
            intermediate_steps: List of intermediate steps from agent execution
            final_answer: The final answer from the agent
            max_context_tokens: Maximum estimated tokens for context (roughly 4 chars per token)

        Returns:
            str: Truncated context that fits within token limits
        """

        # Estimate token count (rough approximation: 4 characters = 1 token)
        def estimate_tokens(text: str) -> int:
            return len(str(text)) // 4

        final_answer_str = str(final_answer)
        final_tokens = estimate_tokens(final_answer_str)

        # If final answer itself is extremely large, truncate it first
        if (
            final_tokens > max_context_tokens * 0.8
        ):  # If final answer uses >80% of tokens
            logging.warning(
                f"Final answer is too large ({final_tokens} tokens), truncating"
            )
            # Keep the most important part of the final answer
            max_final_chars = int(max_context_tokens * 0.8 * 4)
            final_answer_str = (
                final_answer_str[:max_final_chars] + "... [truncated due to length]"
            )
            final_tokens = estimate_tokens(final_answer_str)

        # Reserve tokens for final answer and prompt structure
        available_tokens = max_context_tokens - final_tokens - 2000  # Buffer for prompt

        if available_tokens <= 1000:  # Need reasonable space for context
            # If very little space available, return minimal context
            logging.warning("Very limited tokens available, returning minimal context")
            return "Query executed with large results. " + final_answer_str

        # Convert intermediate steps to string and get essential parts
        intermediate_str = str(intermediate_steps)

        if estimate_tokens(intermediate_str) <= available_tokens:
            # If it fits, return everything
            return intermediate_str + "\n" + final_answer_str

        # Need to truncate - extract key information
        essential_info = []

        # Look for SQL queries and their results in intermediate steps
        if isinstance(intermediate_steps, list):
            for step in intermediate_steps[
                -5:
            ]:  # Only check last 5 steps for efficiency
                step_str = str(step)
                # Extract SQL queries (usually contain SELECT, FROM, WHERE)
                if any(
                    keyword in step_str.upper()
                    for keyword in ["SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY"]
                ):
                    # Keep SQL queries as they're essential for understanding
                    current_length = estimate_tokens("\n".join(essential_info))
                    if current_length + estimate_tokens(step_str) < available_tokens:
                        essential_info.append(
                            f"SQL Query executed: {step_str[:500]}..."
                        )  # Limit individual steps
                    else:
                        # If adding this would exceed limit, truncate the step
                        remaining_tokens = available_tokens - current_length
                        if (
                            remaining_tokens > 50
                        ):  # Only add if we have reasonable space
                            truncated_step = (
                                step_str[: remaining_tokens * 2] + "... [truncated]"
                            )
                            essential_info.append(f"SQL Query: {truncated_step}")
                        break

        # If no essential info found or still too long, use a summary approach
        if (
            not essential_info
            or estimate_tokens("\n".join(essential_info)) > available_tokens
        ):
            # Create a minimal summary
            summary = (
                "Database query executed successfully. Results processed and formatted."
            )
            if estimate_tokens(summary) < available_tokens:
                essential_info = [summary]
            else:
                essential_info = []

        # Combine with final answer
        context = "\n".join(essential_info)
        if context:
            return context + "\n\n" + final_answer_str
        else:
            return final_answer_str


# def main(data_dir: str):
#     handler = TabularDataHandler(data_dir)
#     handler.prepare_database()
#     table_info = handler.get_table_info()
#     handler.interactive_session()


# if __name__ == "__main__":
#     data_dir = "rtl_rag_chatbot_api/tabularData/csv_dir"
#     main(data_dir)
