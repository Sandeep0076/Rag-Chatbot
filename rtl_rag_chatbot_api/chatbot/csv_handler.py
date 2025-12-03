import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_openai import AzureChatOpenAI
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from configs.app_config import Config
from rtl_rag_chatbot_api.chatbot.chatbot_creator import get_azure_non_rag_response
from rtl_rag_chatbot_api.chatbot.gcs_handler import GCSHandler
from rtl_rag_chatbot_api.chatbot.gemini_handler import (
    GeminiSafetyFilterError,
    get_gemini_non_rag_response,
)
from rtl_rag_chatbot_api.chatbot.prompt_handler import (
    format_question,
    resolve_question_with_history,
)
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
        custom_gpt: bool = False,
        system_prompt: str = None,
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
        logging.info("=== Initializing TabularDataHandler ===")
        logging.info(f"Model choice: {model_choice}")
        logging.info(f"File ID: {file_id}")
        logging.info(f"File IDs: {file_ids}")
        logging.info(
            f"Database summaries provided: {len(database_summaries_param) if database_summaries_param else 0}"
        )
        logging.info(
            f"All file infos provided: {len(all_file_infos) if all_file_infos else 0}"
        )

        self.config = config
        self.model_choice = model_choice
        self.custom_gpt = custom_gpt
        self.system_prompt = system_prompt
        self.db_name = "tabular_data.db"
        self.engines = {}
        self.sessions = {}
        self.dbs = {}
        self.agents = {}
        # Cache previous formatted question per file for history-aware resolution
        self.previous_formatted_by_file = {}
        # Cache previous resolved question per file for stronger anchoring
        self.previous_resolved_by_file = {}

        # Set temperature -use provided value or model-specific default
        if temperature is not None:
            self.temperature = temperature
        elif model_choice.lower() in ["gemini-2.5-flash", "gemini-2.5-pro"]:
            self.temperature = 0.6  # Higher temperature for Gemini models
        else:
            self.temperature = 0.4  # Lower temperature for OpenAI models

        logging.info(f"Temperature set to: {self.temperature}")

        # Initialize LLM for all database operations before initializing databases
        logging.info("Initializing LLM...")
        self.llm = self._initialize_llm()
        logging.info("LLM initialized successfully")

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
        logging.info(f"Multi-file mode: {self.is_multi_file}")

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
                logging.info(f"Initializing database for file_id: {f_id}")
                self._initialize_file_database(f_id)
        else:
            # Single file mode (backward compatible)
            self.file_id = file_id
            self.file_ids = [file_id] if file_id else []
            if file_id:
                logging.info(
                    f"Initializing single file database for file_id: {file_id}"
                )
                self._initialize_file_database(file_id)
            else:
                # Legacy path for tests or default initialization
                logging.info("Using legacy path for tests or default initialization")
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
                # Ensure default db.run is also patched
                self._patch_db_run_for_file("default")

        # For backward compatibility, set these attributes for the primary file
        if self.file_id:
            logging.info(
                f"Setting up backward compatibility attributes for file_id: {self.file_id}"
            )
            self.engine = self.engines.get(self.file_id)
            self.Session = self.sessions.get(self.file_id)
            self.db = self.dbs.get(self.file_id)
            self.agent = self.agents.get(self.file_id)

            if self.file_id in self.database_summaries:
                # Set primary DB info and table_info
                self.primary_db_info = self.database_summaries[self.file_id]
                logging.info(
                    f"Primary DB info set from database summaries for file_id: {self.file_id}"
                )

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
                    logging.info(f"Table name set to: {self.table_name}")
                else:
                    error_msg = f"No tables found in the primary database for file_id: {self.file_id}"
                    logging.error(error_msg)
                    raise ValueError(error_msg)
            else:
                error_msg = f"No database summary found for file_id: {self.file_id}"
                logging.error(error_msg)
                raise ValueError(error_msg)

        logging.info("=== TabularDataHandler initialization completed ===")

    def _validate_database_file(self, data_dir: str, db_path: str, file_id: str):
        """
        Validate that the database file exists and is accessible.

        Args:
            data_dir (str): The data directory path
            db_path (str): The database file path
            file_id (str): The file ID for logging

        Raises:
            FileNotFoundError: If directory or file doesn't exist
            PermissionError: If directory or file is not readable
        """
        logging.info(f"Validating database file for file_id: {file_id}")

        # Check if directory exists
        if not os.path.exists(data_dir):
            logging.error(f"Data directory does not exist: {data_dir}")
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Check if directory is readable
        if not os.access(data_dir, os.R_OK):
            logging.error(f"Data directory is not readable: {data_dir}")
            raise PermissionError(f"Data directory not readable: {data_dir}")

        # Check if database file exists
        if not os.path.exists(db_path):
            logging.error(f"Database file does not exist: {db_path}")
            raise FileNotFoundError(f"Database file not found: {db_path}")

        # Check if database file is readable
        if not os.access(db_path, os.R_OK):
            logging.error(f"Database file is not readable: {db_path}")
            raise PermissionError(f"Database file not readable: {db_path}")

        # Check file size
        try:
            file_size = os.path.getsize(db_path)
            logging.info(f"Database file size: {file_size} bytes")
            if file_size == 0:
                logging.warning(f"Database file is empty: {db_path}")
        except OSError as e:
            logging.error(f"Error getting file size: {str(e)}")

        # Check file permissions
        try:
            stat_info = os.stat(db_path)
            logging.info(f"Database file permissions: {oct(stat_info.st_mode)}")
        except OSError as e:
            logging.error(f"Error getting file permissions: {str(e)}")

        logging.info(f"Database file validation successful for file_id: {file_id}")

    def _validate_sqlite_database(self, db_path: str, file_id: str):
        """
        Validate that the file is a valid SQLite database.

        Args:
            db_path (str): The database file path
            file_id (str): The file ID for logging

        Raises:
            sqlite3.Error: If the file is not a valid SQLite database
        """
        logging.info(f"Validating SQLite database for file_id: {file_id}")

        try:
            import sqlite3

            test_conn = sqlite3.connect(db_path, timeout=10)
            test_conn.close()
            logging.info(f"SQLite database validation successful for: {db_path}")
        except sqlite3.Error as e:
            logging.error(f"SQLite database validation failed: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during SQLite validation: {str(e)}")
            raise

    def _create_database_components(self, db_url: str, file_id: str):
        """
        Create SQLAlchemy database components.

        Args:
            db_url (str): The database URL
            file_id (str): The file ID for logging

        Raises:
            Exception: If any component creation fails
        """
        logging.info(f"Creating database components for file_id: {file_id}")

        # Create SQLAlchemy engine
        try:
            self.engines[file_id] = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800,
            )
            logging.info(
                f"SQLAlchemy engine created successfully for file_id: {file_id}"
            )
        except Exception as e:
            logging.error(f"Failed to create SQLAlchemy engine: {str(e)}")
            raise

        # Create session maker
        try:
            self.sessions[file_id] = sessionmaker(bind=self.engines[file_id])
            logging.info(f"Session maker created successfully for file_id: {file_id}")
        except Exception as e:
            logging.error(f"Failed to create session maker: {str(e)}")
            raise

        # Create SQLDatabase instance
        try:
            self.dbs[file_id] = SQLDatabase(engine=self.engines[file_id])
            logging.info(
                f"SQLDatabase instance created successfully for file_id: {file_id}"
            )
        except Exception as e:
            logging.error(f"Failed to create SQLDatabase instance: {str(e)}")
            raise

        # Patch database run method
        try:
            self._patch_db_run_for_file(file_id)
            logging.info(
                f"Database run method patched successfully for file_id: {file_id}"
            )
        except Exception as e:
            logging.error(f"Failed to patch database run method: {str(e)}")
            raise

    def _initialize_file_database(self, file_id: str):
        """
        Initialize a single file's database and related components.

        Args:
            file_id (str): The file ID to initialize
        """
        logging.info(f"=== Starting database initialization for file_id: {file_id} ===")

        data_dir = f"./chroma_db/{file_id}"
        db_path = os.path.join(data_dir, self.db_name)
        db_url = f"sqlite:///{db_path}"

        logging.info(f"Data directory: {data_dir}")
        logging.info(f"Database path: {db_path}")
        logging.info(f"Database URL: {db_url}")

        # If the data directory doesn't exist, it means this pod hasn't processed
        # or downloaded the files yet. We need to fetch them from GCS.
        if not os.path.exists(data_dir):
            logging.warning(
                f"Data directory {data_dir} not found locally. Attempting to download from GCS."
            )
            try:
                gcs_handler = GCSHandler(_configs=self.config)
                gcs_handler.download_files_from_folder_by_id(file_id)
                logging.info(
                    f"Successfully downloaded files from GCS for file_id: {file_id}"
                )
            except Exception as e:
                logging.error(
                    f"Failed to download files from GCS for file_id {file_id}: {e}"
                )
                # Re-raise or handle the error appropriately. If files can't be downloaded,
                # the handler cannot proceed.
                raise FileNotFoundError(
                    f"Data directory {data_dir} not found and could not be downloaded from GCS."
                ) from e

        try:
            # Validate database file
            self._validate_database_file(data_dir, db_path, file_id)

            # Validate SQLite database
            self._validate_sqlite_database(db_path, file_id)

            # Create database components
            self._create_database_components(db_url, file_id)

            # Get database info - check if we already have a pre-loaded summary first
            if file_id in self.database_summaries:
                logging.info(
                    f"Using pre-loaded database summary for file_id: {file_id}"
                )
                # No need to re-extract summary as it's already loaded
            else:
                logging.info(f"Generating database summary for file_id: {file_id}")
                db_info = self._get_table_info_for_file(file_id)
                self.database_summaries[file_id] = db_info
                logging.info(
                    f"Database summary generated successfully for file_id: {file_id}"
                )

            # Initialize SQL agent after database is prepared
            logging.info(f"Initializing SQL agent for file_id: {file_id}")
            self._initialize_agent_for_file(file_id)
            logging.info(f"SQL agent initialized successfully for file_id: {file_id}")

            logging.info(
                f"=== Database initialization completed successfully for file_id: {file_id} ==="
            )

        except Exception as e:
            logging.error(
                f"Error during database initialization for file_id {file_id}: {str(e)}"
            )
            logging.error(f"Exception type: {type(e).__name__}")
            logging.error(f"Exception details: {str(e)}")
            import traceback

            logging.error(f"Full traceback: {traceback.format_exc()}")
            raise

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
                max_output_tokens=4096,
                top_p=1,
                top_k=40,
            )

        # Handle Anthropic (Vertex) models
        if self.model_choice in ["claude-sonnet-4@20250514", "claude-sonnet-4-5"]:
            model_config = self.config.anthropic
            if not model_config:
                raise ValueError("Configuration for Anthropic model not found")

            # Map model choice to actual model name
            model_mapping = {
                "claude-sonnet-4@20250514": model_config.model_sonnet,
                "claude-sonnet-4-5": model_config.model_sonnet_45,
            }
            model_name = model_mapping[self.model_choice]

            logging.info(f"Using Anthropic Vertex model: {model_name}")
            return ChatAnthropicVertex(
                model=model_name,
                project=model_config.project,
                location=model_config.location,
                temperature=self.temperature,
                max_output_tokens=4096,
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
        Cleans SQL query by removing markdown/code-fence formatting and stray wrappers.

        Args:
            query (str): The SQL query that might contain markdown formatting or backticks.

        Returns:
            str: Cleaned SQL query without markdown/backticks/wrappers.
        """
        if query is None:
            return query

        cleaned = str(query).strip()

        # Normalize newlines
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n").strip()

        # Remove all code-fence openings like ``` or ```sql (any language)
        import re

        cleaned = re.sub(r"```[a-zA-Z0-9_\-]*\n?", "", cleaned)
        # Remove any remaining closing fences
        cleaned = cleaned.replace("```", "")

        # Remove leading inline backticks that wrap the entire query
        if cleaned.startswith("`") and cleaned.endswith("`"):
            cleaned = cleaned[1:-1].strip()

        # Remove any stray inline backticks left anywhere
        cleaned = cleaned.replace("`", "")

        # If the first line is a language hint like 'sql', 'mysql', 'sqlite', drop it
        lines = cleaned.split("\n")
        if lines and lines[0].strip().lower() in {"sql", "mysql", "sqlite"}:
            cleaned = "\n".join(lines[1:]).strip()

        # Trim surrounding quotes if the whole string is quoted
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (
            cleaned.startswith("'") and cleaned.endswith("'")
        ):
            cleaned = cleaned[1:-1].strip()

        # Strip trailing semicolon (not required for execution)
        if cleaned.endswith(";"):
            cleaned = cleaned[:-1].strip()

        # As a final guard, try to extract the SQL starting at a known keyword
        # to handle any leading commentary that slipped through
        keyword_match = re.search(
            r"\b("
            r"SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|PRAGMA|"
            r"EXPLAIN|REPLACE|VACUUM|ATTACH|DETACH|BEGIN|COMMIT|ROLLBACK"
            r")\b",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if keyword_match:
            cleaned = cleaned[keyword_match.start() :].strip()

        # ENFORCE: cap result size to 25 rows for safety and consistency
        try:
            # Handle patterns:
            # 1) LIMIT count
            # 2) LIMIT count OFFSET offset
            # 3) LIMIT offset, count
            # Use case-insensitive matching and preserve OFFSET when present
            limit_count_pattern = re.compile(
                r"(?is)\bLIMIT\s+(\d+)\s*(?!,|OFFSET)(?:\b|$)"
            )
            limit_with_offset_pattern = re.compile(
                r"(?is)\bLIMIT\s+(\d+)\s+OFFSET\s+(\d+)\b"
            )
            limit_offset_count_pattern = re.compile(
                r"(?is)\bLIMIT\s+(\d+)\s*,\s*(\d+)\b"
            )

            def replace_limit_count(m):
                count = int(m.group(1))
                new_count = min(count, 25)
                return f"LIMIT {new_count}"

            def replace_limit_with_offset(m):
                count = int(m.group(1))
                offset = int(m.group(2))
                new_count = min(count, 25)
                return f"LIMIT {new_count} OFFSET {offset}"

            def replace_limit_offset_count(m):
                offset = int(m.group(1))
                count = int(m.group(2))
                new_count = min(count, 25)
                return f"LIMIT {offset}, {new_count}"

            original_cleaned = cleaned
            # Apply patterns; order matters to avoid partial overlaps
            cleaned = limit_with_offset_pattern.sub(replace_limit_with_offset, cleaned)
            cleaned = limit_offset_count_pattern.sub(
                replace_limit_offset_count, cleaned
            )
            cleaned = limit_count_pattern.sub(replace_limit_count, cleaned)

            # If no LIMIT found at all, append LIMIT 25 (try before final semicolon if present)
            if re.search(r"(?is)\bLIMIT\b", cleaned) is None:
                cleaned = f"{cleaned} LIMIT 25"

            if original_cleaned != cleaned:
                logging.debug(
                    f"SQL limit normalized to 25 rows. Before: {original_cleaned[:200]} | After: {cleaned[:200]}"
                )
        except Exception:
            # If limit enforcement fails for any reason, still proceed with cleaned query
            pass

        return cleaned

    def _initialize_agent_for_file(self, file_id: str):
        """Initializes the SQL agent for a specific file's database.

        Args:
            file_id (str): The file ID whose SQL agent to initialize
        """
        if file_id not in self.dbs:
            raise ValueError(f"No database initialized for file_id: {file_id}")

        toolkit = SQLDatabaseToolkit(db=self.dbs[file_id], llm=self.llm)
        agent_prefix = None
        if self.custom_gpt and self.system_prompt:
            agent_prefix = self.system_prompt

        self.agents[file_id] = create_sql_agent(
            llm=self.llm,
            toolkit=toolkit,
            verbose=True,
            handle_parsing_errors=True,
            prefix=agent_prefix,
            agent_executor_kwargs={
                "handle_parsing_errors": True,
                "return_intermediate_steps": True,
            },
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

            def wrapped_run(command: str, fetch: str = "all", **kwargs) -> str:
                logging.debug(f"Executing SQL via wrapped_run: {command}")
                try:
                    cleaned_command = self._clean_sql_query(command)
                    result = original_run_method(cleaned_command, fetch)
                    logging.debug(f"SQL result (first 100 chars): {str(result)[:100]}")
                    return result
                except Exception as e:
                    logging.error(
                        f"Error in wrapped_run during SQL execution for command '{command}': {str(e)}"
                    )
                    raise

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
                "max_iterations": 10,
                "early_stopping_method": "generate",
            },
        )

    def _patch_db_run_for_file(self, file_id: str) -> None:
        """
        Patch the SQLDatabase.run for a specific file so that every execution is cleaned.
        """
        if file_id not in self.dbs or not self.dbs[file_id]:
            return

        db = self.dbs[file_id]

        # Avoid double-patching
        if getattr(db, "_run_is_patched", False):
            return

        original_run = db.run

        def wrapped_run(command: str, fetch: str = "all", **kwargs):
            try:
                cleaned_command = self._clean_sql_query(command)
                return original_run(cleaned_command, fetch)
            except Exception:
                # If cleaning or execution fails, bubble up the original exception
                raise

        db.run = wrapped_run
        setattr(db, "_run_is_patched", True)
        logging.info(f"Patched db.run for file_id: {file_id}")

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
                        text(f"SELECT * FROM `{self.table_name}` LIMIT 2")
                    )
                    rows = result.fetchall()
                    logging.info(f"First 2 rows of '{self.table_name}' table: {rows}")
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
                    text(f'SELECT * FROM "{table_name}" LIMIT 2')
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

    def get_answer(self, question: Union[str, List[str]]) -> Union[str, dict]:
        """
        Processes a user's question and returns an answer based on the database content.
        For multi-file mode, it attempts to determine which database to query or uses a combined approach.
        Now supports conversation history for contextual understanding.

        Args:
            question: Either a single question string or a list of conversation messages.
                     If a list is provided, the last item is the current question and
                     previous items are conversation history.

        Returns:
            Union[str, dict]: For tabular data, returns dict with 'answer' and 'intermediate_steps'.
                             For multi-file, returns string (to maintain compatibility).
                             Returns error message string if processing fails.
        """
        try:
            # Extract the actual question text for multi-file routing and forced answers
            if isinstance(question, list):
                actual_question_text = question[-1]
            else:
                actual_question_text = question

            if self.is_multi_file and len(self.file_ids) > 1:
                return self._get_multi_file_answer(actual_question_text)
            else:
                # Standard single-file approach (also works when only one file in multi-file mode)
                # Pass the full question (which could be a list with history) to ask_question
                result = self.ask_question(question)
                if result:
                    logging.info("Direct answer")
                    return result  # Return the dict with answer and intermediate_steps
                else:
                    logging.info("Forced answer")
                    forced_result = self.get_forced_answer(actual_question_text, "")
                    return forced_result  # Return the dict from get_forced_answer
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
                try:
                    response = get_gemini_non_rag_response(
                        self.config, prompt, self.model_choice
                    )
                except GeminiSafetyFilterError as e:
                    logging.warning(
                        f"Safety filter blocked database suggestion: {str(e)}"
                    )
                    return None  # Return None if safety filter blocks
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

    def ask_question(self, question: Union[str, List[str]]) -> Optional[dict]:
        """
        Enhanced question processing with intelligent optimization and SQL filtering.
        Now supports conversation history for contextual question resolution.

        Args:
            question: Either a single question string or a list of conversation messages.
                     If a list is provided, the last item is the current question and
                     previous items are conversation history (alternating user/assistant).

        Returns:
            Optional[dict]: Dictionary with 'answer' and 'intermediate_steps' keys, or None if processing fails.
        """
        if not self.agent:
            self._initialize_agent()

        try:
            # Prepare database_info for formatting and context usage
            database_info = {}
            if (
                self.all_file_infos
                and self.file_id
                and self.file_id in self.all_file_infos
            ):
                file_info = self.all_file_infos.get(self.file_id, {})
                database_info = file_info.get("database_summary", {})

            # Step 1: Handle conversation history if provided
            if isinstance(question, list) and len(question) > 1:
                # Extract conversation history and current question
                conversation_history = question[:-1]
                current_question = question[-1]

                logging.info(
                    f"Processing question with {len(conversation_history)} history messages"
                )

                # Retrieve cached previous formatted question for holistic context if available
                previous_formatted_question = self.previous_formatted_by_file.get(
                    self.file_id, ""
                )
                if previous_formatted_question:
                    logging.info(
                        f"Previous formatted question (cached): {previous_formatted_question}"
                    )

                # Resolve contextual references in the current question (with holistic context)
                resolved_question = resolve_question_with_history(
                    conversation_history,
                    current_question,
                    previous_formatted_question,
                    self.previous_resolved_by_file.get(self.file_id, ""),
                )

                # Use the resolved question for further processing
                question_to_process = resolved_question
            elif isinstance(question, list) and len(question) == 1:
                # Single item in list, extract it
                question_to_process = question[0]
            else:
                # String question, use as-is (backward compatibility)
                question_to_process = question

            # TODO: In future we can insert an llm agent here to verify if it needs answer from database or
            # using the last answer can further explain. example question : I want to know more specifically
            # how to implement and connect this database to my application.
            # User here Wants answer in detail based on History and non sql answer.

            # Step 2: Use enhanced format_question with intelligent context analysis
            format_result = format_question(
                database_info, question_to_process, self.model_choice
            )
            formatted_question = format_result["formatted_question"]
            needs_sql = format_result["needs_sql"]
            classification = format_result.get("classification", {})
            logging.info(f"Needs SQL: {needs_sql}")
            if not needs_sql:
                # Cache formatted for next turn and return direct summary
                logging.info("Direct answer provided from database summary")
                try:
                    self.previous_formatted_by_file[self.file_id] = formatted_question
                except Exception:
                    pass
                return {"answer": formatted_question, "intermediate_steps": None}

            # Get query classification for appropriate response formatting
            logging.info(f"Formatted question: {formatted_question}")
            query_type = classification.get("category", "unknown")
            language = classification.get("language", "en")

            # Cache formatted question for next turn
            try:
                self.previous_formatted_by_file[self.file_id] = formatted_question
                # Save resolved (what we actually used) for next turn anchoring
                self.previous_resolved_by_file[self.file_id] = question_to_process
            except Exception:
                pass

            # Execute SQL query through agent
            logging.info("Executing query through SQL agent")
            try:
                response = self.agent.invoke({"input": formatted_question})
                return self._process_agent_response(
                    response, question_to_process, query_type, language
                )

            except Exception as agent_error:
                return self._handle_agent_error(
                    agent_error,
                    formatted_question,
                    question_to_process,
                    query_type,
                    language,
                )

        except Exception as e:
            logging.error(f"An error occurred while processing the question: {str(e)}")
            raise

    def _process_agent_response(
        self, response: dict, question: str, query_type: str, language: str
    ) -> dict:
        """
        Process the agent response and format it appropriately.

        Args:
            response: The agent response dictionary
            question: Original user question
            query_type: Type of query for formatting
            language: Language for response

        Returns:
            dict: Dictionary with 'answer' and 'intermediate_steps' keys
        """
        # Extract the final answer and intermediate steps
        final_answer = response.get("output", "No final answer found")
        logging.info(f"Final answer: {final_answer}")
        intermediate_steps = response.get("intermediate_steps", [])

        # Debug: Log the structure of intermediate_steps for troubleshooting
        logging.debug("Intermediate steps structure:")
        for idx, step in enumerate(intermediate_steps):
            logging.debug(f"  Step {idx}: {type(step)}")
            if isinstance(step, tuple) and len(step) >= 2:
                action, observation = step[0], step[1]
                logging.debug(
                    f"    Action type: {type(action)}, attributes: {dir(action)}"
                )
                if hasattr(action, "tool"):
                    logging.debug(f"    Tool: {action.tool}")
                if hasattr(action, "log"):
                    logging.debug(f"    Has log: {bool(action.log)}")
                if hasattr(action, "message"):
                    logging.debug(f"    Has message: {bool(action.message)}")
                    if hasattr(action.message, "content"):
                        logging.debug(
                            f"    Message content: {action.message.content[:100]}..."
                        )
                logging.debug(f"    Observation type: {type(observation)}")
                logging.debug(f"    Observation preview: {str(observation)[:200]}...")

        logging.info("Formatting result from intermediate steps and final answer")
        truncated_context = self._truncate_intermediate_steps(
            intermediate_steps, final_answer, 50000, query_type
        )

        # Extract table name from SQL query in intermediate steps
        table_name = None
        for step in intermediate_steps:
            if isinstance(step, tuple) and len(step) >= 2:
                action = step[0]
                if hasattr(action, "tool") and action.tool == "sql_db_query":
                    sql_query = action.tool_input
                    table_name = self._extract_table_name_from_sql(sql_query)
                    if table_name:
                        break

        # Get column headers from database info for better formatting
        column_headers = self._get_column_headers_for_prompt(table_name)

        # Add column headers on the next line after truncated context
        if column_headers:
            truncated_context = (
                f"{truncated_context}\n\nColumn Headers: {column_headers}"
            )

        base_prompt = PromptBuilder.build_forced_answer_prompt(
            question, truncated_context, query_type, language
        )

        # Format the response using the appropriate model
        if self.model_choice.startswith("gemini"):
            try:
                formatted_answer = get_gemini_non_rag_response(
                    self.config, base_prompt, self.model_choice
                )
            except GeminiSafetyFilterError as e:
                logging.warning(
                    f"Safety filter blocked response in CSV handler: {str(e)}"
                )
                formatted_answer = f"I apologize, but I cannot provide a response to this question. {str(e)}"
        else:
            formatted_answer = get_azure_non_rag_response(self.config, base_prompt)

        # Format intermediate steps for display
        formatted_steps = (
            self._format_intermediate_steps(intermediate_steps)
            if intermediate_steps
            else None
        )

        return {"answer": formatted_answer, "intermediate_steps": formatted_steps}

    def _handle_agent_error(
        self,
        agent_error: Exception,
        formatted_question: str,
        question: str,
        query_type: str,
        language: str,
    ) -> dict:
        """
        Handle agent execution errors with fallback strategies.

        Args:
            agent_error: The exception from agent execution
            formatted_question: The formatted question that was sent to agent
            question: Original user question
            query_type: Type of query for formatting
            language: Language for response

        Returns:
            dict: Dictionary with 'answer' and 'intermediate_steps' keys
        """
        error_msg = str(agent_error)
        logging.warning(f"Agent execution error: {error_msg}")

        # Check if this is a parsing error with actual data
        if (
            "Could not parse LLM output" in error_msg
            and "These are the details" in error_msg
        ):
            # Extract the actual response from the error message
            start_idx = error_msg.find("I have successfully retrieved")
            if start_idx != -1:
                # Extract the formatted response from the error message
                actual_response = error_msg[start_idx:]
                # Clean up the response by removing the error prefix
                if "Could not parse LLM output" in actual_response:
                    # Find where the actual response starts
                    response_start = actual_response.find(
                        "I have successfully retrieved"
                    )
                    if response_start != -1:
                        actual_response = actual_response[response_start:]

                logging.info("Extracted response from parsing error")
                return {"answer": actual_response, "intermediate_steps": None}

        # If we can't extract the response, try to get it from intermediate steps
        try:
            # Try to execute the query directly to get the raw data
            logging.info("Attempting direct SQL execution as fallback")
            if hasattr(self, "db") and self.db:
                # Execute the SQL query directly
                sql_query = self._extract_sql_from_formatted_question(
                    formatted_question
                )
                if sql_query:
                    # Prefer engine-based execution to capture accurate headers
                    if hasattr(self, "engine") and self.engine is not None:
                        try:
                            with self.engine.connect() as connection:
                                cleaned_sql = self._clean_sql_query(sql_query)
                                result = connection.execute(text(cleaned_sql))
                                headers = list(result.keys())
                                rows = [list(row) for row in result.fetchall()]
                                return {
                                    "answer": [headers, *rows],
                                    "intermediate_steps": None,
                                }
                        except Exception as engine_exec_error:
                            logging.warning(
                                f"Engine execution failed in fallback, trying db.run: {str(engine_exec_error)}"
                            )

                    raw_result = self.db.run(sql_query)
                    logging.info(f"Direct SQL result: {str(raw_result)[:500]}...")

                    # Extract table name from SQL query
                    table_name = self._extract_table_name_from_sql(sql_query)

                    # Format the raw result with column headers if available
                    column_headers = self._get_column_headers_for_prompt(table_name)

                    # Add column headers on the next line if available
                    context_with_headers = (
                        f"SQL Query: {sql_query}\nResult: {raw_result}"
                    )
                    if column_headers:
                        context_with_headers = f"{context_with_headers}\n\nColumn Headers: {column_headers}"

                    base_prompt = PromptBuilder.build_forced_answer_prompt(
                        question,
                        context_with_headers,
                        query_type,
                        language,
                    )

                    if self.model_choice.startswith("gemini"):
                        try:
                            formatted_answer = get_gemini_non_rag_response(
                                self.config, base_prompt, self.model_choice
                            )
                        except GeminiSafetyFilterError as e:
                            logging.warning(
                                f"Safety filter blocked fallback response: {str(e)}"
                            )
                            formatted_answer = (
                                f"I apologize, but I cannot provide a response to "
                                f"this question. {str(e)}"
                            )
                    else:
                        formatted_answer = get_azure_non_rag_response(
                            self.config, base_prompt
                        )

                    return {"answer": formatted_answer, "intermediate_steps": None}
        except Exception as fallback_error:
            logging.error(f"Fallback execution also failed: {str(fallback_error)}")

        # If all else fails, raise the original error
        raise agent_error

    def _extract_table_name_from_sql(self, sql_query: str) -> Optional[str]:
        """
        Extract the table name from a SQL query.

        Args:
            sql_query: The SQL query string

        Returns:
            Optional[str]: The extracted table name or None if not found
        """
        if not sql_query:
            return None

        try:
            import re

            # Remove comments and normalize whitespace
            sql_query = re.sub(r"--.*$", "", sql_query, flags=re.MULTILINE)
            sql_query = re.sub(r"/\*.*?\*/", "", sql_query, flags=re.DOTALL)
            sql_query = " ".join(sql_query.split())

            # Pattern to match table name after FROM clause
            # Matches: FROM table_name, FROM `table_name`, FROM "table_name", etc.
            pattern = r"\bFROM\s+([`\"]?)(\w+)\1"
            match = re.search(pattern, sql_query, re.IGNORECASE)

            if match:
                table_name = match.group(2)
                logging.info(f"Extracted table name from SQL: {table_name}")
                return table_name

            # If no match, try to find any table name in the query
            # by checking against known table names
            if self.file_id and self.file_id in self.database_summaries:
                db_summary = self.database_summaries[self.file_id]
                table_names = db_summary.get("table_names", [])
                sql_upper = sql_query.upper()
                for table_name in table_names:
                    if table_name.upper() in sql_upper:
                        logging.info(
                            f"Found table name by matching known tables: {table_name}"
                        )
                        return table_name

            return None

        except Exception as e:
            logging.warning(f"Error extracting table name from SQL: {str(e)}")
            return None

    def _get_column_headers_for_prompt(
        self, table_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Extract column headers from the database information for inclusion in prompts.

        Args:
            table_name: Optional. The specific table name to get headers from.
                       If None, defaults to the first table.

        Returns:
            Optional[str]: Comma-separated column headers or None if not available
        """
        try:
            if not self.file_id or self.file_id not in self.database_summaries:
                return None

            db_summary = self.database_summaries[self.file_id]
            if not db_summary or "tables" not in db_summary:
                return None

            # Find the specific table or use the first one
            table = None
            if table_name:
                # Search for the table by name
                for tbl in db_summary["tables"]:
                    if tbl.get("name") == table_name:
                        table = tbl
                        break
                if not table:
                    logging.warning(
                        f"Table '{table_name}' not found, using first table"
                    )
                    table = db_summary["tables"][0] if db_summary["tables"] else None
            else:
                # Default to first table
                table = db_summary["tables"][0] if db_summary["tables"] else None

            if not table or "columns" not in table:
                return None

            # Extract column names
            column_names = [
                col.get("name", "") for col in table["columns"] if isinstance(col, dict)
            ]
            if not column_names:
                return None

            return ", ".join(column_names)

        except Exception as e:
            logging.warning(f"Error extracting column headers: {str(e)}")
            return None

    def get_forced_answer(
        self, question: str, answer: str, language: str = "en"
    ) -> dict:
        """
        Attempts to extract an answer from a given text when a direct answer is not available.

        Args:
            question (str): The original question asked by the user.
            answer (str): The text to search for an answer.
            language (str): The language of the user's question.

        Returns:
            dict: Dictionary with 'answer' and 'intermediate_steps' keys.
        """
        try:
            # Try to extract table name from the answer (which might contain SQL)
            table_name = self._extract_table_name_from_sql(answer)

            # Get column headers if available for better formatting
            column_headers = self._get_column_headers_for_prompt(table_name)

            # Add column headers on the next line if available
            answer_with_headers = answer
            if column_headers:
                answer_with_headers = f"{answer}\n\nColumn Headers: {column_headers}"

            base_prompt = PromptBuilder.build_forced_answer_prompt(
                question, answer_with_headers, "unknown", language
            )

            # Use the appropriate model based on model_choice
            if self.model_choice.startswith("gemini"):
                try:
                    formatted_answer = get_gemini_non_rag_response(
                        self.config, base_prompt, self.model_choice
                    )
                except GeminiSafetyFilterError as e:
                    logging.warning(
                        f"Safety filter blocked response in get_forced_answer: {str(e)}"
                    )
                    formatted_answer = f"I apologize, but I cannot provide a response to this question. {str(e)}"
            else:
                formatted_answer = get_azure_non_rag_response(self.config, base_prompt)

            return {"answer": formatted_answer, "intermediate_steps": None}
        except Exception as e:
            logging.error(f"Error in get_forced_answer: {str(e)}")
            return {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "intermediate_steps": None,
            }

    def _truncate_intermediate_steps(
        self,
        intermediate_steps: list,
        final_answer: str,
        max_context_tokens: int = 50000,
        query_type: str = "unknown",
    ) -> str:
        """
        Intelligently truncate intermediate steps to prevent token overflow while preserving key information.

        Args:
            intermediate_steps: List of intermediate steps from agent execution
            final_answer: The final answer from the agent
            max_context_tokens: Maximum estimated tokens for context (roughly 4 chars per token)
            query_type: Type of query to determine truncation strategy

        Returns:
            str: Truncated context that fits within token limits
        """

        # Estimate token count (rough approximation: 4 characters = 1 token)
        def estimate_tokens(text: str) -> int:
            return len(str(text)) // 4

        final_answer_str = str(final_answer)
        final_tokens = estimate_tokens(final_answer_str)

        # Special handling for FILTERED_SEARCH queries - preserve complete data
        if query_type == "FILTERED_SEARCH":
            logging.info(
                "FILTERED_SEARCH query detected - preserving complete final answer"
            )
            # For filtered search, prioritize preserving the complete final answer
            # Only truncate intermediate steps, never the final answer
            buffer_size = 1000  # Smaller buffer for FILTERED_SEARCH
            max_step_chars = 300  # More aggressive truncation of intermediate steps
            max_steps_to_check = 2  # Check fewer steps
        else:
            # Standard handling for other query types
            buffer_size = 2000  # Standard buffer
            max_step_chars = 500  # Less aggressive truncation
            max_steps_to_check = 3  # Check more steps

            # If final answer itself is extremely large, truncate it first
            if final_tokens > max_context_tokens * 0.8:
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
        available_tokens = max_context_tokens - final_tokens - buffer_size

        if available_tokens <= 500:  # Very limited space
            logging.warning(
                f"Very limited tokens available ({available_tokens}), returning minimal context"
            )
            return (
                "FINAL ANSWER:\n" + final_answer_str + "\n\n"
                "INTERMEDIATE_STEPS:\n[omitted due to token limits]"
            )

        # Convert intermediate steps to string and get essential parts
        intermediate_str = str(intermediate_steps)

        if estimate_tokens(intermediate_str) <= available_tokens:
            # If it fits, return everything
            return (
                "FINAL ANSWER:\n" + final_answer_str + "\n\n"
                "INTERMEDIATE_STEPS:\n" + intermediate_str
            )

        # Need to truncate - extract key information
        essential_info = self._extract_essential_info_from_steps(
            intermediate_steps, available_tokens, max_steps_to_check, max_step_chars
        )

        # If no essential info found, use a summary approach
        if not essential_info:
            if query_type == "FILTERED_SEARCH":
                essential_info = [
                    "Database query executed successfully with filtered search results."
                ]
            else:
                essential_info = [
                    "Database query executed successfully. Results processed and formatted."
                ]

        # Combine with final answer
        context = "\n".join(essential_info)
        if context:
            return (
                "FINAL ANSWER:\n" + final_answer_str + "\n\n"
                "INTERMEDIATE_STEPS (essential):\n" + context
            )
        else:
            return (
                "FINAL ANSWER:\n" + final_answer_str + "\n\n"
                "INTERMEDIATE_STEPS:\n[not required]"
            )

    def _extract_essential_info_from_steps(
        self,
        intermediate_steps: list,
        available_tokens: int,
        max_steps: int,
        max_chars: int,
    ) -> list:
        """
        Extract essential information from intermediate steps within token limits.

        Args:
            intermediate_steps: List of intermediate steps
            available_tokens: Available tokens for context
            max_steps: Maximum number of steps to check
            max_chars: Maximum characters per step

        Returns:
            list: List of essential information strings
        """
        essential_info = []

        # Look for SQL queries and their results in intermediate steps
        if isinstance(intermediate_steps, list):
            for step in intermediate_steps[-max_steps:]:  # Check last N steps
                step_str = str(step)
                # Extract SQL queries (usually contain SELECT, FROM, WHERE)
                if any(
                    keyword in step_str.upper()
                    for keyword in ["SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY"]
                ):
                    # Keep SQL queries as they're essential for understanding
                    # Use the estimate_tokens function from the parent method
                    current_length = len("\n".join(essential_info)) // 4
                    if current_length + len(step_str) // 4 < available_tokens:
                        essential_info.append(
                            f"SQL Query executed: {step_str[:max_chars]}..."
                        )
                    else:
                        # If adding this would exceed limit, truncate the step
                        remaining_tokens = available_tokens - current_length
                        if (
                            remaining_tokens > 50
                        ):  # Only add if we have reasonable space
                            # Ensure remaining_tokens is an integer for string slicing
                            remaining_chars = int(remaining_tokens * 2)
                            truncated_step = (
                                step_str[:remaining_chars] + "... [truncated]"
                            )
                            essential_info.append(f"SQL Query: {truncated_step}")
                        break

        return essential_info

    def _extract_thought_from_action(
        self, action: Any, observation: Any
    ) -> Optional[str]:
        """Extract the agent's thought text from action/observation in a robust way."""
        try:
            if hasattr(action, "log") and action.log:
                return action.log
            if hasattr(action, "message"):
                msg = getattr(action, "message")
                if hasattr(msg, "content") and msg.content:
                    return msg.content
                if (
                    hasattr(msg, "additional_kwargs")
                    and isinstance(msg.additional_kwargs, dict)
                    and "thought" in msg.additional_kwargs
                ):
                    return msg.additional_kwargs.get("thought")
            if hasattr(action, "tool_input") and isinstance(action.tool_input, dict):
                if "thought" in action.tool_input:
                    return action.tool_input.get("thought")
            obs_str = str(observation)
            if "Thought:" in obs_str or "in_tableThought:" in obs_str:
                for line in obs_str.split("\n"):
                    if "Thought:" in line or "in_tableThought:" in line:
                        return (
                            line.replace("Thought:", "")
                            .replace("in_tableThought:", "")
                            .strip()
                        )
            act_str = str(action)
            if "Thought" in act_str or "thinking" in act_str.lower():
                return act_str
        except Exception:
            return None
        return None

    def _format_tool_input_block(self, tool_name: str, tool_input: Any) -> List[str]:
        """Format tool input with SQL-aware pretty printing."""
        lines: List[str] = []
        if tool_name in {"sql_db_query", "query_sql_db"} and isinstance(
            tool_input, str
        ):
            lines.append("\nSQL Query:")
            sql_clean = self._clean_sql_query(tool_input)
            for sql_line in sql_clean.split("\n"):
                lines.append(f"  {sql_line}")
        else:
            lines.append(f"Input: {str(tool_input)[:500]}")
        return lines

    def _format_observation_block(self, observation: Any) -> List[str]:
        """Format observation text with truncation and line splitting."""
        out: List[str] = ["\nObservation:"]
        obs_str = str(observation)
        if len(obs_str) > 2000:
            out.append(f"  {obs_str[:2000]}...")
            out.append(f"  [Truncated - {len(obs_str)} total characters]")
            return out
        obs_lines = obs_str.split("\n")
        for obs_line in obs_lines[:50]:
            out.append(f"  {obs_line}")
        if len(obs_lines) > 50:
            out.append(f"  ... [{len(obs_lines) - 50} more lines]")
        return out

    def _format_step_block(self, idx: int, step: Any) -> List[str]:
        """Format one intermediate step into a list of lines."""
        lines: List[str] = [f"Step {idx}:", "-" * 80]
        if not (isinstance(step, tuple) and len(step) >= 2):
            lines.append(f"  {str(step)[:500]}")
            lines.append("")
            return lines

        action, observation = step[0], step[1]
        logging.debug(f"Action type: {type(action)}, attributes: {dir(action)}")

        thought = self._extract_thought_from_action(action, observation)
        if thought:
            lines.append("Thought:")
            thought_lines = str(thought).split("\n")
            for tl in thought_lines[:10]:
                lines.append(f"  {tl}")
            if len(thought_lines) > 10:
                lines.append(f"  ... [{len(thought_lines) - 10} more lines]")
            lines.append("")

        if hasattr(action, "tool"):
            tool_name = action.tool
            lines.append(f"Tool: {tool_name}")
            if hasattr(action, "tool_input"):
                lines.extend(
                    self._format_tool_input_block(tool_name, action.tool_input)
                )
        else:
            lines.append(f"Action: {str(action)[:500]}")

        lines.extend(self._format_observation_block(observation))
        lines.append("")
        return lines

    def _format_intermediate_steps(self, intermediate_steps: list) -> str:
        """
        Format intermediate steps into human-readable text.

        Args:
            intermediate_steps: List of tuples (action, observation) from agent execution

        Returns:
            str: Formatted text showing agent actions, observations, and reasoning
        """
        if not intermediate_steps:
            return ""

        formatted_lines: List[str] = []
        formatted_lines.append("=" * 80)
        formatted_lines.append("AGENT EXECUTION TRACE")
        formatted_lines.append("=" * 80)
        formatted_lines.append("")

        for idx, step in enumerate(intermediate_steps, 1):
            try:
                formatted_lines.extend(self._format_step_block(idx, step))
            except Exception as e:
                logging.warning(f"Error formatting step {idx}: {str(e)}")
                formatted_lines.append(f"Step {idx}: [Error formatting step]")
                formatted_lines.append("")

        formatted_lines.append("=" * 80)
        formatted_lines.append(f"Total Steps: {len(intermediate_steps)}")
        formatted_lines.append("=" * 80)

        return "\n".join(formatted_lines)

    def _extract_sql_from_formatted_question(
        self, formatted_question: str
    ) -> Optional[str]:
        """
        Extract SQL query from formatted question if it's a simple SQL statement.

        Args:
            formatted_question: The formatted question that may contain SQL

        Returns:
            Optional[str]: The extracted SQL query or None if not found
        """
        # Check if the formatted question is already a SQL query
        if (
            formatted_question.strip()
            .upper()
            .startswith(("SELECT", "SHOW", "LIST", "FIND"))
        ):
            # Clean up the query
            query = formatted_question.strip()
            if query.endswith(";"):
                query = query[:-1]
            return query

        # Try to extract SQL from common patterns
        import re

        # Look for SQL patterns in the text
        sql_patterns = [
            r"SELECT\s+.*?FROM\s+.*?(?:LIMIT\s+\d+)?",
            r"SHOW\s+.*?FROM\s+.*?",
            r"LIST\s+.*?FROM\s+.*?",
        ]

        for pattern in sql_patterns:
            match = re.search(pattern, formatted_question, re.IGNORECASE)
            if match:
                return match.group(0)

        return None
