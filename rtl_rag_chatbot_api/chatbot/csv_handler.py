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
from rtl_rag_chatbot_api.chatbot.anthropic_handler import get_anthropic_non_rag_response
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
from rtl_rag_chatbot_api.chatbot.utils.language_detector import detect_lang
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
        logging.info("TABULAR FLOW: === Initializing TabularDataHandler ===")
        logging.info(f"TABULAR FLOW: Model choice: {model_choice}")
        logging.info(f"TABULAR FLOW: File ID: {file_id}")
        logging.info(f"TABULAR FLOW: File IDs: {file_ids}")
        logging.info(
            f"TABULAR FLOW: Database summaries provided: "
            f"{len(database_summaries_param) if database_summaries_param else 0}"
        )
        logging.info(
            f"TABULAR FLOW: All file infos provided: "
            f"{len(all_file_infos) if all_file_infos else 0}"
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
        # Initialize primary file attributes early so they always exist.
        # This prevents AttributeError when helper methods (like
        # _initialize_unified_database_session or _initialize_agent_for_file)
        # access self.file_id during multi-file initialization before it is
        # reassigned to the unified session ID.
        self.file_id = file_id
        self.file_ids = file_ids or []
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

        logging.info(f"TABULAR FLOW: Temperature set to: {self.temperature}")

        # Initialize LLM for all database operations before initializing databases
        logging.info("TABULAR FLOW: Initializing LLM...")
        self.llm = self._initialize_llm()
        logging.info("TABULAR FLOW: LLM initialized successfully")

        # Initialize database_summaries with pre-loaded data if provided
        self.database_summaries = {}
        if database_summaries_param:
            logging.info(
                f"TABULAR FLOW: Using pre-loaded database summaries for "
                f"{len(database_summaries_param)} files"
            )
            self.database_summaries = database_summaries_param

        # Store all_file_infos for context in responses
        self.all_file_infos = all_file_infos if all_file_infos else {}

        # Determine if we're in multi-file mode
        self.is_multi_file = bool(file_ids and len(file_ids) > 0)
        logging.info(f"TABULAR FLOW: Multi-file mode: {self.is_multi_file}")

        # Track unified-session specific identifiers for multi-file mode
        self.unified_session_id = None
        self.unified_session_dir = None

        if self.is_multi_file:
            # Multi-file mode
            self.file_ids = sorted(list(set(file_ids)))  # Ensure unique and sorted
            logging.info(
                f"TABULAR FLOW: TabularDataHandler initialized in multi-file mode "
                f"with {len(self.file_ids)} files"
            )

            if len(self.file_ids) == 1:
                # Treat a single file in multi-file request as standard single-file mode
                self.file_id = self.file_ids[0]
                logging.info(
                    f"TABULAR FLOW: Multi-file requested with single file_id; falling "
                    f"back to single-file initialization for file_id: {self.file_id}"
                )
                self._initialize_file_database(self.file_id)
            else:
                # Initialize unified SQLite database for all file_ids
                logging.info(
                    "TABULAR FLOW: Initializing unified SQLite database for "
                    "multi-file session"
                )
                self._initialize_unified_database_session()
        else:
            # Single file mode (backward compatible)
            self.file_id = file_id
            self.file_ids = [file_id] if file_id else []
            if file_id:
                logging.info(
                    f"TABULAR FLOW: Initializing single file database for "
                    f"file_id: {file_id}"
                )
                self._initialize_file_database(file_id)
            else:
                # Legacy path for tests or default initialization
                logging.info(
                    "TABULAR FLOW: Using legacy path for tests or default "
                    "initialization"
                )
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
                f"TABULAR FLOW: Setting up backward compatibility attributes for "
                f"file_id: {self.file_id}"
            )
            self.engine = self.engines.get(self.file_id)
            self.Session = self.sessions.get(self.file_id)
            self.db = self.dbs.get(self.file_id)
            self.agent = self.agents.get(self.file_id)

            if self.file_id in self.database_summaries:
                # Set primary DB info and table_info from pre-loaded summaries
                self.primary_db_info = self.database_summaries[self.file_id]
                logging.info(
                    f"TABULAR FLOW: Primary DB info set from database summaries for "
                    f"file_id: {self.file_id}"
                )
            else:
                # Generate and cache database summary when not pre-loaded
                logging.info(
                    f"TABULAR FLOW: No pre-loaded database summary found for "
                    f"file_id: {self.file_id}. Generating summary from database."
                )
                self.primary_db_info = self._get_table_info_for_file(self.file_id)
                self.database_summaries[self.file_id] = self.primary_db_info

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
                logging.info(f"TABULAR FLOW: Table name set to: {self.table_name}")
            else:
                error_msg = f"No tables found in the primary database for file_id: {self.file_id}"
                logging.error(error_msg)
                raise ValueError(error_msg)

        logging.info(
            "TABULAR FLOW: === TabularDataHandler initialization completed ==="
        )

    def _ensure_local_database(self, file_id: str) -> str:
        """
        Ensure that the SQLite database for the given file_id exists locally and is valid.

        This method is used by both single-file initialization and unified multi-file
        initialization to guarantee that the underlying tabular_data.db is present
        (downloading it from GCS if necessary) and passes basic validation.

        Args:
            file_id (str): The file ID whose database should be available locally.

        Returns:
            str: Absolute path to the local SQLite database file.
        """
        logging.info(f"Ensuring local database exists for file_id: {file_id}")

        data_dir = f"./chroma_db/{file_id}"
        db_path = os.path.join(data_dir, self.db_name)

        logging.info(f"Expected data directory: {data_dir}")
        logging.info(f"Expected database path: {db_path}")

        # If the data directory doesn't exist, download the artifacts from GCS
        if not os.path.exists(data_dir):
            logging.warning(
                f"Data directory {data_dir} not found locally. "
                f"Attempting to download from GCS for file_id: {file_id}"
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
                raise FileNotFoundError(
                    f"Data directory {data_dir} not found and could not be "
                    f"downloaded from GCS."
                ) from e

        # Validate database file and SQLite integrity
        self._validate_database_file(data_dir, db_path, file_id)
        self._validate_sqlite_database(db_path, file_id)

        return db_path

    def _initialize_unified_database_session(self):
        """
        Initialize a unified SQLite database for multi-file tabular chat.

        First checks if a pre-built unified database exists (created during upload).
        If found, uses it directly. Otherwise, creates one on-demand (fallback).

        The unified database contains all tables from individual file databases with:
            - Renamed tables to avoid conflicts: {filename}_{tablename}
            - Source tracking columns: _source_file_id, _source_filename
        """
        if not self.file_ids or len(self.file_ids) < 2:
            raise ValueError(
                "Unified database session requires at least two file_ids in multi-file mode"
            )

        try:
            from rtl_rag_chatbot_api.chatbot.unified_db_builder import (
                UnifiedDatabaseBuilder,
            )

            builder = UnifiedDatabaseBuilder()

            # Check for pre-built unified database first
            existing_unified = builder.check_unified_database_exists(self.file_ids)

            if existing_unified:
                # Use pre-built unified database (FAST PATH)
                logging.info(
                    f"Using pre-built unified database: "
                    f"{existing_unified['unified_session_id']}"
                )

                unified_file_id = existing_unified["unified_session_id"]
                unified_db_path = existing_unified["unified_db_path"]
                session_dir = existing_unified["session_dir"]

            else:
                # Fallback: Create unified database on-demand (SLOW PATH)
                logging.warning(
                    "No pre-built unified database found, creating on-demand. "
                    "This should only happen for legacy uploads."
                )

                unified_result = builder.build_unified_database(
                    self.file_ids, self.all_file_infos
                )

                unified_file_id = unified_result["unified_session_id"]
                unified_db_path = unified_result["unified_db_path"]
                session_dir = unified_result["session_dir"]

            # Create SQLAlchemy components for the unified database
            unified_db_url = f"sqlite:///{unified_db_path}"

            logging.info(
                f"Creating SQLAlchemy components for unified session_id: "
                f"{unified_file_id}"
            )
            self._create_database_components(unified_db_url, unified_file_id)

            # Initialize SQL agent for the unified database
            self._initialize_agent_for_file(unified_file_id)

            # Update primary identifiers to point to the unified database
            self.file_id = unified_file_id
            self.unified_session_id = unified_file_id
            self.unified_session_dir = session_dir
            self.data_dir = session_dir
            self.db_path = unified_db_path
            self.db_url = unified_db_url

            logging.info(
                f"Unified database session initialized with file_id: {self.file_id}"
            )

        except Exception as e:
            logging.error(
                f"Failed to initialize unified database session: {str(e)}",
                exc_info=True,
            )
            raise

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
        db_path = self._ensure_local_database(file_id)
        db_url = f"sqlite:///{db_path}"

        logging.info(f"Data directory: {data_dir}")
        logging.info(f"Database path: {db_path}")
        logging.info(f"Database URL: {db_url}")

        # Track primary paths for this file
        self.data_dir = data_dir
        self.db_path = db_path
        self.db_url = db_url

        try:
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
        """
        Cleanup database connections without deleting unified databases.

        Unified databases are persisted and reused across sessions,
        cleaned up later by scheduled tasks (similar to PDF chat flow).
        """
        # Dispose all SQLAlchemy engines to release connections
        try:
            for file_id, engine in getattr(self, "engines", {}).items():
                if engine:
                    try:
                        engine.dispose()
                        logging.info(f"Disposed engine for file_id: {file_id}")
                    except Exception as e:
                        logging.warning(
                            f"Error disposing engine for file_id {file_id}: {str(e)}"
                        )
        except Exception:
            # Best-effort cleanup; do not raise from destructor paths
            logging.warning(
                "Error while disposing engines during cleanup", exc_info=True
            )

        # NOTE: Unified session directories are NOT deleted here.
        # They persist for reuse when users chat again with the same file combination.
        # Cleanup is handled by scheduled tasks (similar to PDF embeddings).
        unified_dir = getattr(self, "unified_session_dir", None)
        if unified_dir:
            logging.info(f"Keeping unified session directory for reuse: {unified_dir}")

    def __del__(self):
        """
        Ensure cleanup on object destruction.

        Only disposes SQL connections; unified databases persist for reuse.
        """
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

        # Build enhanced agent prefix that emphasizes sample data usage
        base_prefix = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results
and return the answer to the input question.
Unless instructed otherwise, return only the final answer.

CRITICAL INSTRUCTIONS FOR ACCURATE SQL GENERATION:
1. ALWAYS use the sql_db_schema tool FIRST to see table structures and sample rows
2. The sample rows in sql_db_schema show ACTUAL data values from the database
3. When filtering by categorical values (e.g., asset names, symbols, product types):
   - EXAMINE the sample rows to see what values actually exist
   - Use the EXACT value format shown in samples (case-sensitive where needed)
   - Example: If samples show asset="Bitcoin", use WHERE asset = 'Bitcoin', NOT WHERE symbol = 'btc'
4. For timestamp/date filters:
   - Check the exact format in sample rows (e.g., "2025-12-03 00:01:00+00:00")
   - Match the format exactly in your WHERE clauses (include timezone offsets if present)
5. Match user terms to actual column values by checking samples BEFORE writing SQL

You have access to the following tools for interacting with the database:"""

        # Use custom prefix if provided, otherwise use enhanced base prefix
        if self.custom_gpt and self.system_prompt:
            agent_prefix = self.system_prompt
        else:
            agent_prefix = base_prefix

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

        # Build enhanced agent prefix that emphasizes sample data usage
        base_prefix = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results
and return the answer to the input question.
Unless instructed otherwise, return only the final answer.

CRITICAL INSTRUCTIONS FOR ACCURATE SQL GENERATION:
1. ALWAYS use the sql_db_schema tool FIRST to see table structures and sample rows
2. The sample rows in sql_db_schema show ACTUAL data values from the database
3. When filtering by categorical values (e.g., asset names, symbols, product types):
   - EXAMINE the sample rows to see what values actually exist
   - Use the EXACT value format shown in samples (case-sensitive where needed)
   - Example: If samples show asset="Bitcoin", use WHERE asset = 'Bitcoin', NOT WHERE symbol = 'btc'
4. For timestamp/date filters:
   - Check the exact format in sample rows (e.g., "2025-12-03 00:01:00+00:00")
   - Match the format exactly in your WHERE clauses (include timezone offsets if present)
5. Match user terms to actual column values by checking samples BEFORE writing SQL

You have access to the following tools for interacting with the database:"""

        toolkit = SQLDatabaseToolkit(
            db=self.db, llm=self.llm, handle_parsing_errors=True
        )
        self.agent = create_sql_agent(
            llm=self.llm,
            toolkit=toolkit,
            verbose=True,
            handle_parsing_errors=True,
            prefix=base_prefix,
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
                    try:
                        # Safe check: only compute stats for numeric types
                        if hasattr(column["type"], "python_type"):
                            py_type = column["type"].python_type
                            if py_type in (int, float):
                                stats = session.execute(
                                    text(
                                        f'SELECT MIN("{column["name"]}"), '
                                        f'MAX("{column["name"]}"), '
                                        f'AVG("{column["name"]}") '
                                        f'FROM "{table_name}"'
                                    )
                                ).fetchone()
                                column_stats[column["name"]] = {
                                    "min": stats[0],
                                    "max": stats[1],
                                    "avg": stats[2],
                                }
                    except (NotImplementedError, AttributeError) as e:
                        # Some column types don't implement python_type
                        logging.debug(
                            f"Skipping stats for column {column['name']} "
                            f"(type: {column['type']}): {str(e)}"
                        )
                        continue

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
        In both single-file and multi-file modes, a single SQL agent is used. In
        multi-file mode, this agent operates on a unified SQLite database that
        contains all tables from the selected files (with source-tracking columns).

        Args:
            question: Either a single question string or a list of conversation messages.
                     If a list is provided, the last item is the current question and
                     previous items are conversation history.

        Returns:
            Union[str, dict]: For tabular data, returns dict with 'answer' and 'intermediate_steps'.
                             Returns error message string if processing fails.
        """
        logging.info("TABULAR FLOW: === get_answer called ===")
        logging.info(
            f"TABULAR FLOW: Question type: {type(question)}, "
            f"is_list: {isinstance(question, list)}"
        )
        try:
            # Extract the actual question text for forced-answer fallback
            if isinstance(question, list):
                actual_question_text = question[-1]
                logging.info(
                    f"TABULAR FLOW: Extracted question from list: "
                    f"{actual_question_text[:100]}"
                )
            else:
                actual_question_text = question
                logging.info(
                    f"TABULAR FLOW: Direct question: {actual_question_text[:100]}"
                )

            # Single-agent approach for both single-file and multi-file (unified DB)
            logging.info("TABULAR FLOW: Calling ask_question...")
            result = self.ask_question(question)
            if result:
                logging.info("TABULAR FLOW: Direct answer from ask_question")
                return result  # Dict with 'answer' and 'intermediate_steps'

            logging.info("TABULAR FLOW: No direct answer, using forced answer fallback")
            # Use lingua-based detector here as well so forced answers match
            # the user's original question language.
            try:
                detected_lang_name = detect_lang(str(actual_question_text))
                lang_name_lower = (detected_lang_name or "").lower()
                if "english" in lang_name_lower:
                    user_language = "en"
                elif "german" in lang_name_lower:
                    user_language = "de"
                else:
                    user_language = "en"
            except Exception:
                user_language = "en"

            forced_result = self.get_forced_answer(
                actual_question_text, "", language=user_language
            )
            return forced_result  # Dict from get_forced_answer
        except Exception as e:
            logging.error(f"TABULAR FLOW: Error in get_answer: {str(e)}")
            return f"An error occurred while processing your question: {str(e)}"

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

        # In multi-file mode, remind about unified naming and source columns
        if self.is_multi_file:
            context_parts.append(
                "Multi-file unified schema: tables are named `{filename}_{tablename}` "
                "and include `_source_file_id`, `_source_filename` for attribution."
            )

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
        logging.info("TABULAR FLOW: === ask_question called ===")
        if not self.agent:
            logging.info("TABULAR FLOW: No agent found, initializing...")
            self._initialize_agent()

        try:
            database_info = self._prepare_database_info()
            (
                question_to_process,
                resolved_question,
            ) = self._resolve_question_with_history_if_needed(question)

            # Step 2: Detect user language once from the raw/resolved question
            # using lingua-based detector (not influenced by schema/DB context).
            user_language = self._detect_user_language(question_to_process)

            # Step 3: Use enhanced format_question with intelligent context analysis
            logging.info("TABULAR FLOW: Formatting question for SQL processing...")
            format_result = format_question(
                database_info,
                question_to_process,
                self.model_choice,
                user_language=user_language,
            )
            formatted_question = format_result["formatted_question"]
            needs_sql = format_result["needs_sql"]
            classification = format_result.get("classification", {})
            logging.info(
                f"TABULAR FLOW: Formatted question: {formatted_question[:100]}"
            )
            logging.info(f"TABULAR FLOW: Needs SQL: {needs_sql}")
            if not needs_sql:
                # Cache formatted for next turn and return direct summary
                logging.info(
                    "TABULAR FLOW: Direct answer provided from database summary "
                    "(no SQL needed)"
                )
                try:
                    self.previous_formatted_by_file[self.file_id] = formatted_question
                except Exception:
                    pass
                return {"answer": formatted_question, "intermediate_steps": None}

            # Get query classification for appropriate response formatting.
            # Language is sourced from detect_lang via user_language override above.
            query_type = classification.get("category", "unknown")
            language = classification.get("language", "en")
            logging.info(
                f"TABULAR FLOW: Query type: {query_type}, Language: {language}"
            )

            # Cache formatted question for next turn
            try:
                self.previous_formatted_by_file[self.file_id] = formatted_question
                # Save resolved (what we actually used) for next turn anchoring
                self.previous_resolved_by_file[self.file_id] = question_to_process
            except Exception:
                pass

            # Execute SQL query through agent
            logging.info("TABULAR FLOW: Executing query through SQL agent...")
            try:
                logging.info(
                    f"TABULAR FLOW: Invoking agent with formatted question: "
                    f"{formatted_question[:100]}"
                )
                response = self.agent.invoke({"input": formatted_question})
                logging.info("TABULAR FLOW: Agent invocation completed successfully")
                return self._process_agent_response(
                    response,
                    question_to_process,
                    query_type,
                    language,
                    resolved_question=resolved_question
                    if isinstance(question, list) and len(question) > 1
                    else "",
                    formatted_question=formatted_question,
                )

            except Exception as agent_error:
                logging.error(f"TABULAR FLOW: Agent error: {str(agent_error)}")
                return self._handle_agent_error(
                    agent_error,
                    formatted_question,
                    question_to_process,
                    query_type,
                    language,
                )

        except Exception as e:
            logging.error(
                f"TABULAR FLOW: Error in ask_question: {str(e)}", exc_info=True
            )
            raise

    def _prepare_database_info(self) -> dict:
        """
        Prepare database_info for formatting and context usage.
        """
        logging.info("TABULAR FLOW: Preparing database info for formatting...")
        database_info: Dict[str, Any] = {}

        try:
            # Try to get database_info from all_file_infos first
            if (
                self.all_file_infos
                and self.file_id
                and self.file_id in self.all_file_infos
            ):
                file_info = self.all_file_infos.get(self.file_id, {})
                database_info = file_info.get("database_summary", {})
            else:
                # Fallback: check database_summaries for unified sessions
                # This handles cases where file_id is a unified_session_id
                if self.file_id in self.database_summaries:
                    database_info = self.database_summaries[self.file_id]
                    logging.info(
                        f"Using database summary from cache for {self.file_id}"
                    )
        except Exception as e:
            logging.error(
                f"TABULAR FLOW: Error preparing database info: {str(e)}", exc_info=True
            )
        return database_info

    def _resolve_question_with_history_if_needed(
        self, question: Union[str, List[str]]
    ) -> tuple[str, str]:
        """
        Resolve contextual references using conversation history when needed.

        Returns:
            (question_to_process, resolved_question)
        """
        resolved_question = ""

        # Step 1: Handle conversation history if provided
        if isinstance(question, list) and len(question) > 1:
            # Extract conversation history and current question
            conversation_history = question[:-1]
            current_question = question[-1]

            logging.info(
                f"TABULAR FLOW: Processing question with "
                f"{len(conversation_history)} history messages"
            )

            # Retrieve cached previous formatted question for holistic context if available
            previous_formatted_question = self.previous_formatted_by_file.get(
                self.file_id, ""
            )
            if previous_formatted_question:
                logging.info(
                    f"TABULAR FLOW: Previous formatted question (cached): "
                    f"{previous_formatted_question[:100]}"
                )

            # Resolve contextual references in the current question (with holistic context)
            logging.info("TABULAR FLOW: Resolving question with history...")
            resolved_question = resolve_question_with_history(
                conversation_history,
                current_question,
                previous_formatted_question,
                self.previous_resolved_by_file.get(self.file_id, ""),
            )
            logging.info(f"TABULAR FLOW: Resolved question: {resolved_question[:100]}")

            # Use the resolved question for further processing
            question_to_process = resolved_question
        elif isinstance(question, list) and len(question) == 1:
            # Single item in list, extract it
            question_to_process = question[0]
        else:
            # String question, use as-is (backward compatibility)
            question_to_process = question

        return str(question_to_process), str(resolved_question)

    def _detect_user_language(self, question_text: str) -> str:
        """
        Detect the user's language from the raw question using lingua
        (independent of schema/DB context).
        """
        try:
            detected_lang_name = detect_lang(str(question_text))
            lang_name_lower = (detected_lang_name or "").lower()
            if "english" in lang_name_lower:
                user_language = "en"
            elif "german" in lang_name_lower:
                user_language = "de"
            else:
                user_language = "en"
            logging.info(
                f"TABULAR FLOW: Detected user language from question: "
                f"{detected_lang_name} -> code={user_language}"
            )
            return user_language
        except Exception as lang_err:
            logging.warning(
                f"TABULAR FLOW: Language detection failed, "
                f"defaulting to 'en': {str(lang_err)}"
            )
            return "en"

    def _process_agent_response(
        self,
        response: dict,
        question: str,
        query_type: str,
        language: str,
        resolved_question: str = "",
        formatted_question: str = "",
    ) -> dict:
        """
        Process the agent response and format it appropriately.

        Args:
            response: The agent response dictionary
            question: Original user question
            query_type: Type of query for formatting
            language: Language for response
            resolved_question: Question after context resolution
            formatted_question: Question after SQL formatting

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
                formatted_answer = (
                    "I apologize, but I cannot provide a response to this question. "
                    f"{str(e)}"
                )
        elif self.model_choice in [
            "claude-sonnet-4@20250514",
            "claude-sonnet-4-5",
        ]:
            formatted_answer = get_anthropic_non_rag_response(
                self.config, base_prompt, self.model_choice
            )
        else:
            formatted_answer = get_azure_non_rag_response(self.config, base_prompt)

        # Format intermediate steps for display
        logging.info("TABULAR FLOW: Formatting intermediate steps for display...")
        formatted_steps = (
            self._format_intermediate_steps(
                intermediate_steps,
                resolved_question=resolved_question,
                formatted_question=formatted_question,
                raw_agent_output=final_answer,
                processed_output=formatted_answer,
            )
            if intermediate_steps
            else None
        )

        logging.info("TABULAR FLOW: === Response processing completed ===")
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

    def _format_intermediate_steps(
        self,
        intermediate_steps: list,
        resolved_question: str = "",
        formatted_question: str = "",
        raw_agent_output: str = "",
        processed_output: str = "",
    ) -> str:
        """
        Format intermediate steps into human-readable text with question context.

        Args:
            intermediate_steps: List of tuples (action, observation) from agent
            resolved_question: Question after context resolution
            formatted_question: Question after SQL formatting
            raw_agent_output: Raw output from agent before processing
            processed_output: Final processed output after formatting

        Returns:
            str: Formatted text showing complete execution trace
        """
        if not intermediate_steps:
            return ""

        formatted_lines: List[str] = []

        # Add question processing section at the start
        formatted_lines.append("=" * 80)
        formatted_lines.append("QUESTION PROCESSING")
        formatted_lines.append("=" * 80)
        formatted_lines.append("")

        if resolved_question:
            formatted_lines.append("Resolved question:")
            formatted_lines.append(f"  {resolved_question}")
            formatted_lines.append("")

        if formatted_question:
            formatted_lines.append("Formatted question:")
            formatted_lines.append(f"  {formatted_question}")
            formatted_lines.append("")

        # Add agent execution trace section
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
        formatted_lines.append("")

        # Add post-processing section if available
        if raw_agent_output or processed_output:
            formatted_lines.append("=" * 80)
            formatted_lines.append("POST-PROCESSING")
            formatted_lines.append("=" * 80)
            formatted_lines.append("")

            if raw_agent_output:
                formatted_lines.append("Raw agent output:")
                formatted_lines.append(f"  {raw_agent_output}")
                formatted_lines.append("")

            if processed_output:
                formatted_lines.append("Final formatted answer:")
                formatted_lines.append(f"  {processed_output}")
                formatted_lines.append("")

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
