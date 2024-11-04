import logging
import os
import shutil
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy.orm import Session

from configs.app_config import Config
from rtl_rag_chatbot_api.common.chroma_manager import ChromaDBManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CleanupCoordinator:
    """
    Coordinates cleanup operations for ChromaDB instances and associated resources.

    This class manages the cleanup of stale ChromaDB instances, local files, and database
    resources based on configurable time thresholds. It implements a comprehensive cleanup
    strategy that includes memory management, file system cleanup, and database maintenance.

    Attributes:
        config (Config): Configuration object containing cleanup settings:
            - staleness_threshold_minutes (int): Time before resource is considered stale
            - min_cleanup_interval (int): Minimum time between cleanup runs
            - cleanup_interval_minutes (int): Scheduled cleanup interval
        session_factory (Optional[Session]): SQLAlchemy session factory for database operations
        chroma_manager (ChromaDBManager): Manager for ChromaDB instances
        last_cleanup (datetime): Timestamp of last cleanup operation
        cleanup_folders (List[str]): List of folders to clean ['chroma_db', 'local_data', 'processed_data']

    Methods:
        cleanup(): Main cleanup method that coordinates all cleanup operations
        _should_cleanup() -> bool: Checks if cleanup should run based on time intervals
        _cleanup_chroma_instance(file_id: str): Cleans up specific ChromaDB instance
        _get_stale_file_ids() -> List[str]: Identifies stale file resources
        _cleanup_folder(folder: str): Cleans up specific folder while preserving active resources
        _is_stale_instance(file_id: str) -> bool: Checks if specific instance is stale

    """

    def __init__(self, config: Config, session_factory: Optional[Session] = None):
        self.config = config
        self.session_factory = session_factory
        self.chroma_manager = ChromaDBManager()
        self.last_cleanup = datetime.now()
        # Use config values instead of hardcoded values
        self.cleanup_threshold = timedelta(
            minutes=self.config.cleanup.staleness_threshold_minutes
        )
        self.min_cleanup_interval = timedelta(
            minutes=self.config.cleanup.min_cleanup_interval
        )
        self.cleanup_folders = ["chroma_db", "local_data", "processed_data"]

    def _should_cleanup(self) -> bool:
        """Check if enough time has passed since last cleanup."""
        time_since_cleanup = datetime.now() - self.last_cleanup
        return time_since_cleanup >= self.min_cleanup_interval

    def _cleanup_chroma_instance(self, file_id: str) -> None:
        """Cleanup a specific ChromaDB instance and its files."""
        try:
            # Cleanup from ChromaDB manager
            self.chroma_manager.cleanup_instance(file_id)

            # Remove local ChromaDB files
            chroma_path = f"./chroma_db/{file_id}"
            if os.path.exists(chroma_path):
                shutil.rmtree(chroma_path)
            logger.info(f"Cleaned up ChromaDB instance and files for {file_id}")
        except Exception as e:
            logger.error(f"Error cleaning up ChromaDB instance {file_id}: {str(e)}")

    def _get_stale_file_ids(self) -> List[str]:
        """Get list of file IDs that haven't been accessed in threshold period."""
        stale_files = []
        try:
            base_dir = "./chroma_db"
            if os.path.exists(base_dir):
                for file_id in os.listdir(base_dir):
                    file_path = os.path.join(base_dir, file_id)
                    if os.path.isdir(file_path):
                        # Check last access time
                        last_access = datetime.fromtimestamp(
                            os.path.getatime(file_path)
                        )
                        if datetime.now() - last_access >= self.cleanup_threshold:
                            stale_files.append(file_id)
        except Exception as e:
            logger.error(f"Error getting stale file IDs: {str(e)}")
        return stale_files

    def _cleanup_folder(self, folder: str) -> None:
        """Clean up a specific folder while preserving important files."""
        try:
            folder_path = os.path.join(os.getcwd(), folder)
            if os.path.exists(folder_path):
                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)
                    # Skip if the item is a currently active ChromaDB instance
                    if folder == "chroma_db" and not self._is_stale_instance(item):
                        continue
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                logger.info(f"Cleaned up {folder}")
        except Exception as e:
            logger.error(f"Error cleaning up {folder}: {str(e)}")

    def _is_stale_instance(self, file_id: str) -> bool:
        """Check if a ChromaDB instance is stale."""
        try:
            instance_path = f"./chroma_db/{file_id}"
            if not os.path.exists(instance_path):
                return False
            last_access = datetime.fromtimestamp(os.path.getatime(instance_path))
            return datetime.now() - last_access >= self.cleanup_threshold
        except Exception:
            return False

    def cleanup(self) -> None:
        """Main cleanup method that coordinates all cleanup operations."""
        if not self._should_cleanup():
            logger.info("Skipping cleanup - minimum interval not reached")
            return

        try:
            logger.info("Starting coordinated cleanup")

            # Get stale file IDs
            stale_files = self._get_stale_file_ids()

            # Cleanup stale ChromaDB instances and their files
            for file_id in stale_files:
                self._cleanup_chroma_instance(file_id)

            # Clean up other folders
            for folder in self.cleanup_folders:
                self._cleanup_folder(folder)

            self.last_cleanup = datetime.now()
            logger.info("Coordinated cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error in coordinated cleanup: {str(e)}")
