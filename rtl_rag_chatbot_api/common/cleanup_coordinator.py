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

    def __init__(
        self,
        config: Config,
        session_factory: Optional[Session] = None,
        gcs_handler=None,
    ):
        self.config = config
        self.session_factory = session_factory
        self.chroma_manager = ChromaDBManager()
        self.gcs_handler = gcs_handler
        # self.gcs_handler = GCSHandler(config)
        self.last_cleanup = datetime.now()
        # Use config values instead of hardcoded values
        self.cleanup_threshold = timedelta(
            minutes=self.config.cleanup.staleness_threshold_minutes
        )
        self.min_cleanup_interval = timedelta(
            minutes=self.config.cleanup.min_cleanup_interval
        )
        self.cleanup_folders = ["chroma_db", "local_data", "processed_data"]

    def _should_cleanup(self, is_manual: bool = False) -> bool:
        """Check if enough time has passed since last cleanup."""
        if is_manual:
            return True  # Always allow manual cleanup
        time_since_cleanup = datetime.now() - self.last_cleanup
        return time_since_cleanup >= self.min_cleanup_interval

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

    def cleanup_chroma_instance(self, file_id: str, include_gcs: bool = False) -> None:
        """
        Cleanup ChromaDB instance and its files.
        Args:
            file_id: The ID of the file to cleanup
            include_gcs: Whether to also cleanup GCS storage
        """
        try:
            # Clean up local files
            chroma_path = f"./chroma_db/{file_id}"
            if os.path.exists(chroma_path):
                shutil.rmtree(chroma_path)
            logging.info(f"Cleaned up local ChromaDB instance and files for {file_id}")

            # Clean up GCS if requested
            if include_gcs:
                try:
                    self.gcs_handler.delete_embeddings(file_id)
                    logging.info(f"Cleaned up GCS embeddings for {file_id}")
                except Exception as e:
                    logging.error(
                        f"Error cleaning up GCS embeddings for {file_id}: {str(e)}"
                    )

        except Exception as e:
            logging.error(f"Error cleaning up ChromaDB instance {file_id}: {str(e)}")

    def _cleanup_folder(self, folder: str) -> None:
        """Clean up a specific local folder."""
        try:
            folder_path = os.path.join(os.getcwd(), folder)
            if os.path.exists(folder_path):
                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                logging.info(f"Cleaned up local folder: {folder}")
        except Exception as e:
            logging.error(f"Error cleaning up folder {folder}: {str(e)}")

    def cleanup(self, is_manual: bool = False) -> None:
        """
        Periodic cleanup method - cleans local resources.
        Args:
            is_manual (bool, optional): If True, bypasses time interval check. Defaults to False.
        """
        if not self._should_cleanup(is_manual):
            logging.info("Skipping cleanup - minimum interval not reached")
            return

        try:
            logging.info(f"Starting {'manual' if is_manual else 'scheduled'} cleanup")

            # Clean up local folders
            for folder in self.cleanup_folders:
                if folder == "chroma_db":
                    # For chroma_db, need to clean both files and memory instances
                    self.chroma_manager.cleanup_old_instances()
                else:
                    self._cleanup_folder(folder)

            self.last_cleanup = datetime.now()
            logging.info(
                f"{'Manual' if is_manual else 'Scheduled'} cleanup completed successfully"
            )
        except Exception as e:
            logging.error(f"Error in cleanup: {str(e)}")
