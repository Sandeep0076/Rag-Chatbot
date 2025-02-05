"""
Simple utility module for managing test resource IDs.
"""

import logging

logger = logging.getLogger(__name__)


class ResourceManager:
    """Manages test resource IDs for cleanup."""

    def __init__(self):
        """Initialize with empty list of file IDs."""
        self._file_ids = []
        logger.info("Initialized ResourceManager")

    def add_file_id(self, file_id: str):
        """Add a file ID if not already present."""
        if file_id not in self._file_ids:
            self._file_ids.append(file_id)
            logger.info(f"Added file ID: {file_id}")
            logger.info(f"Current file IDs: {self._file_ids}")

    def get_file_ids(self):
        """Get list of stored file IDs."""
        logger.info(f"Getting file IDs: {self._file_ids}")
        return self._file_ids

    def clear_file_ids(self):
        """Clear all stored file IDs."""
        logger.info("Clearing all file IDs")
        self._file_ids.clear()
