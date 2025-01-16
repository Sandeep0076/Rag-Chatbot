"""
Utility module for managing test resources and cleanup.
"""

import json
import os
from pathlib import Path
from typing import List, Optional


class TestResourceManager:
    """Manages test resources including file IDs for cleanup."""

    def __init__(self, storage_file: Optional[str] = None):
        """Initialize the TestResourceManager.

        Args:
            storage_file: Optional path to the storage file. If not provided,
                         defaults to 'test_file_ids.json' in the tests directory.
        """
        if storage_file is None:
            storage_file = os.path.join(os.path.dirname(__file__), "test_file_ids.json")
        self.storage_file = Path(storage_file)
        self._ensure_storage_file()

    def _ensure_storage_file(self) -> None:
        """Ensure the storage file exists with valid JSON."""
        if not self.storage_file.exists():
            self.storage_file.write_text('{"file_ids": []}')

    def _read_storage(self) -> List[str]:
        """Read file IDs from storage."""
        try:
            data = json.loads(self.storage_file.read_text())
            return data.get("file_ids", [])
        except json.JSONDecodeError:
            return []

    def _write_storage(self, file_ids: List[str]) -> None:
        """Write file IDs to storage."""
        data = {"file_ids": file_ids}
        self.storage_file.write_text(json.dumps(data, indent=2))

    def store_file_id(self, file_id: str) -> None:
        """Store a file ID for later cleanup.

        Args:
            file_id: The file ID to store.
        """
        file_ids = self._read_storage()
        if file_id not in file_ids:
            file_ids.append(file_id)
            self._write_storage(file_ids)

    def get_all_file_ids(self) -> List[str]:
        """Get all stored file IDs.

        Returns:
            List of stored file IDs.
        """
        return self._read_storage()

    def clear_file_ids(self) -> None:
        """Clear all stored file IDs."""
        self._write_storage([])
