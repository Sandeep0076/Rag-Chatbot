import hashlib
import json
import logging
import os
import sqlite3
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class UnifiedDatabaseBuilder:
    """
    Builds a unified SQLite database from multiple individual tabular_data.db files.

    This class implements the unified database strategy for multi-file tabular chat,
    where all tables from multiple files are merged into a single database with:
    - Renamed tables to avoid conflicts: {filename}_{tablename}
    - Source tracking columns: _source_file_id, _source_filename
    - Session metadata for cleanup and reference
    """

    def __init__(self, base_dir: str = "./chroma_db"):
        """
        Initialize the UnifiedDatabaseBuilder.

        Args:
            base_dir: Base directory for database storage
        """
        self.base_dir = base_dir
        self.db_name = "tabular_data.db"

    def build_unified_database(
        self,
        file_ids: List[str],
        all_file_infos: Dict[str, Dict],
    ) -> Dict[str, any]:
        """
        Build a unified SQLite database from multiple individual databases.

        Args:
            file_ids: List of file IDs to merge
            all_file_infos: Dictionary mapping file_id to file metadata

        Returns:
            Dict containing:
                - unified_session_id: Unique identifier for this unified session
                - unified_db_path: Path to the unified database
                - session_dir: Directory containing the unified database
                - source_mapping: Mapping of unified tables to source files
                - metadata: Session metadata

        Raises:
            ValueError: If less than 2 file_ids provided
            FileNotFoundError: If any individual database file doesn't exist
        """
        if not file_ids or len(file_ids) < 2:
            raise ValueError("Unified database requires at least two file_ids")

        # Sort file_ids for consistent hashing
        sorted_file_ids = sorted(file_ids)

        # Generate unique session identifier
        file_ids_str = ",".join(sorted_file_ids)
        session_hash = hashlib.md5(file_ids_str.encode("utf-8")).hexdigest()[:8]
        timestamp_ms = int(time.time() * 1000)
        session_id = f"{timestamp_ms}_{session_hash}"
        unified_session_id = f"unified_session_{session_id}"

        # Create session directory
        session_dir_name = f"unified_{session_id}"
        session_dir = os.path.join(self.base_dir, session_dir_name)
        os.makedirs(session_dir, exist_ok=True)
        unified_db_path = os.path.join(session_dir, "unified_tabular.db")

        # Build the unified database
        source_mapping = self._merge_databases(
            sorted_file_ids,
            all_file_infos,
            unified_db_path,
        )

        # Persist metadata
        self._save_session_metadata(
            session_dir,
            sorted_file_ids,
            timestamp_ms,
            unified_db_path,
            source_mapping,
        )

        return {
            "unified_session_id": unified_session_id,
            "unified_db_path": unified_db_path,
            "session_dir": session_dir,
            "source_mapping": source_mapping,
            "metadata": {
                "file_ids": sorted_file_ids,
                "created_at": timestamp_ms,
                "table_count": len(source_mapping),
            },
        }

    def _merge_databases(
        self,
        file_ids: List[str],
        all_file_infos: Dict[str, Dict],
        unified_db_path: str,
    ) -> Dict[str, Dict]:
        """
        Merge all individual databases into a unified database.

        Args:
            file_ids: Sorted list of file IDs
            all_file_infos: File metadata dictionary
            unified_db_path: Path to the unified database

        Returns:
            Dictionary mapping unified table names to source information
        """
        conn = sqlite3.connect(unified_db_path, timeout=30)
        source_mapping = {}

        try:
            for idx, file_id in enumerate(file_ids):
                # Get individual database path
                db_path = os.path.join(self.base_dir, file_id, self.db_name)

                if not os.path.exists(db_path):
                    error_msg = (
                        f"Database not found for file_id: {file_id} at {db_path}"
                    )
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)

                alias = f"db_{idx}"
                conn.execute(f'ATTACH DATABASE ? AS "{alias}"', (db_path,))

                # Discover all user tables
                cursor = conn.execute(
                    f"""
                    SELECT name
                    FROM "{alias}".sqlite_master
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                    """
                )
                tables = [row[0] for row in cursor.fetchall()]

                # Get file metadata
                file_info = all_file_infos.get(file_id, {})
                original_filename = file_info.get("original_filename", f"{file_id}.db")

                # Create safe prefix from filename
                base_name = os.path.splitext(os.path.basename(original_filename))[0]
                safe_prefix = base_name.replace(" ", "_").replace("-", "_")

                # Copy each table with source tracking
                for table_name in tables:
                    unified_table = self._create_unified_table_name(
                        safe_prefix, table_name, source_mapping
                    )

                    # Copy table with source tracking columns
                    conn.execute(
                        f"""
                        CREATE TABLE "{unified_table}" AS
                        SELECT
                            *,
                            ? AS "_source_file_id",
                            ? AS "_source_filename"
                        FROM "{alias}"."{table_name}"
                        """,
                        (file_id, original_filename),
                    )

                    source_mapping[unified_table] = {
                        "file_id": file_id,
                        "filename": original_filename,
                        "original_table": table_name,
                    }

                conn.execute(f'DETACH DATABASE "{alias}"')

            conn.commit()
        finally:
            conn.close()

        return source_mapping

    def _create_unified_table_name(
        self,
        safe_prefix: str,
        table_name: str,
        existing_mapping: Dict[str, Dict],
    ) -> str:
        """
        Create a unique table name for the unified database.

        Args:
            safe_prefix: Sanitized filename prefix (not used, kept for compatibility)
            table_name: Original table name (already contains sanitized filename + _table suffix)
            existing_mapping: Current source mapping to check for conflicts

        Returns:
            Unique table name
        """
        # Table names from individual databases already include the sanitized filename
        # and are unique per file. Since we have _source_file_id and _source_filename
        # columns for tracking, we can use the table_name directly.
        # This avoids duplication like: filename_filename_table
        unified_table = table_name

        # Handle name collisions (in case multiple files somehow have identical table names)
        suffix = 1
        original_unified = unified_table
        while unified_table in existing_mapping:
            unified_table = f"{original_unified}_{suffix}"
            suffix += 1

        return unified_table

    def _save_session_metadata(
        self,
        session_dir: str,
        file_ids: List[str],
        timestamp_ms: int,
        unified_db_path: str,
        source_mapping: Dict[str, Dict],
    ) -> None:
        """
        Save session metadata files alongside the unified database.

        Args:
            session_dir: Session directory path
            file_ids: List of file IDs
            timestamp_ms: Creation timestamp
            unified_db_path: Path to unified database
            source_mapping: Table to source file mapping
        """
        try:
            # Save source mapping
            with open(
                os.path.join(session_dir, "source_mapping.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(source_mapping, f, indent=2, default=str)

            # Save session metadata
            session_metadata = {
                "file_ids": file_ids,
                "created_at": timestamp_ms,
                "unified_db_path": unified_db_path,
                "table_count": len(source_mapping),
            }
            with open(
                os.path.join(session_dir, "session_metadata.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(session_metadata, f, indent=2, default=str)

            logger.info(f"Saved session metadata to {session_dir}")
        except Exception as e:
            logger.warning(f"Failed to write session metadata: {str(e)}")

    def check_unified_database_exists(
        self, file_ids: List[str]
    ) -> Optional[Dict[str, any]]:
        """
        Check if a unified database already exists for the given file_ids.

        Args:
            file_ids: List of file IDs to check

        Returns:
            Dict with unified session info if found, None otherwise
        """
        if not file_ids or len(file_ids) < 2:
            return None

        # Generate expected session identifier
        sorted_file_ids = sorted(file_ids)
        file_ids_str = ",".join(sorted_file_ids)
        session_hash = hashlib.md5(file_ids_str.encode("utf-8")).hexdigest()[:8]

        # Search for matching session directories
        try:
            for entry in os.scandir(self.base_dir):
                if entry.is_dir() and entry.name.startswith("unified_"):
                    # Check if hash matches
                    if session_hash in entry.name:
                        session_dir = entry.path
                        metadata_path = os.path.join(
                            session_dir, "session_metadata.json"
                        )
                        unified_db_path = os.path.join(
                            session_dir, "unified_tabular.db"
                        )

                        # Verify metadata and database exist
                        if os.path.exists(metadata_path) and os.path.exists(
                            unified_db_path
                        ):
                            with open(metadata_path, "r", encoding="utf-8") as f:
                                metadata = json.load(f)

                            # Verify file_ids match exactly
                            if sorted(metadata.get("file_ids", [])) == sorted_file_ids:
                                return {
                                    "unified_session_id": f"unified_session_{entry.name.replace('unified_', '')}",
                                    "unified_db_path": unified_db_path,
                                    "session_dir": session_dir,
                                    "metadata": metadata,
                                }
        except Exception as e:
            logger.warning(f"Error checking for existing unified database: {str(e)}")

        return None
