"""
Auto Migration Service for Embedding Type Migration

This service provides automatic, transparent migration from legacy azure embeddings
to azure-3-large embeddings. It is designed to work seamlessly across all file
access points (upload, chat, status checks).

Key Features:
- Database-first embedding type lookup (single source of truth)
- Transparent auto-migration for single and multi-file scenarios
- Reuses existing infrastructure (GCSHandler, EmbeddingHandler, CleanupCoordinator)
- Handles migration during upload AND chat time
- Supports concurrent users with efficient resource usage

Architecture:
- Simple rule: If embedding_type == "azure" → migrate to "azure-3-large"
- Works for ANY file access (no special multi-file logic)
- Preserves all metadata (username, file_id, file_hash)
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from rtl_rag_chatbot_api.chatbot.embedding_handler import EmbeddingHandler
from rtl_rag_chatbot_api.chatbot.gcs_handler import GCSHandler
from rtl_rag_chatbot_api.common.cleanup_coordinator import CleanupCoordinator

logger = logging.getLogger(__name__)

# Constants for embedding types
LEGACY_EMBEDDING_TYPE = "azure"
NEW_EMBEDDING_TYPE = "azure-3-large"


class AutoMigrationService:
    """
    Centralized service for automatic embedding migration.

    This service handles the detection and migration of legacy embeddings
    to the new embedding format transparently during any file access.
    """

    def __init__(self, configs, gcs_handler: GCSHandler, session_local=None):
        """
        Initialize the AutoMigrationService.

        Args:
            configs: Application configuration object
            gcs_handler: GCS handler instance for cloud operations
            session_local: Database session factory (optional)
        """
        self.configs = configs
        self.gcs_handler = gcs_handler
        self.session_local = session_local
        self.use_file_hash_db = getattr(configs, "use_file_hash_db", False)

    def get_embedding_type_from_db(self, file_id: str) -> Optional[str]:
        """
        Get embedding type from database (single source of truth).

        Args:
            file_id: The file ID to check

        Returns:
            Embedding type string or None if not found
        """
        if not self.use_file_hash_db or not self.session_local:
            logger.debug(f"Database lookup disabled or unavailable for {file_id}")
            return None

        try:
            from rtl_rag_chatbot_api.common.db import get_file_info_by_file_id

            with self.session_local() as db_session:
                record = get_file_info_by_file_id(db_session, file_id)

                if record and record.embedding_type:
                    logger.info(
                        f"Database lookup for {file_id}: embedding_type = {record.embedding_type}"
                    )
                    return record.embedding_type
                else:
                    logger.debug(f"No database record found for {file_id}")
                    return None

        except Exception as e:
            logger.error(
                f"Error getting embedding type from database for {file_id}: {e}"
            )
            return None

    def check_needs_migration(self, file_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a file needs migration based on its embedding type.

        This method:
        1. Checks if file is tabular (CSV/Excel) - skips migration
        2. Checks database first (single source of truth)
        3. Falls back to GCS file_info.json if database unavailable
        4. Returns whether migration is needed

        Args:
            file_id: The file ID to check

        Returns:
            Tuple of (needs_migration: bool, current_embedding_type: str)
        """
        # Check if file is tabular (CSV/Excel) - these don't use embeddings
        try:
            file_info = self.gcs_handler.get_file_info(file_id)
            if file_info and file_info.get("is_tabular", False):
                logger.info(
                    f"File {file_id} is tabular (CSV/Excel) - skipping migration (no embeddings used)"
                )
                return False, "tabular"  # No migration needed for tabular files
        except Exception as e:
            logger.debug(f"Error checking if file is tabular: {e}")

        # First, try database lookup (highest priority)
        embedding_type = self.get_embedding_type_from_db(file_id)

        # Fallback to GCS if database doesn't have the info
        if not embedding_type:
            try:
                file_info = self.gcs_handler.get_file_info(file_id)
                if file_info:
                    embedding_type = file_info.get("embedding_type")
                    logger.info(
                        f"GCS lookup for {file_id}: embedding_type = {embedding_type}"
                    )
            except Exception as e:
                logger.warning(f"Error getting file info from GCS for {file_id}: {e}")

        # If still no embedding type found, assume new embedding (default)
        if not embedding_type:
            logger.info(
                f"No embedding type found for {file_id}, assuming {NEW_EMBEDDING_TYPE}"
            )
            return False, NEW_EMBEDDING_TYPE

        # Check if it's legacy embedding that needs migration
        needs_migration = embedding_type == LEGACY_EMBEDDING_TYPE

        if needs_migration:
            logger.info(
                f"File {file_id} has legacy embedding ({LEGACY_EMBEDDING_TYPE}) - migration needed"
            )
        else:
            logger.debug(f"File {file_id} has {embedding_type} - no migration needed")

        return needs_migration, embedding_type

    async def migrate_file(
        self,
        file_id: str,
        file_path: Optional[str] = None,
        embedding_handler: Optional[EmbeddingHandler] = None,
        background_tasks=None,
    ) -> Dict[str, Any]:
        """
        Migrate a single file from legacy to new embedding type.

        This method:
        1. Downloads the file if not provided locally
        2. Deletes old embeddings
        3. Creates new embeddings with azure-3-large
        4. Updates database with new embedding type
        5. Preserves all metadata (usernames, file_hash, etc.)

        Args:
            file_id: The file ID to migrate
            file_path: Optional path to local file (if already downloaded)
            embedding_handler: Optional EmbeddingHandler instance
            background_tasks: Optional FastAPI BackgroundTasks for async operations

        Returns:
            Dict with migration result
        """
        logger.info(f"Starting migration for file {file_id}")

        try:
            # Get file metadata from GCS
            file_info = self.gcs_handler.get_file_info(file_id)
            if not file_info:
                return {
                    "status": "error",
                    "message": f"File info not found for {file_id}",
                    "file_id": file_id,
                }

            # Extract metadata
            usernames = file_info.get("username", [])
            if not isinstance(usernames, list):
                usernames = [usernames] if usernames else []

            original_filename = file_info.get("original_filename", f"{file_id}.pdf")
            file_hash = file_info.get("file_hash")

            logger.info(
                f"Migration metadata for {file_id}: "
                f"usernames={usernames}, filename={original_filename}, hash={file_hash}"
            )

            # Step 1: Ensure file is available locally
            # Check if file_path was provided and exists
            if file_path and os.path.exists(file_path):
                logger.info(f"Using provided file for migration: {file_path}")
            else:
                # Try to find file in local_data directory first
                local_file_path = f"local_data/{file_id}_{original_filename}"
                if os.path.exists(local_file_path):
                    logger.info(
                        f"Found file in local_data for migration: {local_file_path}"
                    )
                    file_path = local_file_path
                else:
                    # Download from GCS as last resort
                    logger.info(f"Downloading file {file_id} from GCS for migration")
                    file_path = await self._download_file_for_migration(
                        file_id, original_filename
                    )

                    if not file_path or not os.path.exists(file_path):
                        return {
                            "status": "error",
                            "message": f"Failed to download file {file_id} for migration",
                            "file_id": file_id,
                        }

            # Step 2: Delete old embeddings (both local and GCS)
            logger.info(f"Deleting old embeddings for {file_id}")
            delete_result = await self._delete_old_embeddings(file_id)

            if not delete_result.get("success"):
                logger.error(
                    f"Failed to delete old embeddings: {delete_result.get('error')}"
                )
                # Continue anyway - we'll overwrite

            # Step 3: Create new embeddings with azure-3-large
            logger.info(
                f"Creating new embeddings for {file_id} with {NEW_EMBEDDING_TYPE}"
            )

            # Create or use provided embedding handler
            if not embedding_handler:
                embedding_handler = EmbeddingHandler(self.configs, self.gcs_handler)

            # Import the parallel embedding creator to reuse existing logic
            from rtl_rag_chatbot_api.chatbot.parallel_embedding_creator import (
                create_embeddings_parallel,
            )

            # Create new embeddings using the existing parallel creator
            embedding_results = await create_embeddings_parallel(
                file_ids=[file_id],
                file_paths=[file_path],
                embedding_handler=embedding_handler,
                configs=self.configs,
                session_local=self.session_local,
                background_tasks=background_tasks,
                username_lists=[usernames],
                file_metadata_list=[file_info],
                max_concurrent_tasks=1,
            )

            # Check if embedding creation was successful
            if embedding_results and len(embedding_results) > 0:
                result = embedding_results[0]
                if result.get("status") == "error":
                    return {
                        "status": "error",
                        "message": f"Failed to create new embeddings: {result.get('message')}",
                        "file_id": file_id,
                    }

            # Step 4: Update database with new embedding type
            logger.info(f"Updating database with new embedding type for {file_id}")
            update_result = await self._update_embedding_type_in_db(
                file_id, NEW_EMBEDDING_TYPE, file_hash
            )

            if not update_result.get("success"):
                logger.warning(
                    f"Failed to update database for {file_id}: {update_result.get('error')}"
                )
                # Don't fail the migration - embeddings are created

            logger.info(f"Successfully migrated {file_id} to {NEW_EMBEDDING_TYPE}")

            return {
                "status": "success",
                "message": f"Successfully migrated from {LEGACY_EMBEDDING_TYPE} to {NEW_EMBEDDING_TYPE}",
                "file_id": file_id,
                "old_embedding_type": LEGACY_EMBEDDING_TYPE,
                "new_embedding_type": NEW_EMBEDDING_TYPE,
                "usernames": usernames,
            }

        except Exception as e:
            logger.error(f"Error migrating file {file_id}: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Migration failed: {str(e)}",
                "file_id": file_id,
            }

    async def _download_file_for_migration(
        self, file_id: str, original_filename: str
    ) -> Optional[str]:
        """
        Download and decrypt a file from GCS for migration.

        Args:
            file_id: The file ID to download
            original_filename: Original filename for local storage

        Returns:
            Path to downloaded file or None if failed
        """
        try:
            # Create temp directory if it doesn't exist
            temp_dir = "temp_files"
            os.makedirs(temp_dir, exist_ok=True)

            # Use basename for safety
            safe_filename = os.path.basename(original_filename)
            destination_path = os.path.join(temp_dir, f"{file_id}_{safe_filename}")

            # Download and decrypt using GCS handler
            decrypted_path = self.gcs_handler.download_encrypted_file_by_id(
                file_id, destination_path=destination_path
            )

            if decrypted_path and os.path.exists(decrypted_path):
                logger.info(f"Downloaded file for migration: {decrypted_path}")
                return decrypted_path
            else:
                logger.error(f"Failed to download file {file_id}")
                return None

        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {str(e)}")
            return None

    async def _delete_old_embeddings(self, file_id: str) -> Dict[str, Any]:
        """
        Delete old embeddings for a file using CleanupCoordinator.

        Args:
            file_id: The file ID whose embeddings to delete

        Returns:
            Dict with deletion result
        """
        try:
            cleanup_coordinator = CleanupCoordinator(
                self.configs, self.session_local, self.gcs_handler
            )

            # Delete both local and GCS embeddings
            cleanup_coordinator.cleanup_chroma_instance(file_id, include_gcs=True)

            logger.info(f"Successfully deleted old embeddings for {file_id}")
            return {"success": True}

        except Exception as e:
            logger.error(f"Error deleting old embeddings for {file_id}: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _update_embedding_type_in_db(
        self, file_id: str, new_embedding_type: str, file_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update the embedding type in the database after successful migration.

        Args:
            file_id: The file ID to update
            new_embedding_type: The new embedding type
            file_hash: Optional file hash for verification

        Returns:
            Dict with update result
        """
        if not self.use_file_hash_db or not self.session_local:
            logger.debug("Database updates disabled or unavailable")
            return {"success": True, "message": "Database disabled"}

        try:
            from rtl_rag_chatbot_api.common.db import (
                get_file_info_by_file_id,
                update_file_info_embedding_type,
            )

            with self.session_local() as db_session:
                # Check if record exists
                record = get_file_info_by_file_id(db_session, file_id)

                if record:
                    # Update existing record
                    result = update_file_info_embedding_type(
                        db_session, file_id, new_embedding_type
                    )
                    logger.info(f"Updated database record for {file_id}: {result}")
                    return {"success": True, "result": result}
                else:
                    logger.warning(f"No database record found for {file_id} to update")
                    return {
                        "success": False,
                        "error": "No database record found",
                    }

        except Exception as e:
            logger.error(f"Error updating database for {file_id}: {str(e)}")
            return {"success": False, "error": str(e)}

    async def check_and_migrate_if_needed(
        self,
        file_id: str,
        file_path: Optional[str] = None,
        embedding_handler: Optional[EmbeddingHandler] = None,
        background_tasks=None,
        embedding_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Check if a file needs migration and migrate if necessary.

        This is the main entry point for transparent auto-migration.
        Use this method at any file access point (upload, chat, etc.)

        Args:
            file_id: The file ID to check and migrate
            file_path: Optional path to local file
            embedding_handler: Optional EmbeddingHandler instance
            background_tasks: Optional FastAPI BackgroundTasks

        Returns:
            Dict with result:
            {
                "needs_migration": bool,
                "migrated": bool,
                "embedding_type": str,
                "migration_result": dict (if migrated)
            }
        """
        # Check if migration is needed (use provided embedding_type if available)
        if embedding_type is not None:
            needs_migration = embedding_type == "azure"
            current_embedding_type = embedding_type
        else:
            needs_migration, current_embedding_type = self.check_needs_migration(
                file_id
            )

        if not needs_migration:
            return {
                "needs_migration": False,
                "migrated": False,
                "embedding_type": current_embedding_type,
                "message": f"File {file_id} already uses {current_embedding_type}",
            }

        # Perform migration
        logger.info(
            f"Auto-migrating {file_id} from {current_embedding_type} to {NEW_EMBEDDING_TYPE}"
        )

        migration_result = await self.migrate_file(
            file_id=file_id,
            file_path=file_path,
            embedding_handler=embedding_handler,
            background_tasks=background_tasks,
        )

        return {
            "needs_migration": True,
            "migrated": (migration_result.get("status") == "success"),
            "embedding_type": (
                NEW_EMBEDDING_TYPE
                if migration_result.get("status") == "success"
                else current_embedding_type
            ),
            "migration_result": migration_result,
        }

    async def check_and_migrate_multiple_files(
        self,
        file_ids: List[str],
        file_paths: Optional[Dict[str, str]] = None,
        embedding_handler: Optional[EmbeddingHandler] = None,
        background_tasks=None,
        max_concurrent: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Check and migrate multiple files concurrently.

        This method processes multiple files in parallel with concurrency control.

        Args:
            file_ids: List of file IDs to check and migrate
            file_paths: Optional dict mapping file_id to local file path
            embedding_handler: Optional EmbeddingHandler instance
            background_tasks: Optional FastAPI BackgroundTasks
            max_concurrent: Maximum concurrent migrations (default: 3)

        Returns:
            List of migration results for each file
        """
        if not file_paths:
            file_paths = {}

        async def migrate_single(file_id: str) -> Dict[str, Any]:
            return await self.check_and_migrate_if_needed(
                file_id=file_id,
                file_path=file_paths.get(file_id),
                embedding_handler=embedding_handler,
                background_tasks=background_tasks,
            )

        # Use semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def migrate_with_limit(file_id: str) -> Dict[str, Any]:
            async with semaphore:
                return await migrate_single(file_id)

        # Process all files concurrently with limit
        tasks = [migrate_with_limit(file_id) for file_id in file_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "needs_migration": False,
                        "migrated": False,
                        "embedding_type": "unknown",
                        "error": str(result),
                        "file_id": file_ids[i],
                    }
                )
            else:
                processed_results.append(result)

        return processed_results
