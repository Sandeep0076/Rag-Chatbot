"""
Migration Handler for Legacy Embeddings

This module handles the migration of legacy azure embeddings to azure-03-small
embeddings based on the flowchart logic. It provides functionality to:
1. Check file embedding types (database first, then GCS fallback)
2. Classify files as legacy or new embedding types
3. Migrate legacy embeddings to new format
4. Handle mixed file scenarios in uploads
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

# Import existing handlers and utilities
from fastapi import UploadFile

from rtl_rag_chatbot_api.chatbot.gcs_handler import GCSHandler

logger = logging.getLogger(__name__)

# Constants for embedding types
LEGACY_EMBEDDING_TYPE = "azure"
NEW_EMBEDDING_TYPE = "azure-03-small"


class MigrationResult:
    """Result class for migration operations"""

    def __init__(
        self, file_id: str, success: bool, error: str = None, embedding_type: str = None
    ):
        self.file_id = file_id
        self.success = success
        self.error = error
        self.embedding_type = embedding_type


class FileEmbeddingInfo:
    """Information about a file's embedding status"""

    def __init__(
        self,
        file_id: str,
        exists: bool,
        embedding_type: str = None,
        needs_migration: bool = False,
    ):
        self.file_id = file_id
        self.exists = exists
        self.embedding_type = embedding_type
        self.needs_migration = needs_migration


async def check_file_id_and_embedding_type(
    file_content: bytes, configs: dict
) -> Tuple[Optional[str], Optional[str]]:
    """
    Calculate file hash and check if file exists using find_existing_file_by_hash.

    Args:
        file_content: The file content bytes
        configs: Configuration dictionary

    Returns:
        Tuple of (file_id, embedding_type) or (None, None) if not found
    """
    import hashlib

    # Calculate file hash from content
    file_hash = hashlib.md5(file_content).hexdigest()
    logger.info(f"Calculated file hash: {file_hash}")

    try:
        # Use GCSHandler's find_existing_file_by_hash method
        gcs_handler = GCSHandler(configs)
        # Only use database if it's enabled
        use_db = getattr(configs, "use_file_hash_db", False)
        result = gcs_handler.find_existing_file_by_hash(file_hash, use_db=use_db)

        if result:
            file_id, embedding_type = result
            logger.info(
                f"Found file with hash {file_hash} with embedding type: {embedding_type}"
            )
            return file_id, embedding_type
        else:
            logger.info(f"File with hash {file_hash} not found")
            return None, None

    except Exception as e:
        logger.warning(f"Error checking file hash {file_hash}: {str(e)}")
        return None, None


async def check_files_embedding_status(
    file_contents: List[Tuple[str, bytes]], configs: dict
) -> List[FileEmbeddingInfo]:
    """
    Check embedding status for multiple files in parallel

    Args:
        file_contents: List of tuples (file_id, file_content_bytes) to check
        configs: Configuration dictionary

    Returns:
        List of FileEmbeddingInfo objects
    """

    async def check_single_file(file_data: Tuple[str, bytes]) -> FileEmbeddingInfo:
        input_file_id, file_content = file_data
        try:
            found_file_id, embedding_type = await check_file_id_and_embedding_type(
                file_content, configs
            )

            if found_file_id is None:
                # File doesn't exist - use input_file_id for new files
                return FileEmbeddingInfo(input_file_id, exists=False)

            # File exists - use the found_file_id (actual file ID from database/GCS)
            needs_migration = embedding_type == LEGACY_EMBEDDING_TYPE
            return FileEmbeddingInfo(
                file_id=found_file_id,
                exists=True,
                embedding_type=embedding_type,
                needs_migration=needs_migration,
            )

        except Exception as e:
            logger.error(f"Error checking file {input_file_id}: {str(e)}")
            return FileEmbeddingInfo(input_file_id, exists=False)

    # Process files in parallel
    tasks = [check_single_file(file_data) for file_data in file_contents]
    return await asyncio.gather(*tasks)


def classify_files_by_embedding_type(
    file_infos: List[FileEmbeddingInfo],
) -> Dict[str, List[str]]:
    """
    Classify files based on their embedding types

    Args:
        file_infos: List of FileEmbeddingInfo objects

    Returns:
        Dictionary with classification:
        {
            'new_files': [file_ids],  # Files that don't exist
            'legacy_files': [file_ids],  # Files with legacy embeddings
            'new_embedding_files': [file_ids],  # Files with new embeddings
            'missing_files': [file_ids]  # Files that couldn't be found/checked
        }
    """
    classification = {
        "new_files": [],
        "legacy_files": [],
        "new_embedding_files": [],
        "missing_files": [],
    }

    for file_info in file_infos:
        if not file_info.exists:
            classification["new_files"].append(file_info.file_id)
        elif file_info.embedding_type == LEGACY_EMBEDDING_TYPE:
            classification["legacy_files"].append(file_info.file_id)
        elif file_info.embedding_type == NEW_EMBEDDING_TYPE:
            classification["new_embedding_files"].append(file_info.file_id)
        else:
            classification["missing_files"].append(file_info.file_id)

    return classification


async def migrate_single_file_embedding(
    file_id: str, usernames: List[str], configs: dict, file: Optional[UploadFile] = None
) -> MigrationResult:
    """
    Migrate a single file from legacy to new embedding type
    This is a dummy function for now - implement actual migration logic later

    Args:
        file_id: File ID to migrate
        usernames: List of usernames associated with the file to migrate
        configs: Configuration dictionary
        file: Optional UploadFile object for new file uploads (similar to upload function)

    Returns:
        MigrationResult object
    """
    try:
        logger.info(
            f"Starting migration for file {file_id} for users {usernames} "
            f"from {LEGACY_EMBEDDING_TYPE} to {NEW_EMBEDDING_TYPE}"
        )

        # Handle file parameter if provided
        if file:
            logger.info(f"File provided for migration: {file.filename}")
            # TODO: Use the file content for migration logic
            # For now, just log that we have the file
            try:
                content = await file.read()
                logger.info(f"Read {len(content)} bytes from file {file.filename}")
                # Reset file pointer for potential future use
                await file.seek(0)
            except Exception as e:
                logger.error(f"Error reading file {file.filename}: {str(e)}")
                return MigrationResult(
                    file_id=file_id,
                    success=False,
                    error=f"Error reading file: {str(e)}",
                )

        # TODO: Implement actual migration logic:
        # 1. Use usernames list to scope the migration if needed
        # 2. Delete old embeddings for all users
        # 3. Create new embeddings with azure-03-small
        # 4. Update database records

        # Dummy implementation for now
        await asyncio.sleep(0.1)  # Simulate processing time

        logger.info(
            f"Migration completed successfully for file {file_id} for users {usernames}"
        )
        return MigrationResult(
            file_id=file_id, success=True, embedding_type=NEW_EMBEDDING_TYPE
        )

    except Exception as e:
        logger.error(
            f"Migration failed for file {file_id} for users {usernames}: {str(e)}"
        )
        return MigrationResult(file_id=file_id, success=False, error=str(e))


def decide_migration_files(file_infos: List[FileEmbeddingInfo]) -> Dict[str, Any]:
    """
    Single decision-making function that determines which files need migration.
    Implements the correct flowchart logic:
    - If ALL files are legacy → NO migration (consistent state)
    - If ALL files are new embeddings → NO migration (consistent state)
    - If ALL files are new → NO migration (no existing embeddings)
    - If MIX of legacy + new embeddings → Migrate legacy files
    - If MIX of legacy + new files → Migrate legacy files (NEW FILES WILL CREATE azure-03-small!)
    - If MIX of new embeddings + new files → NO migration

    Args:
        file_infos: List of FileEmbeddingInfo objects

    Returns:
        Dictionary with:
        {
            'files_to_migrate': List[str],              # File IDs that need migration
            'existing_files_no_migration': List[str],    # Existing file IDs that don't need migration
            'new_files': List[str],                     # New file IDs that need to be processed
            'migration_needed': bool,                   # Whether any migration is needed
            'reason': str,                              # Explanation of the decision
            'file_counts': Dict,                       # Breakdown of file types
            'file_classification': Dict                # Complete classification of all files
        }
    """
    # Classify files by type
    classification = classify_files_by_embedding_type(file_infos)

    legacy_files = classification["legacy_files"]
    new_embedding_files = classification["new_embedding_files"]
    new_files = classification["new_files"]
    missing_files = classification["missing_files"]

    # Count each type
    legacy_count = len(legacy_files)
    new_embedding_count = len(new_embedding_files)
    new_files_count = len(new_files)
    total_existing_files = legacy_count + new_embedding_count

    file_counts = {
        "legacy": legacy_count,
        "new_embeddings": new_embedding_count,
        "new_files": new_files_count,
        "missing": len(missing_files),
        "total_existing": total_existing_files,
    }

    # Decision logic based on flowchart
    if total_existing_files == 0:
        # All files are new - no migration needed
        return {
            "files_to_migrate": [],
            "existing_files_no_migration": [],  # No existing files
            "new_files": new_files,
            "migration_needed": False,
            "reason": "All files are new - no existing embeddings to migrate",
            "file_counts": file_counts,
            "file_classification": classification,
        }

    elif legacy_count > 0 and new_embedding_count == 0 and new_files_count == 0:
        # ALL existing files are legacy AND no new files - consistent state, no migration needed
        # BUT legacy files still need to be processed (downloaded)
        return {
            "files_to_migrate": [],
            # Include legacy files for processing
            "existing_files_no_migration": legacy_files + missing_files,
            "new_files": new_files,
            "migration_needed": False,
            "reason": "All existing files have legacy embeddings - consistent state, no migration needed",
            "file_counts": file_counts,
            "file_classification": classification,
        }

    elif legacy_count == 0 and new_embedding_count > 0:
        # ALL existing files have new embeddings - consistent state (regardless of new files)
        return {
            "files_to_migrate": [],
            # Include new embedding files for processing
            "existing_files_no_migration": new_embedding_files + missing_files,
            "new_files": new_files,
            "migration_needed": False,
            "reason": "All existing files have new embeddings - consistent state, no migration needed",
            "file_counts": file_counts,
            "file_classification": classification,
        }

    elif legacy_count > 0 and (new_embedding_count > 0 or new_files_count > 0):
        # MIX: legacy files + (new embeddings OR new files) - migrate legacy files
        # New files will create azure-03-small embeddings, creating mixed state
        reason_parts = []
        if new_embedding_count > 0:
            reason_parts.append(f"{new_embedding_count} existing new-embedding files")
        if new_files_count > 0:
            reason_parts.append(
                f"{new_files_count} new files (will create azure-03-small)"
            )

        additional_info = " + ".join(reason_parts)
        reason = f"Mixed embedding types: {legacy_count} legacy + {additional_info} - migrating legacy"

        return {
            "files_to_migrate": legacy_files,
            "existing_files_no_migration": new_embedding_files
            + missing_files,  # Non-legacy files for processing
            "new_files": new_files,
            "migration_needed": True,
            "reason": reason,
            "file_counts": file_counts,
            "file_classification": classification,
        }

    else:
        # Edge case - shouldn't happen but handle gracefully
        return {
            "files_to_migrate": [],
            # All existing files for processing
            "existing_files_no_migration": legacy_files
            + new_embedding_files
            + missing_files,
            "new_files": new_files,
            "migration_needed": False,
            "reason": "No clear migration decision could be made",
            "file_counts": file_counts,
            "file_classification": classification,
        }


async def check_existing_file_ids_embedding_status(
    file_ids: List[str], configs: dict
) -> List[FileEmbeddingInfo]:
    """
    Check embedding status for existing file IDs (no hash calculation needed).

    Args:
        file_ids: List of existing file IDs
        configs: Configuration dictionary

    Returns:
        List of FileEmbeddingInfo objects
    """

    async def check_single_existing_file_id(file_id: str) -> FileEmbeddingInfo:
        try:
            gcs_handler = GCSHandler(configs)
            file_info = gcs_handler.get_file_info(file_id)

            if not file_info:
                return FileEmbeddingInfo(file_id, exists=False)

            # Get embedding type from file info (default to new if not specified)
            embedding_type = file_info.get("embedding_type", NEW_EMBEDDING_TYPE)
            needs_migration = embedding_type == LEGACY_EMBEDDING_TYPE

            return FileEmbeddingInfo(
                file_id=file_id,
                exists=True,
                embedding_type=embedding_type,
                needs_migration=needs_migration,
            )

        except Exception as e:
            logger.error(f"Error checking existing file {file_id}: {str(e)}")
            return FileEmbeddingInfo(file_id, exists=False)

    # Process file IDs in parallel
    tasks = [check_single_existing_file_id(file_id) for file_id in file_ids]
    return await asyncio.gather(*tasks)


async def decide_migration_for_mixed_upload(
    file_contents: List[Tuple[str, bytes]] = None,
    existing_file_ids: List[str] = None,
    configs: dict = None,
) -> Dict[str, Any]:
    """
    Single comprehensive migration decision function for both file content and existing file IDs.

    Args:
        file_contents: List of tuples (temp_file_id, file_content_bytes) for new uploads
        existing_file_ids: List of existing file IDs
        configs: Configuration dictionary

    Returns:
        Dictionary with migration decision:
        {
            'files_to_migrate': List[str],  # File IDs that need migration
            'migration_needed': bool,       # Whether any migration is needed
            'reason': str,                  # Explanation of the decision
            'file_breakdown': Dict         # Detailed breakdown by source and type
        }
    """
    all_file_infos = []

    # Process new file uploads (calculate hash, check embedding type)
    if file_contents:
        new_file_infos = await check_files_embedding_status(file_contents, configs)
        all_file_infos.extend(new_file_infos)

    # Process existing file IDs (directly get embedding type)
    if existing_file_ids:
        existing_file_infos = await check_existing_file_ids_embedding_status(
            existing_file_ids, configs
        )
        all_file_infos.extend(existing_file_infos)

    # Apply the corrected migration decision logic
    return decide_migration_files(all_file_infos)


async def get_detailed_migration_file_info(
    files_to_migrate: List[str],
    existing_files_no_migration: List[str],
    new_files: List[str],
    configs: dict,
    existing_file_ids: List[str] = None,
) -> Dict[str, Any]:
    """
    Get detailed file information for all files (migration needed, existing no migration, and new files).

    Args:
        files_to_migrate: List of file IDs that need migration
        existing_files_no_migration: List of existing file IDs that don't need migration
        new_files: List of new file IDs that need to be processed
        configs: Configuration dictionary
        existing_file_ids: List of file IDs that were passed as existing_file_ids parameter

    Returns:
        Dictionary containing detailed file information:
        {
            'files_to_migrate': List[Dict],  # Detailed info for files needing migration
            'existing_files_no_migration': List[Dict],  # Detailed info for existing files not needing migration
            'new_files': List[Dict],  # Detailed info for new files
            'summary': Dict  # Summary statistics
        }
    """
    gcs_handler = GCSHandler(configs)
    existing_ids_set = set(existing_file_ids or [])

    result = {
        "files_to_migrate": [],
        "existing_files_no_migration": [],
        "new_files": [],
        "summary": {},
    }

    # Get detailed info for files that need migration
    if files_to_migrate:
        logger.info("=== FILES THAT NEED MIGRATION ===")
        for file_id in files_to_migrate:
            file_info = gcs_handler.get_file_info(file_id)
            if file_info:
                file_id_from_info = file_info.get("file_id", "N/A")
                usernames = file_info.get("username", [])
                if not isinstance(usernames, list):
                    usernames = [usernames] if usernames else []
                original_filename = file_info.get("original_filename", "N/A")

                # Determine if this file came from existing_file_ids or uploaded files
                source = (
                    "existing_file_ids"
                    if file_id in existing_ids_set
                    else "uploaded_file"
                )

                detailed_info = {
                    "file_id": file_id_from_info,
                    "usernames": usernames,
                    "embedding_type": file_info.get("embedding_type", "N/A"),
                    "original_filename": original_filename,
                    "source": source,
                }
                result["files_to_migrate"].append(detailed_info)

                logger.info(
                    f"File ID: {file_id_from_info}, "
                    f"Usernames: {usernames}, "
                    f"Embedding Type: {file_info.get('embedding_type', 'N/A')}, "
                    f"Source: {source}"
                )
            else:
                logger.warning(f"No file info found for file ID: {file_id}")
                source = (
                    "existing_file_ids"
                    if file_id in existing_ids_set
                    else "uploaded_file"
                )
                result["files_to_migrate"].append(
                    {
                        "file_id": file_id,
                        "usernames": [],
                        "embedding_type": "N/A",
                        "original_filename": "N/A",
                        "source": source,
                    }
                )

    # Get detailed info for existing files that don't need migration
    if existing_files_no_migration:
        logger.info("=== EXISTING FILES THAT DON'T NEED MIGRATION ===")
        for file_id in existing_files_no_migration:
            file_info = gcs_handler.get_file_info(file_id)
            if file_info:
                file_id_from_info = file_info.get("file_id", "N/A")
                usernames = file_info.get("username", [])
                if not isinstance(usernames, list):
                    usernames = [usernames] if usernames else []
                original_filename = file_info.get("original_filename", "N/A")

                # Determine if this file came from existing_file_ids or uploaded files
                source = (
                    "existing_file_ids"
                    if file_id in existing_ids_set
                    else "uploaded_file"
                )

                detailed_info = {
                    "file_id": file_id_from_info,
                    "usernames": usernames,
                    "embedding_type": file_info.get("embedding_type", "N/A"),
                    "original_filename": original_filename,
                    "source": source,
                }
                result["existing_files_no_migration"].append(detailed_info)

                logger.info(
                    f"File ID: {file_id_from_info}, "
                    f"Usernames: {usernames}, "
                    f"Embedding Type: {file_info.get('embedding_type', 'N/A')}, "
                    f"Source: {source}"
                )
            else:
                logger.warning(f"No file info found for file ID: {file_id}")
                source = (
                    "existing_file_ids"
                    if file_id in existing_ids_set
                    else "uploaded_file"
                )
                result["existing_files_no_migration"].append(
                    {
                        "file_id": file_id,
                        "usernames": [],
                        "embedding_type": "N/A",
                        "original_filename": "N/A",
                        "source": source,
                    }
                )

    # Get detailed info for new files that need to be processed
    if new_files:
        logger.info("=== NEW FILES THAT NEED TO BE PROCESSED ===")
        for file_id in new_files:
            detailed_info = {
                "file_id": file_id,
                "usernames": [],
                "embedding_type": "azure-03-small",
                "original_filename": file_id.replace("temp_", ""),
                "source": "uploaded_file",  # New files are always from uploads
            }
            result["new_files"].append(detailed_info)

            logger.info(
                f"New File ID: {file_id} - Will be processed with azure-03-small embeddings"
            )

    # Create summary
    total_files = (
        len(files_to_migrate) + len(existing_files_no_migration) + len(new_files)
    )
    result["summary"] = {
        "files_to_migrate_count": len(files_to_migrate),
        "existing_files_no_migration_count": len(existing_files_no_migration),
        "new_files_count": len(new_files),
        "total_files": total_files,
    }

    logger.info(
        f"=== SUMMARY: {len(files_to_migrate)} files need migration, "
        f"{len(existing_files_no_migration)} existing files don't need migration, "
        f"{len(new_files)} new files to process (Total: {total_files}) ==="
    )

    return result


def _is_multi_file_scenario(
    all_files: List, parsed_existing_file_ids: List[str]
) -> bool:
    """Check if this is a multi-file scenario."""
    return (
        len(all_files) > 1
        or len(parsed_existing_file_ids) > 1  # Multiple new files
        or (  # Existing file IDs
            len(all_files) == 1 and len(parsed_existing_file_ids) > 0
        )  # One new file + existing file IDs
    )


async def _prepare_file_contents_for_migration(all_files: List) -> Tuple[List, Dict]:
    """Prepare file contents for migration check and return file objects map."""
    file_contents_for_migration = []
    file_objects_map = {}

    if len(all_files) > 0:
        for uploaded_file in all_files:
            try:
                content = await uploaded_file.read()
                await uploaded_file.seek(0)
                temp_id = f"temp_{uploaded_file.filename}"
                file_contents_for_migration.append((temp_id, content))
                file_objects_map[uploaded_file.filename] = uploaded_file
                logger.info(f"Added file {uploaded_file.filename} for migration check")
            except Exception as e:
                logger.error(f"Error reading file {uploaded_file.filename}: {str(e)}")

    return file_contents_for_migration, file_objects_map


async def _download_migration_files(files_to_migrate: List[str], configs: dict) -> None:
    """Download and decrypt files that need migration."""
    if not files_to_migrate:
        return

    logger.info(
        "Starting migration for existing files - downloading and decrypting files from GCS"
    )

    for file_id in files_to_migrate:
        try:
            logger.info(f"Processing migration for file_id: {file_id}")
            gcs_handler = GCSHandler(configs)
            decrypted_file_path = gcs_handler.download_encrypted_file_by_id(file_id)

            if decrypted_file_path and os.path.exists(decrypted_file_path):
                logger.info(
                    f"Successfully downloaded and decrypted file for {file_id} to {decrypted_file_path}"
                )
            else:
                logger.error(f"Failed to download or decrypt file for {file_id}")

        except Exception as e:
            logger.error(f"Error processing migration for file {file_id}: {str(e)}")
            continue


async def _delete_old_embeddings(files_to_migrate: List[Dict], configs: dict) -> None:
    """Delete old embeddings for files that need migration."""
    if not files_to_migrate:
        return

    for file_info in files_to_migrate:
        file_id = file_info["file_id"]
        usernames = file_info["usernames"]
        original_filename = file_info.get("original_filename", "")

        logger.info(f"Detailed migration file info: {file_info}")
        logger.info(f"File ID: {file_id}")
        logger.info(f"Usernames: {usernames}")
        logger.info(f"Original filename: {original_filename}")

        try:
            logger.info(f"Deleting old embeddings for file_id: {file_id}")

            from rtl_rag_chatbot_api.common.cleanup_coordinator import (
                CleanupCoordinator,
            )

            gcs_handler = GCSHandler(configs)

            if hasattr(configs, "use_file_hash_db") and configs.use_file_hash_db:
                try:
                    from rtl_rag_chatbot_api.app import SessionLocal

                    cleanup_coordinator = CleanupCoordinator(
                        configs, SessionLocal, gcs_handler
                    )
                except ImportError:
                    logger.warning(
                        "SessionLocal not available, proceeding with database disabled"
                    )
                    cleanup_coordinator = CleanupCoordinator(configs, None, gcs_handler)
            else:
                cleanup_coordinator = CleanupCoordinator(configs, None, gcs_handler)

            cleanup_coordinator.cleanup_chroma_instance(file_id, include_gcs=True)
            logger.info(f"Successfully deleted old embeddings for file_id: {file_id}")

        except Exception as e:
            logger.error(
                f"Error deleting old embeddings for file_id {file_id}: {str(e)}"
            )
            continue


async def handle_migration_for_upload(
    all_files: List,
    parsed_existing_file_ids: List[str],
    configs: dict,
) -> Tuple[bool, Optional[Dict], Optional[Dict]]:
    """
    Handle migration logic for upload scenarios.

    This function encapsulates all the migration decision logic that was previously
    in the upload function. It handles:
    1. Multi-file scenario detection
    2. File content preparation for migration check
    3. Migration decision making
    4. Detailed info gathering
    5. Migration execution or blocking

    Args:
        all_files: List of uploaded files
        parsed_existing_file_ids: List of existing file IDs
        configs: Configuration dictionary

    Returns:
        Tuple of (is_multi_file_scenario, migration_result, detailed_info)
        - is_multi_file_scenario: Boolean indicating if this is a multi-file scenario
        - migration_result: Dict with migration decision results or None
        - detailed_info: Dict with detailed file information or None
    """
    if not _is_multi_file_scenario(all_files, parsed_existing_file_ids):
        return False, None, None

    logger.info("=== MULTI-FILE SCENARIO DETECTED - RUNNING MIGRATION CHECK ===")
    logger.info(
        f"New files: {len(all_files)}, Existing file IDs: {len(parsed_existing_file_ids)}"
    )

    (
        file_contents_for_migration,
        file_objects_map,
    ) = await _prepare_file_contents_for_migration(all_files)

    try:
        migration_decision = await decide_migration_for_mixed_upload(
            file_contents=file_contents_for_migration,
            existing_file_ids=parsed_existing_file_ids,
            configs=configs,
        )

        logger.info("=== MIGRATION DECISION RESULTS ===")
        logger.info(f"Migration needed: {migration_decision['migration_needed']}")
        logger.info(f"File breakdown: {migration_decision['file_counts']}")

        detailed_info = await get_detailed_migration_file_info(
            migration_decision["files_to_migrate"],
            migration_decision["existing_files_no_migration"],
            migration_decision["new_files"],
            configs,
            parsed_existing_file_ids,
        )

        await _download_migration_files(migration_decision["files_to_migrate"], configs)
        await _delete_old_embeddings(detailed_info["files_to_migrate"], configs)

        return (
            True,
            {
                "blocked": False,
                "message": "Migration decision check completed - DEBUG MODE",
                "migration_decision": migration_decision,
                "debug_info": {
                    "is_multi_file_scenario": True,
                    "new_files_count": len(file_contents_for_migration),
                    "existing_file_ids_count": len(parsed_existing_file_ids),
                    "total_files_checked": len(file_contents_for_migration)
                    + len(parsed_existing_file_ids),
                },
            },
            detailed_info,
        )

    except Exception as e:
        logger.error(f"Error in migration decision logic: {str(e)}")
        logger.info(
            "Continuing with normal file processing due to migration check error"
        )
        return True, None, None
