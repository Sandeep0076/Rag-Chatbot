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
from typing import Any, Dict, List, Optional, Tuple

# Import existing handlers and utilities
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
        result = gcs_handler.find_existing_file_by_hash(file_hash, use_db=True)

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


async def migrate_single_file_embedding(file_id: str, configs: dict) -> MigrationResult:
    """
    Migrate a single file from legacy to new embedding type
    This is a dummy function for now - implement actual migration logic later

    Args:
        file_id: File ID to migrate
        configs: Configuration dictionary

    Returns:
        MigrationResult object
    """
    try:
        logger.info(
            f"Starting migration for file {file_id} from {LEGACY_EMBEDDING_TYPE} to {NEW_EMBEDDING_TYPE}"
        )

        # TODO: Implement actual migration logic:
        # 1. Load existing embeddings
        # 2. Delete old embeddings
        # 3. Create new embeddings with azure-03-small
        # 4. Update database records

        # Dummy implementation for now
        await asyncio.sleep(0.1)  # Simulate processing time

        logger.info(f"Migration completed successfully for file {file_id}")
        return MigrationResult(
            file_id=file_id, success=True, embedding_type=NEW_EMBEDDING_TYPE
        )

    except Exception as e:
        logger.error(f"Migration failed for file {file_id}: {str(e)}")
        return MigrationResult(file_id=file_id, success=False, error=str(e))


# async def migrate_files_in_parallel(file_ids: List[str], configs: dict) -> List[MigrationResult]:
#     """
#     Migrate multiple files in parallel

#     Args:
#         file_ids: List of file IDs to migrate
#         configs: Configuration dictionary

#     Returns:
#         List of MigrationResult objects
#     """
#     if not file_ids:
#         return []

#     logger.info(f"Starting parallel migration for {len(file_ids)} files")

#     # Create migration tasks
#     tasks = [migrate_single_file_embedding(file_id, configs) for file_id in file_ids]

#     # Execute migrations in parallel
#     results = await asyncio.gather(*tasks, return_exceptions=True)

#     # Process results and handle exceptions
#     migration_results = []
#     for i, result in enumerate(results):
#         if isinstance(result, Exception):
#             migration_results.append(MigrationResult(
#                 file_id=file_ids[i],
#                 success=False,
#                 error=str(result)
#             ))
#         else:
#             migration_results.append(result)

#     # Log summary
#     successful = sum(1 for r in migration_results if r.success)
#     failed = len(migration_results) - successful
#     logger.info(f"Migration completed: {successful} successful, {failed} failed")

#     return migration_results


# async def process_upload_migration_logic_for_existing_files
# (existing_file_ids: List[str], configs: dict) -> Dict[str, Any]:
#     """
#     Main function that implements the flowchart logic for handling existing file IDs in uploads
#     This should be called at the very beginning of the upload endpoint for existing files

#     Args:
#         existing_file_ids: List of existing file IDs from the upload
#         configs: Configuration dictionary

#     Returns:
#         Dictionary containing:
#         {
#             'needs_processing': bool,  # Whether files need further processing
#             'migration_results': List[MigrationResult],  # Results of any migrations
#             'file_classification': Dict,  # Classification of files
#             'recommendations': List[str]  # Recommendations for next steps
#         }
#     """
#     result = {
#         'needs_processing': True,
#         'migration_results': [],
#         'file_classification': {},
#         'recommendations': []
#     }

#     # If no existing files, proceed with normal upload
#     if not existing_file_ids:
#         result['recommendations'].append("All new files - proceed with normal upload")
#         return result

#     logger.info(f"Processing migration logic for {len(existing_file_ids)} existing files")

#     # For existing file IDs, check embedding types in database
#     # This is simplified - in practice, download and check actual files
#     legacy_files = []
#     new_embedding_files = []

#     try:
#         with get_db_session() as db:
#             for file_id in existing_file_ids:
#                 # For existing files, get embedding type from database
#                 # This is a placeholder - implement proper embedding type lookup
#                 file_info = db.query(FileInfo).filter(
#                     FileInfo.file_id == file_id
#                 ).first()
#                 if file_info:
#                     embedding_type = getattr(file_info, 'embedding_type', NEW_EMBEDDING_TYPE)
#                     if embedding_type == LEGACY_EMBEDDING_TYPE:
#                         legacy_files.append(file_id)
#                     else:
#                         new_embedding_files.append(file_id)
#                 else:
#                     # File not found in database - treat as new
#                     logger.warning(f"File ID {file_id} not found in database")

#     except Exception as e:
#         logger.error(f"Error checking existing files: {str(e)}")
#         result['recommendations'].append(f"Error checking existing files: {str(e)}")
#         return result

#     # Classify files
#     classification = {
#         'new_files': [],
#         'legacy_files': legacy_files,
#         'new_embedding_files': new_embedding_files,
#         'missing_files': []
#     }
#     result['file_classification'] = classification

#     # Log current state
#     logger.info(
#         f"File classification: {len(legacy_files)} legacy, "
#         f"{len(new_embedding_files)} new embeddings"
#     )

#     # Decision logic based on flowchart
#     if len(legacy_files) > 0:
#         # Any legacy embeddings found - need migration
#         logger.info("Legacy embeddings detected - starting migration process")
#         result['recommendations'].append("Legacy embeddings found - migration required")

#         # Migrate legacy files
#         migration_results = await migrate_files_in_parallel(legacy_files, configs)
#         result['migration_results'] = migration_results

#         # Check if all migrations were successful
#         failed_migrations = [r for r in migration_results if not r.success]
#         if failed_migrations:
#             result['recommendations'].append(
#                 f"Migration failed for {len(failed_migrations)} files"
#             )
#             logger.warning(
#                 f"Migration failed for files: {[r.file_id for r in failed_migrations]}"
#             )
#         else:
#             result['recommendations'].append("All legacy files migrated successfully")

#     elif len(new_embedding_files) > 0:
#         # All existing files have new embeddings - no migration needed
#         logger.info("All existing files have new embeddings - no migration required")
#         result['recommendations'].append("All files have current embeddings - proceed normally")

#     return result


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

    # Separate existing files that don't need migration from new files
    existing_files_no_migration = new_embedding_files + missing_files

    # Decision logic based on flowchart
    if total_existing_files == 0:
        # All files are new - no migration needed
        return {
            "files_to_migrate": [],
            "existing_files_no_migration": existing_files_no_migration,
            "new_files": new_files,
            "migration_needed": False,
            "reason": "All files are new - no existing embeddings to migrate",
            "file_counts": file_counts,
            "file_classification": classification,
        }

    elif legacy_count > 0 and new_embedding_count == 0 and new_files_count == 0:
        # ALL existing files are legacy AND no new files - consistent state, no migration needed
        return {
            "files_to_migrate": [],
            "existing_files_no_migration": existing_files_no_migration,
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
            "existing_files_no_migration": existing_files_no_migration,
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
            "existing_files_no_migration": existing_files_no_migration,
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
            "existing_files_no_migration": existing_files_no_migration,
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


async def log_detailed_migration_file_info(
    files_to_migrate: List[str],
    existing_files_no_migration: List[str],
    new_files: List[str],
    configs: dict,
) -> None:
    """
    Log detailed file information for all files (migration needed, existing no migration, and new files).

    Args:
        files_to_migrate: List of file IDs that need migration
        existing_files_no_migration: List of existing file IDs that don't need migration
        new_files: List of new file IDs that need to be processed
        configs: Configuration dictionary
    """
    gcs_handler = GCSHandler(configs)

    # Log files that need migration
    if files_to_migrate:
        logger.info("=== FILES THAT NEED MIGRATION ===")
        for file_id in files_to_migrate:
            file_info = gcs_handler.get_file_info(file_id)
            if file_info:
                file_id_from_info = file_info.get("file_id", "N/A")
                usernames = file_info.get("username", [])
                if not isinstance(usernames, list):
                    usernames = [usernames] if usernames else []
                logger.info(
                    f"File ID: {file_id_from_info}, "
                    f"Usernames: {usernames}, "
                    f"Embedding Type: {file_info.get('embedding_type', 'N/A')}"
                )
            else:
                logger.warning(f"No file info found for file ID: {file_id}")

    # Log existing files that don't need migration
    if existing_files_no_migration:
        logger.info("=== EXISTING FILES THAT DON'T NEED MIGRATION ===")
        for file_id in existing_files_no_migration:
            file_info = gcs_handler.get_file_info(file_id)
            if file_info:
                file_id_from_info = file_info.get("file_id", "N/A")
                usernames = file_info.get("username", [])
                if not isinstance(usernames, list):
                    usernames = [usernames] if usernames else []
                logger.info(
                    f"File ID: {file_id_from_info}, "
                    f"Usernames: {usernames}, "
                    f"Embedding Type: {file_info.get('embedding_type', 'N/A')}"
                )
            else:
                logger.warning(f"No file info found for file ID: {file_id}")

    # Log new files that need to be processed
    if new_files:
        logger.info("=== NEW FILES THAT NEED TO BE PROCESSED ===")
        for file_id in new_files:
            logger.info(
                f"New File ID: {file_id} - Will be processed with azure-03-small embeddings"
            )

    # Log summary
    total_files = (
        len(files_to_migrate) + len(existing_files_no_migration) + len(new_files)
    )
    logger.info(
        f"=== SUMMARY: {len(files_to_migrate)} files need migration, "
        f"{len(existing_files_no_migration)} existing files don't need migration, "
        f"{len(new_files)} new files to process (Total: {total_files}) ==="
    )
