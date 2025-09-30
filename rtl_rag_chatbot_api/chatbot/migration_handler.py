"""
DEPRECATED: Legacy Migration Handler

This module has been DEPRECATED and replaced by AutoMigrationService.

The complex migration logic with flowchart decisions and multi-file scenarios
has been simplified to a single rule:
    - If embedding_type == "azure" → Auto-migrate to "azure-3-large"

New implementation:
    - Location: rtl_rag_chatbot_api/chatbot/auto_migration_service.py
    - Class: AutoMigrationService
    - Features:
        * Database-first embedding type lookup (single source of truth)
        * Transparent auto-migration during upload AND chat
        * Works for single and multi-file scenarios
        * No complex decision logic
        * Reuses existing infrastructure

Migration Points:
    1. File Upload: /file/upload endpoint
    2. Chat Access: /file/chat endpoint
    3. Status Check: /embeddings/status/{file_id} endpoint

DO NOT USE THIS MODULE FOR NEW CODE.
Use AutoMigrationService instead.

For backward compatibility, some helper classes and constants are retained below.
"""

import logging

logger = logging.getLogger(__name__)

# Constants for embedding types (kept for backward compatibility)
LEGACY_EMBEDDING_TYPE = "azure"
NEW_EMBEDDING_TYPE = "azure-3-large"


class MigrationResult:
    """
    DEPRECATED: Result class for migration operations.
    Use AutoMigrationService.migrate_file() which returns a dict instead.
    """

    def __init__(
        self, file_id: str, success: bool, error: str = None, embedding_type: str = None
    ):
        self.file_id = file_id
        self.success = success
        self.error = error
        self.embedding_type = embedding_type
        logger.warning(
            "MigrationResult class is deprecated. Use AutoMigrationService instead."
        )


class FileEmbeddingInfo:
    """
    DEPRECATED: Information about a file's embedding status.
    Use AutoMigrationService.check_needs_migration() instead.
    """

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
        logger.warning(
            "FileEmbeddingInfo class is deprecated. Use AutoMigrationService instead."
        )


# All other functions have been removed.
# Please use AutoMigrationService for all migration needs.
