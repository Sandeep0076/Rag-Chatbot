"""
ChromaDB Telemetry Fix

This module suppresses ChromaDB telemetry errors.
Import this before any ChromaDB imports in your application.
"""

import logging
import os


def disable_chromadb_telemetry():
    """
    Suppresses ChromaDB telemetry errors by setting environment variables
    and configuring the telemetry logger to ignore errors.
    """
    # Set environment variables to disable telemetry
    os.environ["CHROMA_TELEMETRY_IMPL"] = "none"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"

    # Suppress ChromaDB telemetry error logs
    # This prevents PostHog telemetry errors from appearing in logs
    telemetry_logger = logging.getLogger("chromadb.telemetry.product.posthog")
    telemetry_logger.setLevel(logging.CRITICAL)  # Only show CRITICAL, not ERROR


# Auto-execute when module is imported
disable_chromadb_telemetry()
