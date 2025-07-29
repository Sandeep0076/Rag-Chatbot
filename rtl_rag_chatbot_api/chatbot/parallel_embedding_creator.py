"""
Parallel Embedding Creator for RAG PDF Chatbot API

This module provides functionality for creating embeddings in parallel
for multiple documents.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks

from rtl_rag_chatbot_api.chatbot.embedding_handler import EmbeddingHandler


async def create_embeddings_parallel(
    file_ids: List[str],
    file_paths: List[str],
    embedding_handler: EmbeddingHandler,
    configs: Dict[str, Any],
    session_local: Any,
    background_tasks: BackgroundTasks = None,
    username_lists: Optional[List[List[str]]] = None,
    file_metadata_list: Optional[List[Dict[str, Any]]] = None,
    max_concurrent_tasks: int = 4,
) -> List[Dict[str, Any]]:
    """
    Create embeddings for multiple documents in parallel.

    Args:
        file_ids: List of file IDs to create embeddings for
        file_paths: List of paths to the temporary files
        embedding_handler: Handler for creating embeddings
        configs: Application configuration
        session_local: Database session factory
        username_lists: Optional list of username lists for each file
        file_metadata_list: Optional list of file metadata for each file
        max_concurrent_tasks: Maximum number of concurrent embedding creation tasks

    Returns:
        List of results from embedding creation tasks
    """
    # Prepare tasks for embedding creation
    tasks = []
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    # Make sure we have default values for optional parameters
    if not username_lists:
        username_lists = [None] * len(file_ids)
    if not file_metadata_list:
        file_metadata_list = [None] * len(file_ids)

    async def create_embeddings_with_semaphore(
        file_id: str,
        file_path: str,
        username_list: List[str] = None,
        file_metadata: Dict[str, Any] = None,
    ):
        async with semaphore:
            from rtl_rag_chatbot_api.app import create_embeddings_background

            try:
                result = await create_embeddings_background(
                    file_id,
                    file_path,
                    embedding_handler,
                    configs,
                    session_local,
                    username_list,
                    file_metadata,
                    background_tasks,
                )

                # Check if create_embeddings_background returned an error result
                if result.get("status") == "error":
                    error_msg = result.get("message", "Unknown error")
                    logging.error(
                        f"Embedding creation failed for file {file_id}: {error_msg}"
                    )
                    return {"file_id": file_id, "status": "error", "error": error_msg}

                return {"file_id": file_id, "status": "success", "result": result}
            except Exception as e:
                logging.error(f"Error creating embeddings for file {file_id}: {str(e)}")
                return {"file_id": file_id, "status": "error", "error": str(e)}

    # Create tasks for all files
    for i, file_id in enumerate(file_ids):
        tasks.append(
            create_embeddings_with_semaphore(
                file_id,
                file_paths[i],
                username_lists[i] if i < len(username_lists) else None,
                file_metadata_list[i] if i < len(file_metadata_list) else None,
            )
        )

    # Execute all tasks concurrently with max_concurrent_tasks limit
    logging.info(
        f"Creating embeddings for {len(file_ids)} files in parallel with max {max_concurrent_tasks} concurrent tasks"
    )
    results = await asyncio.gather(*tasks)

    logging.info(f"Completed parallel embedding creation for {len(file_ids)} files")
    return results
