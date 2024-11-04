# Deprecated file
import logging
import os
from datetime import datetime, timedelta

import httpx
from sqlalchemy.orm import Session

from rtl_rag_chatbot_api.common.chroma_manager import ChromaDBManager
from rtl_rag_chatbot_api.common.db import Conversation, get_conversations_by_file_ids

# TODO centralize logging instance, e.g. in __init__
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# base directory for chromadb files
BASE_DIR = "./chroma_db"
DELETE_ENDPOINT = "http://localhost:8080/chroma/delete"
TIME_THRESHOLD = timedelta(hours=4)
DEV = os.getenv("DEV")


def is_stale_conversation(conversation: Conversation) -> bool:
    """
    Check if a conversation is stale by comparing its last updated time
    with the current time.
    """
    age = datetime.now() - conversation.updatedAt
    if age >= TIME_THRESHOLD:
        return True


def cleanup_chromadb_instances():
    """Cleanup old ChromaDB instances."""
    try:
        chroma_manager = ChromaDBManager()
        chroma_manager.cleanup_old_instances()
        logging.info("ChromaDB instance cleanup completed successfully")
    except Exception as e:
        logging.error(f"Error during ChromaDB instance cleanup: {str(e)}")


def offload_chromadb_embeddings(session_factory: Session):
    """Combined function for cleaning up conversations and ChromaDB instances."""

    log.info("Running scheduled job `offload_chromadb_embeddings`")

    if not session_factory:
        log.info(
            "No db connection, session_factory is None. Could not offload embeddings. Are you running in DEV?"
        )
        return

    # note: manually getting a session as we can't use `Depends` in background tasks
    with session_factory() as db_session:
        if not os.path.exists(BASE_DIR):
            log.warning(
                f"Cannot check chromadb folder, folder does not exist: {BASE_DIR}"
            )
            return

        file_ids = os.listdir(BASE_DIR)

        conversations = get_conversations_by_file_ids(
            session=db_session, file_ids=file_ids
        )

        if len(conversations) == 0:
            log.info(
                "No stale conversations found related to file chat. Checking later."
            )
            return

        # filter stale conversations
        log.info(
            f"Checking {len(conversations)} conversations for stale in-memory embeddings."
        )
        conversations = list(
            map(
                lambda c: c.fileId,
                filter(
                    lambda c: is_stale_conversation(conversation=c),
                    conversations,
                ),
            )
        )

        try:
            log.info(
                f"Found {len(conversations)} conversations in stale mode. "
                f"About to trigger delete for: {','.join(conversations)}"
            )
            for conversation in conversations:
                # make a DELETE request to the /chroma/delete endpoint
                response = httpx.request(
                    url=f"{DELETE_ENDPOINT}",
                    method="DELETE",
                    json={"file_id": conversation},
                )
                if response.status_code == 200:
                    log.info(f"Successfully triggered delete for {conversation}")
                else:
                    log.error(
                        f"Failed to delete {conversation}: {response.status_code}, {response.text}"
                    )
                cleanup_chromadb_instances()
        except Exception as e:
            log.error(f"Error while calling delete endpoint for {conversation}: {e}")
