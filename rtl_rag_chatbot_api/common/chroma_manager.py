# chroma_manager.py

import logging
import threading
from datetime import datetime, timedelta
from typing import Dict

import chromadb
from chromadb.config import Settings


class ChromaDBManager:
    """
    Singleton manager for ChromaDB instances to ensure consistent settings and reuse.
    Thread-safe implementation for concurrent access.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ChromaDBManager, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self._instances: Dict[
            str, dict
        ] = {}  # {file_id: {'client': client, 'last_used': timestamp}}
        self._settings = Settings(
            allow_reset=True, is_persistent=True, anonymized_telemetry=False
        )
        self._instance_lock = threading.Lock()
        self._cleanup_threshold = timedelta(hours=4)  # Configurable cleanup threshold

    def get_instance(
        self, file_id: str, embedding_type: str, user_id: str = None
    ) -> chromadb.PersistentClient:
        """Get or create a ChromaDB instance with optional user-specific connection."""
        # Always use the base path for persistent storage
        base_path = f"./chroma_db/{file_id}/{embedding_type}"

        # Use different instance keys for managing connections
        if user_id is None:
            instance_key = f"{file_id}/{embedding_type}"
        else:
            # For chat sessions, track user instance but use same base path
            instance_key = f"{file_id}/{embedding_type}/user_{user_id}"

        with self._instance_lock:
            if instance_key in self._instances:
                instance_data = self._instances[instance_key]
                instance_data["last_used"] = datetime.now()
                return instance_data["client"]

            try:
                # Always create client with base path to access shared embeddings
                client = chromadb.PersistentClient(
                    path=base_path, settings=self._settings
                )

                self._instances[instance_key] = {
                    "client": client,
                    "last_used": datetime.now(),
                }

                logging.info(f"Created new ChromaDB instance for {instance_key}")
                return client
            except Exception as e:
                logging.error(
                    f"Error creating ChromaDB instance for {instance_key}: {str(e)}"
                )
                raise

    def get_collection(
        self,
        file_id: str,
        embedding_type: str,
        collection_name: str,
        user_id: str = None,
    ):
        """Get or create a collection with optional user tracking."""
        client = self.get_instance(file_id, embedding_type, user_id)

        # Always use the base collection for querying
        collection = client.get_or_create_collection(
            name=collection_name, metadata={"file_id": file_id}
        )

        return collection

    def cleanup_old_instances(self):
        """Clean up instances that haven't been used for a while."""
        current_time = datetime.now()

        with self._instance_lock:
            for instance_key in list(self._instances.keys()):
                instance_data = self._instances[instance_key]
                if current_time - instance_data["last_used"] > self._cleanup_threshold:
                    try:
                        del self._instances[instance_key]
                        logging.info(f"Cleaned up ChromaDB instance for {instance_key}")
                    except Exception as e:
                        logging.error(
                            f"Error cleaning up ChromaDB instance for {instance_key}: {str(e)}"
                        )
