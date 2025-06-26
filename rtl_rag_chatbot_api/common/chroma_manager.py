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
    This version uses thread-local storage to ensure thread safety for ChromaDB clients,
    which is critical during parallel operations like embedding creation.
    """

    _instance = None
    _lock = threading.Lock()  # Lock for singleton creation

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ChromaDBManager, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initializes the manager with thread-local storage."""
        self._thread_local = threading.local()
        # A global lock to serialize client creation, preventing race conditions
        # within ChromaDB's internals during concurrent instantiation.
        self._creation_lock = threading.Lock()

    def _get_thread_instances(self) -> Dict[str, dict]:
        """Gets the dictionary of instances for the current thread."""
        if not hasattr(self._thread_local, "instances"):
            self._thread_local.instances = {}
        return self._thread_local.instances

    def get_instance(
        self,
        file_id: str,
        embedding_type: str,
        user_id: str = None,
        is_embedding: bool = False,
    ):
        """
        Get or create a ChromaDB instance. Instances are now thread-local to ensure
        safety during parallel processing.
        """
        base_path = f"./chroma_db/{file_id}/{embedding_type}"

        if is_embedding:
            instance_key = f"{file_id}/{embedding_type}"
        else:
            instance_key = (
                f"{file_id}/{embedding_type}/user_{user_id}"
                if user_id
                else f"{file_id}/{embedding_type}"
            )

        thread_instances = self._get_thread_instances()

        if instance_key in thread_instances:
            instance_data = thread_instances[instance_key]
            instance_data["last_used"] = datetime.now()
            return instance_data["client"]

        with self._creation_lock:
            # Double-check in case another thread created the instance while this
            # thread was waiting for the lock.
            if instance_key in thread_instances:
                return thread_instances[instance_key]["client"]

            try:
                settings = Settings(
                    allow_reset=True, is_persistent=True, anonymized_telemetry=False
                )
                client = chromadb.PersistentClient(path=base_path, settings=settings)

                thread_instances[instance_key] = {
                    "client": client,
                    "last_used": datetime.now(),
                }
                logging.info(
                    f"Created new thread-local ChromaDB instance for {instance_key}"
                )
                return client
            except Exception as e:
                logging.error(
                    f"Error creating ChromaDB instance for {instance_key}: {str(e)}"
                )
                raise

    def cleanup(self):
        """Public cleanup method that can be called from outside the class."""
        logging.info("Cleaning up ChromaDB instances")
        try:
            self.cleanup_old_instances()
        except Exception as e:
            logging.error(f"Error in ChromaDB cleanup: {str(e)}")

    def cleanup_old_instances(self):
        """Clean up instances that haven't been used for a while and embedding instances."""
        current_time = datetime.now()

        with self._lock:  # Use _lock instead of _instance_lock
            thread_instances = self._get_thread_instances()
            for instance_key in list(thread_instances.keys()):
                instance_data = thread_instances[instance_key]
                # Clean up if it hasn't been used recently
                if current_time - instance_data["last_used"] > timedelta(
                    minutes=5
                ):  # Use a default threshold
                    try:
                        del thread_instances[instance_key]
                        logging.info(f"Cleaned up ChromaDB instance for {instance_key}")
                    except Exception as e:
                        logging.error(
                            f"Error cleaning up ChromaDB instance for {instance_key}: {str(e)}"
                        )

    def get_collection(
        self,
        file_id: str,
        embedding_type: str,
        collection_name: str,
        user_id: str = None,
        is_embedding: bool = False,
    ):
        """Get or create a collection using the thread-local client."""
        client = self.get_instance(file_id, embedding_type, user_id, is_embedding)
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"file_id": file_id, "embedding_type": embedding_type},
        )
        return collection
