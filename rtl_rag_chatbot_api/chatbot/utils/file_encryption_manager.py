import asyncio
import logging
import os
from typing import Optional, Tuple

from rtl_rag_chatbot_api.chatbot.gcs_handler import GCSHandler
from rtl_rag_chatbot_api.chatbot.utils.encryption import encrypt_file


class FileEncryptionManager:
    """
    Manages file encryption operations for the RAG chatbot API.
    Handles encryption checking, encrypting, and uploading to GCS.
    """

    def __init__(self, gcs_handler: GCSHandler):
        """
        Initialize the FileEncryptionManager with a GCS handler.

        Args:
            gcs_handler: Google Cloud Storage handler for cloud operations.
        """
        self.gcs_handler = gcs_handler

    async def ensure_file_encryption(
        self,
        file_id: str,
        original_filename: str,
        temp_file_path: str,
        is_tabular: bool = False,
        is_database: bool = False,
    ) -> Optional[str]:
        """
        Ensures a file is encrypted and uploaded to GCS if needed.
        Checks if an encrypted version already exists, and if not,
        encrypts and uploads it.

        Args:
            file_id: The ID of the file
            original_filename: The original filename
            temp_file_path: Path to the temporary file
            is_tabular: Whether the file is tabular data
            is_database: Whether the file is a database

        Returns:
            Optional[str]: Path to the encrypted file if created, None if not needed or if operation failed
        """
        if is_tabular or is_database:
            # We don't encrypt tabular or database files using this method
            return None

        # Check if encrypted file exists in GCS
        encrypted_file_exists = await asyncio.to_thread(
            self.gcs_handler.check_file_exists,
            f"{self.gcs_handler.configs.gcp_resource.gcp_embeddings_folder}/{file_id}/{original_filename}.encrypted",
        )

        if not encrypted_file_exists:
            # Encrypt the file since it wasn't encrypted earlier
            encrypted_file_path = await self._encrypt_file(
                temp_file_path, original_filename
            )

            # Upload the encrypted file
            if encrypted_file_path:
                success = await self._upload_encrypted_file(
                    encrypted_file_path,
                    (
                        f"{self.gcs_handler.configs.gcp_resource.gcp_embeddings_folder}/"
                        f"{file_id}/{original_filename}.encrypted"
                    ),
                    original_filename,
                )
                if success:
                    return encrypted_file_path

        return None

    async def encrypt_and_upload(
        self, file_path: str, file_name: str, file_id: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Encrypts a file and uploads it to GCS.

        Args:
            file_path: Path to the file to encrypt
            file_name: Name of the file
            file_id: ID of the file

        Returns:
            Tuple[bool, Optional[str]]: (success status, encrypted file path if created)
        """
        encrypted_file_path = await self._encrypt_file(file_path, file_name)
        if encrypted_file_path:
            success = await self._upload_encrypted_file(
                encrypted_file_path,
                f"{self.gcs_handler.configs.gcp_resource.gcp_embeddings_folder}/{file_id}/{file_name}.encrypted",
                file_name,
            )
            return success, encrypted_file_path
        return False, None

    async def _encrypt_file(self, file_path: str, file_name: str) -> Optional[str]:
        """
        Encrypt a file and return the path to the encrypted file.

        Args:
            file_path: Path to the file to encrypt
            file_name: Name of the file for logging purposes

        Returns:
            Optional[str]: Path to the encrypted file, or None if encryption failed
        """
        try:
            encrypted_file_path = await asyncio.to_thread(encrypt_file, file_path)
            logging.info(f"Successfully encrypted file: {file_name}")
            return encrypted_file_path
        except Exception as e:
            logging.error(f"Error encrypting file {file_name}: {str(e)}")
            return None

    async def _upload_encrypted_file(
        self, encrypted_file_path: str, destination_path: str, original_filename: str
    ) -> bool:
        """
        Upload an encrypted file to GCS and clean it up afterwards.

        Args:
            encrypted_file_path: Path to the encrypted file
            destination_path: Destination path in GCS
            original_filename: Original filename for logging purposes

        Returns:
            bool: True if upload succeeded, False otherwise
        """
        if not encrypted_file_path:
            return False

        try:
            # Extract file_id from destination path to check for conflicts
            # Expected format: "file-embeddings/{file_id}/{filename}.encrypted"
            path_parts = destination_path.split("/")
            if len(path_parts) >= 3:
                file_id = path_parts[1]

                # Check if there are any other encrypted files in this folder with different names
                prefix = f"{self.gcs_handler.configs.gcp_resource.gcp_embeddings_folder}/{file_id}/"
                blobs = list(self.gcs_handler.bucket.list_blobs(prefix=prefix))

                for blob in blobs:
                    # Check if we have a different PDF already in this directory
                    if (
                        blob.name.endswith(".encrypted")
                        and blob.name != destination_path
                        and not blob.name.endswith(f"{original_filename}.encrypted")
                    ):
                        logging.warning(
                            f"Found conflicting file in {file_id} directory: {blob.name} vs {destination_path}"
                        )
                        # Log this as an error for investigation
                        logging.error(
                            f"CONFLICT DETECTED: Multiple PDFs in same directory. "
                            f"Existing: {blob.name}, New: {destination_path}"
                        )

            files_to_upload = {
                "encrypted_file": (
                    encrypted_file_path,
                    destination_path,
                )
            }
            await asyncio.to_thread(
                self.gcs_handler.upload_to_gcs,
                self.gcs_handler.configs.gcp_resource.bucket_name,
                files_to_upload,
            )
            logging.info(
                f"Successfully uploaded encrypted file for {original_filename}"
            )

            # Clean up encrypted file after upload
            if os.path.exists(encrypted_file_path):
                os.remove(encrypted_file_path)
            return True
        except Exception as e:
            logging.error(f"Error uploading encrypted file: {str(e)}")
            # Clean up encrypted file in case of error
            if os.path.exists(encrypted_file_path):
                os.remove(encrypted_file_path)
            return False
