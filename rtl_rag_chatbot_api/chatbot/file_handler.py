import hashlib
import os

from fastapi import UploadFile


class FileHandler:
    """
    Handles file processing, encryption, and storage operations.

    This class manages file uploads, calculates file hashes, checks for existing files,
    encrypts new files, and interacts with Google Cloud Storage for file storage and retrieval.

    Attributes:
        configs: Configuration object containing necessary settings.
        gcs_handler: Google Cloud Storage handler for cloud operations.

    Methods:
        calculate_file_hash(file_content): Calculates MD5 hash of file content.
        process_file(file, file_id, is_image): Processes uploaded files, including encryption and storage.
        download_existing_file(file_id): Downloads existing files from Google Cloud Storage.
    """

    def __init__(self, configs, gcs_handler):
        """
        Initializes the FileHandler with configurations and GCS handler.

        Args:
            configs: Configuration object containing necessary settings.
            gcs_handler: Google Cloud Storage handler for cloud operations.
        """
        self.configs = configs
        self.gcs_handler = gcs_handler

    def calculate_file_hash(self, file_content):
        """
        Calculates the MD5 hash of the given file content.

        Args:
            file_content (bytes): The content of the file.

        Returns:
            str: The hexadecimal digest of the MD5 hash.
        """
        return hashlib.md5(file_content).hexdigest()

    async def process_file(
        self, file: UploadFile, file_id: str, is_image: bool, username: str
    ):
        """
        Processes an uploaded file, including hash calculation, duplicate checking,
        encryption, and storage in Google Cloud Storage.

        Args:
            file (UploadFile): The uploaded file object.
            file_id (str): Unique identifier for the file.
            is_image (bool): Flag indicating if the file is an image.

        Returns:
            dict: A dictionary containing processing results, including file_id,
                  is_image flag, status message, and processing status.
        """
        original_filename = file.filename
        file_content = await file.read()
        file_hash = self.calculate_file_hash(file_content)

        existing_file_id = self.gcs_handler.find_existing_file_by_hash(file_hash)

        if existing_file_id:
            existing_file_info = self.gcs_handler.get_file_info(existing_file_id)
            if existing_file_info.get("embeddings"):
                return {
                    "file_id": existing_file_id,
                    "is_image": is_image,
                    "message": "File already exists and has embeddings.",
                    "status": "existing",
                }
            else:
                return {
                    "file_id": existing_file_id,
                    "is_image": is_image,
                    "message": "File exists but embeddings need to be created.",
                    "status": "pending_embeddings",
                }

        # Save file temporarily
        temp_file_path = f"local_data/{file_id}_{original_filename}"
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file_content)

        # Store metadata in GCS
        metadata = {
            "is_image": is_image,
            "file_hash": file_hash,
            "username": username,
            "original_filename": original_filename,
            "file_id": file_id,
            "embeddings_status": "pending",
        }
        self.gcs_handler.upload_to_gcs(
            self.configs.gcp_resource.bucket_name,
            {
                "metadata": (metadata, f"file-embeddings/{file_id}/file_info.json"),
            },
        )

        return {
            "file_id": file_id,
            "is_image": is_image,
            "message": "File processed and metadata stored successfully. Embeddings pending.",
            "status": "new",
            "temp_file_path": temp_file_path,
        }

    def download_existing_file(self, file_id: str):
        """
        Downloads existing files from Google Cloud Storage for a given file_id.

        Args:
            file_id (str): Unique identifier for the file to download.

        Returns:
            bool: True if download is successful, False otherwise.
        """
        chroma_db_path = f"./chroma_db/{file_id}"
        os.makedirs(chroma_db_path, exist_ok=True)

        try:
            self.gcs_handler.download_files_from_folder_by_id(file_id)
            return True
        except Exception as e:
            print(f"Error downloading embeddings: {str(e)}")
            return False
