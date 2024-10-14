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
        Asynchronously processes an uploaded file, including hash calculation, duplicate checking,
        and storage in Google Cloud Storage.

        Args:
            file (UploadFile): The uploaded file object.
            file_id (str): Unique identifier for the file.
            is_image (bool): Flag indicating if the file is an image.
            username (str): The username associated with the uploaded file.

        Returns:
            dict: A dictionary containing processing results, including:
                - file_id (str): The unique identifier for the file.
                - is_image (bool): Flag indicating if the file is an image.
                - is_tabular (bool): Flag indicating if the file is a tabular data file.
                - message (str): A status message describing the processing outcome.
                - status (str): The processing status ('new', 'existing', or 'pending_embeddings').
                - temp_file_path (str): The path to the temporary stored file.

        Raises:
            Exception: If an error occurs during file processing.
        """
        original_filename = file.filename
        file_content = await file.read()
        file_hash = self.calculate_file_hash(file_content)
        is_tabular = original_filename.lower().endswith((".csv", ".xlsx", ".xls"))

        existing_file_id = self.gcs_handler.find_existing_file_by_hash(file_hash)

        temp_file_path = f"local_data/{file_id}_{original_filename}"
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file_content)
        del file_content
        if existing_file_id:
            existing_file_info = self.gcs_handler.get_file_info(existing_file_id)
            if is_tabular:
                return {
                    "file_id": existing_file_id,
                    "is_image": is_image,
                    "is_tabular": is_tabular,
                    "message": "File already exists. Downloading necessary files.",
                    "status": "existing",
                    "temp_file_path": temp_file_path,
                }
            elif existing_file_info.get("embeddings"):
                return {
                    "file_id": existing_file_id,
                    "is_image": is_image,
                    "is_tabular": is_tabular,
                    "message": "File already exists and has embeddings.",
                    "status": "existing",
                    "temp_file_path": temp_file_path,
                }
            else:
                return {
                    "file_id": existing_file_id,
                    "is_image": is_image,
                    "is_tabular": is_tabular,
                    "message": "File exists but embeddings need to be created.",
                    "status": "pending_embeddings",
                    "temp_file_path": temp_file_path,
                }

        # Store metadata in GCS
        metadata = {
            "is_image": is_image,
            "is_tabular": is_tabular,
            "file_hash": file_hash,
            "username": username,
            "original_filename": original_filename,
            "file_id": file_id,
            "embeddings_status": "completed" if is_tabular else "pending",
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
            "is_tabular": is_tabular,
            "message": "File processed and metadata stored successfully. Embeddings pending."
            if not is_tabular
            else "File processed and ready for use.",
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
