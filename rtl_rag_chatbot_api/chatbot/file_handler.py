import hashlib
import os
import shutil

from fastapi import UploadFile

from rtl_rag_chatbot_api.chatbot.image_reader import analyze_images
from rtl_rag_chatbot_api.common.prepare_sqlitedb_from_csv_xlsx import (
    PrepareSQLFromTabularData,
)


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
        Process uploaded file including handling images, tabular data and existing files.
        """
        try:
            original_filename = file.filename
            file_content = await file.read()
            file_hash = self.calculate_file_hash(file_content)
            is_tabular = original_filename.lower().endswith((".csv", ".xlsx", ".xls"))

            existing_file_id = self.gcs_handler.find_existing_file_by_hash(file_hash)

            # Always save the uploaded file first
            temp_file_path = f"local_data/{file_id}_{original_filename}"
            os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
            with open(temp_file_path, "wb") as buffer:
                buffer.write(file_content)
            del file_content

            # For new image files, analyze first
            analysis_text_path = None
            if is_image and not existing_file_id:
                # Analyze image and store analysis
                analysis_result = analyze_images(temp_file_path)
                analysis_text_path = f"local_data/{file_id}_analysis.txt"
                with open(analysis_text_path, "w") as f:
                    f.write(analysis_result[0]["analysis"])

            if existing_file_id:
                if is_tabular:
                    # For tabular files, always prepare SQLite database with new file
                    data_dir = f"./chroma_db/{existing_file_id}"
                    os.makedirs(data_dir, exist_ok=True)

                    # Prepare SQLite database
                    data_preparer = PrepareSQLFromTabularData(temp_file_path, data_dir)
                    data_preparer.run_pipeline()
                else:
                    # For non-tabular files, download existing embeddings
                    self.gcs_handler.download_files_from_folder_by_id(existing_file_id)

                    # Copy the temp file to match the existing file ID path
                    existing_temp_path = (
                        f"local_data/{existing_file_id}_{original_filename}"
                    )
                    os.makedirs(os.path.dirname(existing_temp_path), exist_ok=True)
                    shutil.copy2(temp_file_path, existing_temp_path)
                    temp_file_path = existing_temp_path

                    if is_image:
                        # For existing images, copy analysis file if it exists
                        existing_analysis_path = (
                            f"local_data/{existing_file_id}_analysis.txt"
                        )
                        if os.path.exists(analysis_text_path):
                            shutil.copy2(analysis_text_path, existing_analysis_path)
                            analysis_text_path = existing_analysis_path

                return {
                    "file_id": existing_file_id,
                    "is_image": is_image,
                    "is_tabular": is_tabular,
                    "message": "File already exists. Processing database."
                    if is_tabular
                    else "File already exists and has embeddings.",
                    "status": "existing",
                    "temp_file_path": analysis_text_path
                    if is_image
                    else temp_file_path,
                }

            # Prepare metadata for new file
            metadata = {
                "is_image": is_image,
                "is_tabular": is_tabular,
                "file_hash": file_hash,
                "username": username,
                "original_filename": original_filename,
                "file_id": file_id,
                "embeddings_status": "pending",
            }

            # Add analysis info to metadata for new images
            if is_image and analysis_text_path:
                metadata.update(
                    {"analysis_path": analysis_text_path, "has_analysis": True}
                )

            # Store metadata in GCS
            self.gcs_handler.upload_to_gcs(
                self.configs.gcp_resource.bucket_name,
                {
                    "metadata": (metadata, f"file-embeddings/{file_id}/file_info.json"),
                },
            )

            # If it's a new tabular file, prepare SQLite database
            if is_tabular:
                data_dir = f"./chroma_db/{file_id}"
                os.makedirs(data_dir, exist_ok=True)
                data_preparer = PrepareSQLFromTabularData(temp_file_path, data_dir)
                data_preparer.run_pipeline()

            return {
                "file_id": file_id,
                "is_image": is_image,
                "is_tabular": is_tabular,
                "message": "File processed and metadata stored successfully. Embeddings pending."
                if not is_tabular
                else "File processed and ready for use.",
                "status": "new",
                "temp_file_path": analysis_text_path if is_image else temp_file_path,
            }
        except Exception as e:
            print(f"Exception in process_file: {str(e)}")
            # Clean up temp files in case of error
            if "temp_file_path" in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if (
                "analysis_text_path" in locals()
                and analysis_text_path
                and os.path.exists(analysis_text_path)
            ):
                os.remove(analysis_text_path)
            raise

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
