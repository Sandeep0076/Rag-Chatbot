import asyncio
import hashlib
import logging
import os
import shutil

import aiofiles
from fastapi import UploadFile

from rtl_rag_chatbot_api.chatbot.csv_handler import TabularDataHandler
from rtl_rag_chatbot_api.chatbot.embedding_handler import EmbeddingHandler
from rtl_rag_chatbot_api.chatbot.image_reader import analyze_images
from rtl_rag_chatbot_api.chatbot.utils.encryption import decrypt_file, encrypt_file
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
        gemini_handler: Gemini model handler for image analysis.
    """

    def __init__(self, configs, gcs_handler, gemini_handler=None):
        """
        Initializes the FileHandler with configurations and handlers.

        Args:
            configs: Configuration object containing necessary settings.
            gcs_handler: Google Cloud Storage handler for cloud operations.
            gemini_handler: Optional Gemini handler for image analysis.
        """
        self.configs = configs
        self.gcs_handler = gcs_handler
        self.gemini_handler = gemini_handler

    def calculate_file_hash(self, file_content):
        """
        Calculates the MD5 hash of the given file content.

        Args:
            file_content (bytes): The content of the file.

        Returns:
            str: The hexadecimal digest of the MD5 hash.
        """
        return hashlib.md5(file_content).hexdigest()

    async def _handle_image_analysis(
        self, file_id: str, temp_file_path: str, analysis_files: list
    ) -> None:
        """Handle image analysis using both GPT-4 and Gemini models."""
        try:
            logging.info(f"Starting image analysis for {temp_file_path}")
            analysis_result = await analyze_images(
                temp_file_path, model="both", gemini_handler=self.gemini_handler
            )

            if analysis_result:
                await self._save_analysis_results(
                    file_id, analysis_result, analysis_files
                )

        except Exception as e:
            logging.error(f"Error in image analysis: {str(e)}", exc_info=True)
            raise

    async def _save_analysis_results(
        self, file_id: str, analysis_result: dict, analysis_files: list
    ) -> None:
        """Save analysis results from both models to files."""
        try:
            # Save GPT-4 analysis
            gpt4_analysis_path = f"local_data/{file_id}_gpt4_analysis.txt"
            await self._write_analysis_file(
                gpt4_analysis_path, analysis_result["gpt4_analysis"], "GPT-4"
            )
            analysis_files.append(gpt4_analysis_path)

            # Save Gemini analysis
            gemini_analysis_path = f"local_data/{file_id}_gemini_analysis.txt"
            gemini_content = analysis_result["gemini_analysis"]
            if isinstance(gemini_content, str):
                try:
                    gemini_content = eval(gemini_content)["analysis"]
                except Exception as e:
                    logging.warning(
                        f"Failed to eval Gemini content, using as is: {str(e)}"
                    )

            await self._write_analysis_file(
                gemini_analysis_path, str(gemini_content), "Gemini"
            )
            analysis_files.append(gemini_analysis_path)

        except Exception as e:
            logging.error(f"Error saving analysis results: {str(e)}", exc_info=True)
            raise

    async def _write_analysis_file(
        self, file_path: str, content: str, model_name: str
    ) -> None:
        """Write analysis content to file with proper validation."""
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(content)
            await f.flush()

        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            raise IOError(
                f"Failed to create or write to {model_name} analysis file: {file_path}"
            )

        logging.info(
            f"Successfully wrote {model_name} analysis file. Size: {os.path.getsize(file_path)} bytes"
        )

    async def _handle_new_tabular_data(
        self,
        file_id: str,
        temp_file_path: str,
        file_hash: str,
        original_filename: str,
        username: str,
    ):
        """Handle processing of new tabular data files"""
        metadata = {
            "file_id": file_id,
            "file_hash": file_hash,
            "original_filename": original_filename,
            "is_tabular": True,
            "is_image": False,
            "username": username,
            "embeddings_status": "completed",  # CSV files don't need embeddings
        }

        # Create data directory and prepare SQLite database
        data_dir = f"./chroma_db/{file_id}"
        db_path = os.path.join(data_dir, "tabular_data.db")
        os.makedirs(data_dir, exist_ok=True)

        # Prepare SQLite database
        data_preparer = PrepareSQLFromTabularData(temp_file_path, data_dir)
        data_preparer.run_pipeline()

        # Upload metadata and encrypted database
        try:
            encrypted_db_path = encrypt_file(db_path)
            try:
                files_to_upload = {
                    "metadata": (metadata, f"file-embeddings/{file_id}/file_info.json"),
                    "database": (
                        encrypted_db_path,
                        f"file-embeddings/{file_id}/tabular_data.db.encrypted",
                    ),
                }
                await asyncio.to_thread(
                    self.gcs_handler.upload_to_gcs,
                    self.configs.gcp_resource.bucket_name,
                    files_to_upload,
                )
            finally:
                if os.path.exists(encrypted_db_path):
                    os.remove(encrypted_db_path)
        except Exception as e:
            logging.error(f"Error uploading metadata: {str(e)}")
            raise

        return {
            "file_id": file_id,
            "is_image": False,
            "is_tabular": True,
            "message": "File processed and ready for querying.",
            "status": "success",
            "temp_file_path": temp_file_path,
        }

    async def _handle_existing_tabular_data(
        self, existing_file_id: str, original_filename: str, temp_file_path: str
    ):
        """Handle processing of existing tabular data files"""
        data_dir = f"./chroma_db/{existing_file_id}"
        os.makedirs(data_dir, exist_ok=True)

        # Download and decrypt the database file
        db_path = os.path.join(data_dir, "tabular_data.db")
        encrypted_db_path = os.path.join(data_dir, "tabular_data.db.encrypted")

        try:
            # Download the encrypted database
            self.gcs_handler.download_files_from_folder_by_id(existing_file_id)

            # Decrypt the database if it exists
            if os.path.exists(encrypted_db_path):
                decrypt_file(encrypted_db_path, db_path)
                os.remove(encrypted_db_path)  # Clean up encrypted file
            else:
                logging.error(
                    f"Encrypted database not found for file_id: {existing_file_id}"
                )

            # Initialize handler with decrypted database
            handler = TabularDataHandler(self.configs, existing_file_id)
            # metadata = self.gcs_handler.get_file_info(existing_file_id)
            if handler.initialize_database(is_new_file=False):
                return {
                    "file_id": existing_file_id,
                    "is_image": False,
                    "is_tabular": True,
                    "message": "File exists. Database ready for querying.",
                    "status": "existing",
                    "temp_file_path": None,
                }
        except Exception as e:
            logging.error(f"Error processing existing database: {str(e)}")
            if os.path.exists(encrypted_db_path):
                os.remove(encrypted_db_path)
            if os.path.exists(db_path):
                os.remove(db_path)
            raise

    async def process_file(
        self, file: UploadFile, file_id: str, is_image: bool, username: str
    ) -> dict:
        """Process uploaded file including handling images, tabular data and existing files."""
        try:
            # Sanitize filename
            original_filename = self._sanitize_filename(file.filename)

            # Read and hash file content
            file_content = await file.read()
            file_hash = self.calculate_file_hash(file_content)

            # Determine file type
            file_extension = os.path.splitext(original_filename)[1].lower()
            logging.info(f"Processing file with extension: {file_extension}")

            is_tabular = file_extension in [".csv", ".xlsx", ".xls"]
            is_database = file_extension in [".db", ".sqlite", ".sqlite3"]
            is_text = file_extension in [".txt", ".doc", ".docx"]

            if is_text:
                logging.info(f"Detected text file: {original_filename}")

            # Check for existing file
            existing_file_id = await self.find_existing_file_by_hash_async(file_hash)
            if existing_file_id:
                logging.info(f"Existing file found with hash: {existing_file_id}")
                # file_id = existing_file_id
                embedding_handler = EmbeddingHandler(self.configs, self.gcs_handler)
                google_result = await embedding_handler.check_embeddings_exist(
                    existing_file_id, "gemini-pro"
                )
                azure_result = await embedding_handler.check_embeddings_exist(
                    existing_file_id, "gpt_4_omni"
                )

            # Create necessary directories
            os.makedirs("local_data", exist_ok=True)
            if not existing_file_id:
                chroma_dir = f"./chroma_db/{file_id}"
                os.makedirs(chroma_dir, exist_ok=True)
                logging.info(f"Created directory: {chroma_dir}")

            # Save file locally
            temp_file_path = f"local_data/{file_id}_{original_filename}"
            async with aiofiles.open(temp_file_path, "wb") as buffer:
                await buffer.write(file_content)
            del file_content

            # Process based on file type
            if (
                existing_file_id
                and is_image
                and (
                    not google_result["embeddings_exist"]
                    or not azure_result["embeddings_exist"]
                )
            ):
                analysis_files = []
                await self._handle_image_analysis(
                    existing_file_id, temp_file_path, analysis_files
                )

            elif is_image and not existing_file_id:
                analysis_files = []
                await self._handle_image_analysis(
                    file_id, temp_file_path, analysis_files
                )

            # Prepare metadata for new file
            metadata = {
                "is_image": is_image,
                "is_tabular": is_tabular or is_database,
                "file_hash": file_hash,
                "username": username,
                "original_filename": original_filename,
                "file_id": file_id,
            }

            if is_image and analysis_files:
                metadata.update(
                    {
                        "gpt4_analysis_path": analysis_files[0],
                        "gemini_analysis_path": analysis_files[1],
                        "has_analysis": True,
                    }
                )

            self.gcs_handler.temp_metadata = metadata

            # If it's a new tabular file, prepare SQLite database
            if (is_tabular or is_database) and not existing_file_id:
                return await self._handle_new_tabular_data(
                    file_id, temp_file_path, file_hash, original_filename, username
                )

            if existing_file_id:
                logging.info(f"Found embeddings for: {original_filename}")
                if is_tabular or is_database:
                    return await self._handle_existing_tabular_data(
                        existing_file_id, original_filename, temp_file_path
                    )

                # Check if local embeddings exist first
                azure_path = f"./chroma_db/{existing_file_id}/azure"
                gemini_path = f"./chroma_db/{existing_file_id}/google"
                local_exists = (
                    os.path.exists(azure_path)
                    and os.path.exists(gemini_path)
                    and os.path.exists(os.path.join(azure_path, "chroma.sqlite3"))
                    and os.path.exists(os.path.join(gemini_path, "chroma.sqlite3"))
                )

                if (
                    google_result["embeddings_exist"]
                    and azure_result["embeddings_exist"]
                ):
                    if not local_exists:
                        self.gcs_handler.download_files_from_folder_by_id(
                            existing_file_id
                        )

                # For images, we only need the embeddings to chat
                if is_image:
                    return {
                        "file_id": existing_file_id,
                        "is_image": is_image,
                        "is_tabular": is_tabular,
                        "message": "File already exists and has embeddings.",
                        "status": "existing",
                        "temp_file_path": temp_file_path,  # Keep original temp file for reference
                    }

                # Copy the temp file to match the existing file ID path
                existing_temp_path = (
                    f"local_data/{existing_file_id}_{original_filename}"
                )
                os.makedirs(os.path.dirname(existing_temp_path), exist_ok=True)
                shutil.copy2(temp_file_path, existing_temp_path)
                temp_file_path = existing_temp_path

                return {
                    "file_id": existing_file_id,
                    "is_image": is_image,
                    "is_tabular": is_tabular,
                    "message": "File already exists. Processing database."
                    if is_tabular
                    else "File already exists and has embeddings.",
                    "status": "existing",
                    "temp_file_path": temp_file_path,
                }

            return {
                "file_id": file_id,
                "is_image": is_image,
                "is_tabular": is_tabular,
                "message": "File processed and ready for embedding creation."
                if not is_tabular
                else "File processed and ready for use.",
                "status": "success",
                "temp_file_path": analysis_files[0] if is_image else temp_file_path,
            }

        except Exception as e:
            logging.error(f"Exception in process_file: {str(e)}")
            # Clean up temp files in case of error
            if "temp_file_path" in locals() and os.path.exists(temp_file_path):
                await asyncio.to_thread(os.remove, temp_file_path)
            if (
                "analysis_files" in locals()
                and analysis_files
                and all(os.path.exists(path) for path in analysis_files)
            ):
                await asyncio.to_thread(os.remove, analysis_files[0])
                await asyncio.to_thread(os.remove, analysis_files[1])
            return {
                "status": "error",
                "message": str(e),
                "file_id": file_id,
                "is_image": is_image,
                "temp_file_path": None,
            }

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent security issues."""
        if len(filename) > 100:
            ext = os.path.splitext(filename)[1]
            return filename[:96] + ext
        return filename

    async def download_existing_file(self, file_id: str):
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

    async def find_existing_file_by_hash_async(self, file_hash: str):
        """Asynchronous version of finding existing file by hash."""
        return await asyncio.to_thread(
            self.gcs_handler.find_existing_file_by_hash, file_hash
        )
