import asyncio
import hashlib
import logging
import os
import shutil

import aiofiles
from fastapi import UploadFile

from rtl_rag_chatbot_api.chatbot.chatbot_creator import get_azure_non_rag_response
from rtl_rag_chatbot_api.chatbot.embedding_handler import EmbeddingHandler
from rtl_rag_chatbot_api.chatbot.image_reader import analyze_images
from rtl_rag_chatbot_api.chatbot.utils.encryption import encrypt_file
from rtl_rag_chatbot_api.common.prepare_sqlitedb_from_csv_xlsx import (
    PrepareSQLFromTabularData,
)
from rtl_rag_chatbot_api.common.prompts_storage import BOILERPLATE_PROMPT


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

    def calculate_url_hash(self, url):
        """
        Calculates the MD5 hash of a URL string.

        Args:
            url (str): The URL to hash.

        Returns:
            str: The hexadecimal digest of the MD5 hash.
        """
        return hashlib.md5(url.encode("utf-8")).hexdigest()

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
            "username": [username],  # Store username as an array
            "embeddings_status": "completed",  # CSV files don't need embeddings
        }

        # Create data directory and prepare SQLite database
        data_dir = f"./chroma_db/{file_id}"
        db_path = os.path.join(data_dir, "tabular_data.db")
        os.makedirs(data_dir, exist_ok=True)

        # Prepare SQLite database
        data_preparer = PrepareSQLFromTabularData(temp_file_path, data_dir)
        data_preparer.run_pipeline()

        # Extract database_summary directly without using TabularDataHandler
        try:
            # Import necessary modules for direct database access
            from sqlalchemy import create_engine, inspect, text

            # Create direct connection to the database
            db_url = f"sqlite:///{db_path}"
            engine = create_engine(db_url)
            inspector = inspect(engine)

            # Extract table info directly
            table_info = []
            with engine.connect() as connection:
                for table_name in inspector.get_table_names():
                    columns = inspector.get_columns(table_name)
                    row_count = connection.execute(
                        text(f'SELECT COUNT(*) FROM "{table_name}"')
                    ).scalar()
                    sample_data = connection.execute(
                        text(f'SELECT * FROM "{table_name}" LIMIT 3')
                    ).fetchall()

                    column_stats = {}
                    for column in columns:
                        if hasattr(column["type"], "python_type") and column[
                            "type"
                        ].python_type in (int, float):
                            stats = connection.execute(
                                text(
                                    f'SELECT MIN("{column["name"]}"), MAX("{column["name"]}"), '
                                    f'AVG("{column["name"]}") FROM "{table_name}"'
                                )
                            ).fetchone()
                            column_stats[column["name"]] = {
                                "min": stats[0],
                                "max": stats[1],
                                "avg": stats[2],
                            }

                    table_info.append(
                        {
                            "name": table_name,
                            "columns": [
                                {"name": col["name"], "type": str(col["type"])}
                                for col in columns
                            ],
                            "row_count": row_count,
                            "sample_data": [
                                [str(cell) for cell in row] for row in sample_data
                            ],
                            "column_stats": column_stats,
                        }
                    )

            # Create database summary
            database_summary = {
                "table_count": len(table_info),
                "table_names": [t["name"] for t in table_info],
                "tables": table_info,
            }

            metadata["database_summary"] = database_summary
            logging.info(
                f"Successfully extracted database_summary with {len(table_info)} tables"
            )

        except Exception as e:
            logging.error(
                f"Failed to extract database_summary for new file: {str(e)}",
                exc_info=True,
            )

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

            # download_files_from_folder_by_id already decrypts the database
            # # Decrypt the database if it exists
            # if os.path.exists(encrypted_db_path):
            #     decrypt_file(encrypted_db_path, db_path)
            #     os.remove(encrypted_db_path)  # Clean up encrypted file
            # else:
            #     logging.error(
            #         f"Encrypted database not found for file_id: {existing_file_id}"
            #     )

            # Check if file_info.json has database_summary, if not, add it
            file_info = self.gcs_handler.get_file_info(existing_file_id)

            # Extract database_summary if it doesn't exist
            if "database_summary" not in file_info:
                try:
                    # Import necessary modules for direct database access
                    from sqlalchemy import create_engine, inspect, text

                    # Create direct connection to the database
                    db_url = f"sqlite:///{db_path}"
                    engine = create_engine(db_url)
                    inspector = inspect(engine)

                    # Extract table info directly
                    table_info = []
                    with engine.connect() as connection:
                        for table_name in inspector.get_table_names():
                            columns = inspector.get_columns(table_name)
                            row_count = connection.execute(
                                text(f'SELECT COUNT(*) FROM "{table_name}"')
                            ).scalar()
                            sample_data = connection.execute(
                                text(f'SELECT * FROM "{table_name}" LIMIT 3')
                            ).fetchall()

                            column_stats = {}
                            for column in columns:
                                if hasattr(column["type"], "python_type") and column[
                                    "type"
                                ].python_type in (int, float):
                                    stats = connection.execute(
                                        text(
                                            f'SELECT MIN("{column["name"]}"), MAX("{column["name"]}"), '
                                            f'AVG("{column["name"]}") FROM "{table_name}"'
                                        )
                                    ).fetchone()
                                    column_stats[column["name"]] = {
                                        "min": stats[0],
                                        "max": stats[1],
                                        "avg": stats[2],
                                    }

                            table_info.append(
                                {
                                    "name": table_name,
                                    "columns": [
                                        {"name": col["name"], "type": str(col["type"])}
                                        for col in columns
                                    ],
                                    "row_count": row_count,
                                    "sample_data": [
                                        [str(cell) for cell in row]
                                        for row in sample_data
                                    ],
                                    "column_stats": column_stats,
                                }
                            )

                    # Create database summary
                    database_summary = {
                        "table_count": len(table_info),
                        "table_names": [t["name"] for t in table_info],
                        "tables": table_info,
                    }

                    # Update file_info.json with database_summary
                    self.gcs_handler.update_file_info(
                        existing_file_id, {"database_summary": database_summary}
                    )
                    logging.info(
                        f"Added database_summary to existing file_info.json with {len(table_info)} tables"
                    )

                except Exception as e:
                    logging.error(
                        f"Failed to extract database_summary for existing file: {str(e)}",
                        exc_info=True,
                    )

            # Return success directly - we've already extracted the database summary if needed
            # No need to initialize TabularDataHandler which might fail
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

    async def _encrypt_file(self, file_path: str, file_name: str) -> str:
        """
        Encrypt a file and return the path to the encrypted file.

        Args:
            file_path (str): Path to the file to encrypt
            file_name (str): Name of the file for logging purposes

        Returns:
            str: Path to the encrypted file, or None if encryption failed
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
    ):
        """
        Upload an encrypted file to GCS and clean it up afterwards.

        Args:
            encrypted_file_path (str): Path to the encrypted file
            destination_path (str): Destination path in GCS
            original_filename (str): Original filename for logging purposes

        Returns:
            bool: True if upload succeeded, False otherwise
        """
        if not encrypted_file_path:
            return False

        try:
            files_to_upload = {
                "encrypted_file": (
                    encrypted_file_path,
                    destination_path,
                )
            }
            await asyncio.to_thread(
                self.gcs_handler.upload_to_gcs,
                self.configs.gcp_resource.bucket_name,
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

    async def _determine_file_types(self, original_filename):
        """Determine file types based on extension."""
        file_extension = os.path.splitext(original_filename)[1].lower()
        logging.info(f"Processing file with extension: {file_extension}")

        is_tabular = file_extension in [
            ".csv",
            ".xlsx",
            ".xls",
            ".db",
            ".sqlite",
            ".sqlite3",
        ]
        is_database = file_extension in [".db", ".sqlite", ".sqlite3"]
        is_text = file_extension in [".txt", ".doc", ".docx"]

        if is_text:
            logging.info(f"Detected text file: {original_filename}")

        return is_tabular, is_database, is_text

    async def _prepare_file_directories(self, file_id, existing_file_id):
        """Create necessary directories for file processing."""
        os.makedirs("local_data", exist_ok=True)
        if not existing_file_id:
            chroma_dir = f"./chroma_db/{file_id}"
            os.makedirs(chroma_dir, exist_ok=True)
            logging.info(f"Created directory: {chroma_dir}")

    async def _process_image_analysis(
        self,
        is_image,
        existing_file_id,
        google_result,
        azure_result,
        actual_file_id,
        temp_file_path,
        metadata,
    ):
        """Process image analysis for new or incomplete embeddings."""
        if not is_image or (
            existing_file_id
            and google_result["embeddings_exist"]
            and azure_result["embeddings_exist"]
        ):
            return []

        analysis_files = []
        # Use the actual_file_id for image analysis to ensure consistency
        await self._handle_image_analysis(
            actual_file_id, temp_file_path, analysis_files
        )
        metadata.update(
            {
                "gpt4_analysis_path": analysis_files[0],
                "gemini_analysis_path": analysis_files[1],
                "has_analysis": True,
            }
        )
        return analysis_files

    async def _check_local_embeddings(self, existing_file_id):
        """Check if local embeddings exist."""
        azure_path = f"./chroma_db/{existing_file_id}/azure"
        gemini_path = f"./chroma_db/{existing_file_id}/google"
        return (
            os.path.exists(azure_path)
            and os.path.exists(gemini_path)
            and os.path.exists(os.path.join(azure_path, "chroma.sqlite3"))
            and os.path.exists(os.path.join(gemini_path, "chroma.sqlite3"))
        )

    async def _handle_existing_file_encryption(
        self,
        encrypted_file_path,
        existing_file_id,
        original_filename,
        temp_file_path,
        is_tabular,
        is_database,
    ):
        """Check and handle encryption for existing files."""
        # Check if encrypted file exists in GCS
        encrypted_file_exists = await asyncio.to_thread(
            self.gcs_handler.check_file_exists,
            f"file-embeddings/{existing_file_id}/{original_filename}.encrypted",
        )

        # If encrypted file doesn't exist, encrypt and upload the current file
        if not encrypted_file_exists and not is_tabular and not is_database:
            # For existing files, we need to ensure we have an encrypted version
            existing_encrypted_file_path = encrypted_file_path

            # Encrypt the file if it wasn't encrypted earlier
            if not existing_encrypted_file_path:
                existing_encrypted_file_path = await self._encrypt_file(
                    temp_file_path, original_filename
                )

            # Upload the encrypted file
            if existing_encrypted_file_path:
                await self._upload_encrypted_file(
                    existing_encrypted_file_path,
                    f"file-embeddings/{existing_file_id}/{original_filename}.encrypted",
                    original_filename,
                )

    async def _handle_existing_file(
        self,
        existing_file_id,
        encrypted_file_path,
        temp_file_path,
        original_filename,
        is_image,
        is_tabular,
        is_database,
        username,
        google_result,
        azure_result,
    ):
        """Handle processing for files that already exist in the system."""
        # Early return if the file doesn't exist
        if not existing_file_id:
            return None

        logging.info(f"Found embeddings for: {original_filename}")

        # Update file_info.json with the new username
        self.gcs_handler.update_file_info(existing_file_id, {"username": username})
        logging.info(f"Updated file_info.json with username: {username}")

        # Handle encryption for existing files
        await self._handle_existing_file_encryption(
            encrypted_file_path,
            existing_file_id,
            original_filename,
            temp_file_path,
            is_tabular,
            is_database,
        )

        # Handle tabular data separately
        if is_tabular or is_database:
            return await self._handle_existing_tabular_data(
                existing_file_id, original_filename, temp_file_path
            )

        # Check if local embeddings exist first
        local_exists = await self._check_local_embeddings(existing_file_id)

        # Download embeddings if they exist remotely but not locally
        if (
            google_result["embeddings_exist"]
            and azure_result["embeddings_exist"]
            and not local_exists
        ):
            self.gcs_handler.download_files_from_folder_by_id(existing_file_id)

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
        existing_temp_path = f"local_data/{existing_file_id}_{original_filename}"
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

    async def process_file(
        self, file: UploadFile, file_id: str, is_image: bool, username: str
    ) -> dict:
        """Process uploaded file including handling images, tabular data and existing files."""
        try:
            # breakpoint()
            # Sanitize filename
            original_filename = self._sanitize_filename(file.filename)

            # Read and hash file content
            file_content = await file.read()
            file_hash = self.calculate_file_hash(file_content)
            # Determine file types
            is_tabular, is_database, is_text = await self._determine_file_types(
                original_filename
            )

            # Check for existing file and get embedding status
            existing_file_id = None
            google_result = {"embeddings_exist": False}
            azure_result = {"embeddings_exist": False}

            existing_file_id = await self.find_existing_file_by_hash_async(file_hash)
            if existing_file_id:
                # Get embedding status for existing file
                embedding_handler = EmbeddingHandler(self.configs, self.gcs_handler)
                google_result = await embedding_handler.check_embeddings_exist(
                    existing_file_id, "gemini-flash"
                )
                azure_result = await embedding_handler.check_embeddings_exist(
                    existing_file_id, "gpt_4o_mini"
                )
                logging.info(f"Existing file found with hash: {existing_file_id}")

            # Create necessary directories
            await self._prepare_file_directories(file_id, existing_file_id)

            # Save file locally
            temp_file_path = f"local_data/{file_id}_{original_filename}"
            async with aiofiles.open(temp_file_path, "wb") as buffer:
                await buffer.write(file_content)
            del file_content

            # Encrypt the original uploaded file (tabular files are handled separately)
            encrypted_file_path = None
            if not is_tabular:
                encrypted_file_path = await self._encrypt_file(
                    temp_file_path, original_filename
                )

            # Determine the actual file_id to use - if an existing file is found, use that ID
            actual_file_id = existing_file_id if existing_file_id else file_id

            # Prepare metadata for file with consistent file_id
            metadata = {
                "is_image": is_image,
                "is_tabular": is_tabular or is_database,
                "file_hash": file_hash,
                "username": [username],  # Store username as an array
                "original_filename": original_filename,
                "file_id": actual_file_id,  # Always use the actual file_id (existing or new)
            }

            # Process based on file type
            # Process image analysis if needed
            analysis_files = await self._process_image_analysis(
                is_image,
                existing_file_id,
                google_result,
                azure_result,
                actual_file_id,
                temp_file_path,
                metadata,
            )

            # Add encryption status to metadata
            metadata["is_encrypted"] = encrypted_file_path is not None
            self.gcs_handler.temp_metadata = metadata

            # If it's a new tabular file, prepare SQLite database
            if (is_tabular or is_database) and not existing_file_id:
                return await self._handle_new_tabular_data(
                    actual_file_id,
                    temp_file_path,
                    file_hash,
                    original_filename,
                    username,
                )

            # Handle existing file case
            existing_file_result = await self._handle_existing_file(
                existing_file_id,
                encrypted_file_path,
                temp_file_path,
                original_filename,
                is_image,
                is_tabular,
                is_database,
                username,
                google_result,
                azure_result,
            )

            if existing_file_result:
                return existing_file_result

            # Upload the encrypted file to GCS if available (skip for tabular files - handled separately)
            if not is_tabular and encrypted_file_path:
                await self._upload_encrypted_file(
                    encrypted_file_path,
                    f"file-embeddings/{actual_file_id}/{original_filename}.encrypted",
                    original_filename,
                )

            # Return response for new files
            return {
                "file_id": actual_file_id,  # Always use the consistent file_id
                "is_image": is_image,
                "is_tabular": is_tabular,
                "message": "File processed and ready for embedding creation."
                if not is_tabular
                else "File processed and ready for use.",
                "status": "success",
                "temp_file_path": analysis_files[0]
                if is_image and analysis_files
                else temp_file_path,
            }

        except Exception as e:
            logging.error(f"Exception in process_file: {str(e)}")
            # Clean up temp files in case of error
            if "temp_file_path" in locals() and os.path.exists(temp_file_path):
                await asyncio.to_thread(os.remove, temp_file_path)
            if (
                "encrypted_file_path" in locals()
                and encrypted_file_path
                and os.path.exists(encrypted_file_path)
            ):
                await asyncio.to_thread(os.remove, encrypted_file_path)
            # We don't need to check for existing_encrypted_file_path as it's not defined in this scope
            # This was likely added during refactoring but is not needed here
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

    async def process_urls(
        self,
        urls: str,
        username: str,
        temp_file_id: str,
        background_tasks,
        embedding_handler=None,
    ):
        """
        Process URLs from the input and create embeddings from their content.

        Args:
            urls (str): The URLs to process, separated by commas or newlines
            username (str): The username of the user uploading the URLs
            temp_file_id (str): A temporary file ID for the URL content
            background_tasks: Background tasks queue for async processing
            embedding_handler: Optional embedding handler instance

        Returns:
            dict: Dictionary containing file_id, status, and other metadata
        """
        from rtl_rag_chatbot_api.chatbot.website_handler import WebsiteHandler

        # Handle both comma and newline separated URLs
        urls_normalized = urls.replace("\n", ",")
        url_list = [url.strip() for url in urls_normalized.split(",") if url.strip()]

        if not url_list:
            from fastapi import HTTPException

            raise HTTPException(status_code=400, detail="No valid URLs provided")

        # Create a URL hash for all URLs combined
        url_hash = self.calculate_url_hash(",".join(url_list))

        # Check if we've already processed these URLs
        existing_file_id = await self.find_existing_file_by_hash_async(url_hash)

        if existing_file_id:
            # Update the file info with the new username
            self.gcs_handler.update_file_info(existing_file_id, {"username": username})
            return {
                "file_id": existing_file_id,
                "status": "existing",
                "message": "URLs already processed",
                "is_image": False,
                "is_tabular": False,
                "original_filename": "url_content.txt",
                "temp_file_path": None,
            }

        # Process the URLs and save content to a text file
        website_handler = WebsiteHandler()

        # Create directory if it doesn't exist
        os.makedirs("local_data", exist_ok=True)

        # Create a text file to store the extracted content
        temp_file_path = f"local_data/{temp_file_id}_url_content.txt"

        with open(temp_file_path, "w", encoding="utf-8") as f:
            for i, url in enumerate(url_list):
                # Add header for this URL with clear separation
                f.write(f"This text is extracted from URL -- {url}\n\n")

                # Helper function to count words in text
                def count_words(text):
                    if not text:
                        return 0
                    # Split by whitespace and count non-empty words
                    words = [word for word in text.split() if word.strip()]
                    return len(words)

                # Extract content from the URL
                try:
                    documents = website_handler.get_vectorstore_from_url(url)

                    # Check if we have valid content
                    if documents and len(documents) > 0:
                        content = documents[0].page_content
                        word_count = count_words(content)

                        # If word count is less than 150, use BOILERPLATE_PROMPT to verify content quality
                        if word_count < 150:
                            extraction_result = get_azure_non_rag_response(
                                self.configs, BOILERPLATE_PROMPT
                            )
                            is_substantive = extraction_result == "True"
                        else:
                            # If we have more than 150 words, assume it's substantive content
                            is_substantive = True

                        if is_substantive:
                            # Write the content to the file
                            f.write(documents[0].page_content)
                            f.write(f"\n\nWord count: {word_count} words")
                        else:
                            # If content is not substantive, write an error message to the file
                            # but continue processing other URLs if there are any
                            error_msg = (
                                f"Error: Content from {url} appears to be boilerplate "
                                f"or insufficient (Word count: {word_count})"
                            )
                            f.write(error_msg)

                            # If this is the only URL, return an error immediately
                            if len(url_list) == 1:
                                return {
                                    "file_id": temp_file_id,
                                    "status": "error",
                                    "message": "The website is not allowing to extract sufficient text, "
                                    "please try another website.",
                                    "is_image": False,
                                    "is_tabular": False,
                                    "original_filename": "url_content.txt",
                                    "temp_file_path": temp_file_path,
                                }
                    else:
                        f.write(f"Error: Could not extract content from {url}")
                        return {
                            "file_id": temp_file_id,
                            "status": "error",
                            "message": "The website is not allowing to extract text, please try another website.",
                            "is_image": False,
                            "is_tabular": False,
                            "original_filename": "url_content.txt",
                            "temp_file_path": temp_file_path,
                        }
                except Exception as e:
                    f.write(f"Error extracting content from {url}: {str(e)}")

                # Add footer for this URL
                f.write(f"\n\nText extraction for the website {url} finished\n\n")

                # Add separator between URLs, but not after the last one
                if i < len(url_list) - 1:
                    f.write("-" * 80 + "\n\n")

        # Check if the file contains any actual content or just error messages
        with open(temp_file_path, "r", encoding="utf-8") as check_file:
            file_content = check_file.read()
            # Check if the file contains any substantive content or only error messages
            if all(
                f"Error: Content from {url}" in file_content for url in url_list
            ) or all(
                f"Error: Could not extract content from {url}" in file_content
                for url in url_list
            ):
                # All URLs failed to extract substantive content
                website_handler.cleanup()
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                return {
                    "file_id": temp_file_id,
                    "status": "error",
                    "message": "Could not extract sufficient content from any of the "
                    "provided URLs. Please try different URLs.",
                    "is_image": False,
                    "is_tabular": False,
                    "original_filename": "url_content.txt",
                    "temp_file_path": None,
                }

        # Create metadata for the file
        metadata = {
            "file_hash": url_hash,
            "original_filename": "url_content.txt",
            "username": [username],
            "is_url": True,
            "urls": url_list,
            "file_id": temp_file_id,
        }

        # Save metadata
        self.gcs_handler.temp_metadata = metadata

        # Import here to avoid circular imports
        from rtl_rag_chatbot_api.app import SessionLocal, create_embeddings_background

        # Schedule background task to create embeddings
        background_tasks.add_task(
            create_embeddings_background,
            temp_file_id,
            temp_file_path,
            embedding_handler,  # Use the embedding_handler passed as parameter
            self.configs,
            SessionLocal,
            [username],
        )

        # Clean up the website handler
        website_handler.cleanup()

        return {
            "file_id": temp_file_id,
            "status": "success",
            "message": "URLs processed successfully",
            "is_image": False,
            "is_tabular": False,
            "original_filename": "url_content.txt",
            "temp_file_path": temp_file_path,
        }
