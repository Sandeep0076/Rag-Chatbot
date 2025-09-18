import asyncio
import hashlib
import logging
import os
import shutil
import time
import uuid

import aiofiles
from fastapi import UploadFile

from rtl_rag_chatbot_api.chatbot.embedding_handler import EmbeddingHandler
from rtl_rag_chatbot_api.chatbot.image_reader import analyze_images
from rtl_rag_chatbot_api.chatbot.utils.encryption import encrypt_file
from rtl_rag_chatbot_api.chatbot.utils.file_encryption_manager import (
    FileEncryptionManager,
)
from rtl_rag_chatbot_api.common.base_handler import BaseRAGHandler
from rtl_rag_chatbot_api.common.prepare_sqlitedb_from_csv_xlsx import (
    PrepareSQLFromTabularData,
)


class FileHandler:
    """
    Handles file processing operations including upload, download, and metadata management.
    Supports both local and cloud storage operations.
    """

    def __init__(self, configs, gcs_handler, gemini_handler=None, db_session=None):
        """
        Initializes the FileHandler with configurations and handlers.

        Args:
            configs: Application configuration
            gcs_handler: GCS handler for cloud operations
            gemini_handler: Optional Gemini handler for AI operations
            db_session: Optional database session for database operations
        """
        self.configs = configs
        self.gcs_handler = gcs_handler
        self.gemini_handler = gemini_handler
        self.db_session = db_session
        self.use_file_hash_db = getattr(configs, "use_file_hash_db", False)
        self.encryption_manager = FileEncryptionManager(gcs_handler)
        # Temporary storage for extracted text to pass to embedding handler
        self._extracted_text_storage = {}

    def update_db_session(self, db_session):
        """
        Update the database session for this FileHandler instance.

        Args:
            db_session: New database session
        """
        self.db_session = db_session

    def store_extracted_text(self, file_id: str, extracted_text: str):
        """
        Store extracted text temporarily for passing to embedding handler.

        Args:
            file_id: The file ID
            extracted_text: The extracted text to store
        """
        self._extracted_text_storage[file_id] = extracted_text
        logging.info(
            f"Stored extracted text for {file_id} (length: {len(extracted_text)})"
        )

    def get_extracted_text(self, file_id: str) -> str:
        """
        Retrieve and remove extracted text for a file.

        Args:
            file_id: The file ID

        Returns:
            The extracted text or None if not found
        """
        extracted_text = self._extracted_text_storage.pop(file_id, None)
        if extracted_text:
            logging.info(
                f"Retrieved extracted text for {file_id} (length: {len(extracted_text)})"
            )
        return extracted_text

    async def store_file_hash_in_db(
        self,
        file_id: str,
        file_hash: str,
        filename: str = None,
        embedding_type: str = None,
    ):
        """
        Store file hash and filename in the database if the feature is enabled.

        Args:
            file_id: The file ID
            file_hash: The file hash to store
            filename: The original filename to store (optional)
            embedding_type: The embedding type to use (if None, uses configurable default)
        """
        if self.use_file_hash_db:
            try:
                from rtl_rag_chatbot_api.app import get_db_session
                from rtl_rag_chatbot_api.common.db import insert_file_info_record

                with get_db_session() as db_session:
                    result = insert_file_info_record(
                        db_session,
                        file_id,
                        file_hash,
                        filename,
                        embedding_type,
                        self.configs,
                    )
                    if result["status"] == "success":
                        logging.info(
                            f"Successfully stored file hash and filename in database for file_id: {file_id}"
                        )
                    else:
                        logging.error(
                            f"Failed to store file hash in database: {result['message']}"
                        )
            except Exception as e:
                logging.error(f"Error storing file hash in database: {str(e)}")

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
        """Handle image analysis using GPT-4 (Azure) model as part of unified Azure approach."""
        try:
            logging.info(f"Starting image analysis for {temp_file_path}")
            # Run image analysis (only using GPT-4/Azure as part of unified Azure approach)
            analysis_result = await analyze_images(temp_file_path, model="gpt4-omni")

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
        """Save analysis results from Azure GPT-4 model to files.

        Note: As part of the unified Azure approach, we only use GPT-4 (Azure) for image analysis,
        but maintain compatibility by creating a placeholder for Gemini analysis.
        """
        try:
            # Save GPT-4 analysis
            gpt4_analysis_path = f"local_data/{file_id}_gpt4_analysis.txt"
            await self._write_analysis_file(
                gpt4_analysis_path, analysis_result["gpt4_analysis"], "GPT-4"
            )
            analysis_files.append(gpt4_analysis_path)

            # Create placeholder for Gemini analysis to maintain compatibility
            gemini_analysis_path = f"local_data/{file_id}_gemini_analysis.txt"
            placeholder_content = "This file is a placeholder. Using unified Azure approach for image analysis."

            await self._write_analysis_file(
                gemini_analysis_path, placeholder_content, "Gemini (Placeholder)"
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
            "migrated": False,  # New tabular files are not migrated
        }

        # Create data directory and prepare SQLite database
        data_dir = f"./chroma_db/{file_id}"
        db_path = os.path.join(data_dir, "tabular_data.db")
        os.makedirs(data_dir, exist_ok=True)

        # Prepare SQLite database
        data_preparer = PrepareSQLFromTabularData(temp_file_path, data_dir)
        pipeline_success = data_preparer.run_pipeline()

        # Check if pipeline failed
        if not pipeline_success:
            logging.error(f"Failed to prepare database from file: {original_filename}")
            raise ValueError(
                f"Failed to process tabular file: {original_filename}. "
                f"The file may be corrupted, empty, or in an unsupported format."
            )

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
                        text(f'SELECT * FROM "{table_name}" LIMIT 2')
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
                "tables": [],
            }
            for t in table_info:
                database_summary["tables"].append(
                    {
                        "name": t["name"],
                        "columns": t["columns"],
                        "row_count": t["row_count"],
                        "top_rows": t[
                            "sample_data"
                        ],  # Use "top_rows" to match csv_handler format
                        "column_stats": t["column_stats"],
                    }
                )

            # Validate that the database has valid tables with data
            if len(table_info) == 0:
                logging.error(
                    f"No tables found in the database for file: {original_filename}"
                )
                raise ValueError(
                    f"No tables found in the database for file: {original_filename}. "
                    f"The file may be empty or corrupted."
                )

            # Check if all tables are empty
            total_rows = sum(table["row_count"] for table in table_info)
            if total_rows == 0:
                logging.error(
                    f"All tables are empty in the database for file: {original_filename}"
                )
                raise ValueError(
                    f"All tables are empty in the database for file: {original_filename}. "
                    f"Please check if the file contains valid data."
                )

            metadata["database_summary"] = database_summary
            logging.info(
                f"Successfully extracted database_summary with {len(table_info)} tables "
                f"and {total_rows} total rows"
            )

        except Exception as e:
            logging.error(
                f"Failed to extract database_summary for new file: {str(e)}",
                exc_info=True,
            )
            # Re-raise the exception to prevent continuing with invalid data
            if isinstance(e, ValueError):
                # Re-raise ValueError as-is (our validation errors)
                raise
            else:
                # Wrap other exceptions with a descriptive message
                raise ValueError(
                    f"Failed to analyze database structure for file: {original_filename}. {str(e)}"
                )

        # Upload metadata and encrypted database
        try:
            # Run encryption in a thread to avoid blocking
            encrypted_db_path = await asyncio.to_thread(encrypt_file, db_path)
            try:
                files_to_upload = {
                    "metadata": (
                        metadata,
                        f"{self.configs.gcp_resource.gcp_embeddings_folder}/{file_id}/file_info.json",
                    ),
                    "database": (
                        encrypted_db_path,
                        f"{self.configs.gcp_resource.gcp_embeddings_folder}/{file_id}/tabular_data.db.encrypted",
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

        # Store file hash and filename in database if enabled for tabular files
        await self.store_file_hash_in_db(
            file_id, file_hash, original_filename, None  # Use configurable default
        )

        return {
            "file_id": file_id,
            "is_image": False,
            "is_tabular": True,
            "message": "File processed and ready for querying.",
            "status": "success",
            "temp_file_path": temp_file_path,
            "metadata": metadata,  # Include file-specific metadata
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
            # # if os.path.exists(encrypted_db_path):
            # #     decrypt_file(encrypted_db_path, db_path)
            # #     os.remove(encrypted_db_path)  # Clean up encrypted file
            # # else:
            # #     logging.error(
            # #         f"Encrypted database not found for file_id: {existing_file_id}"
            # #     )

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
                                text(f'SELECT * FROM "{table_name}" LIMIT 2')
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
                        "tables": [],
                    }
                    for t in table_info:
                        database_summary["tables"].append(
                            {
                                "name": t["name"],
                                "columns": t["columns"],
                                "row_count": t["row_count"],
                                "top_rows": t[
                                    "sample_data"
                                ],  # Use "top_rows" to match csv_handler format
                                "column_stats": t["column_stats"],
                            }
                        )

                    # Validate that the database has valid tables with data
                    if len(table_info) == 0:
                        logging.error(
                            f"No tables found in the existing database for file_id: {existing_file_id}"
                        )
                        raise ValueError(
                            f"No tables found in the existing database for file_id: {existing_file_id}. "
                            f"The database may be corrupted."
                        )

                    # Check if all tables are empty
                    total_rows = sum(table["row_count"] for table in table_info)
                    if total_rows == 0:
                        logging.error(
                            f"All tables are empty in the existing database for file_id: {existing_file_id}"
                        )
                        raise ValueError(
                            f"All tables are empty in the existing database for file_id: {existing_file_id}. "
                            f"Please check if the database contains valid data."
                        )

                    # Update file_info.json with database_summary
                    update_data = {"database_summary": database_summary}

                    self.gcs_handler.update_file_info(existing_file_id, update_data)
                    logging.info(
                        f"Added database_summary to existing file_info.json with {len(table_info)} tables "
                        f"and {total_rows} total rows"
                    )

                except Exception as e:
                    logging.error(
                        f"Failed to extract database_summary for existing file: {str(e)}",
                        exc_info=True,
                    )
                    # Re-raise validation errors for corrupted databases
                    if isinstance(e, ValueError):
                        raise
                    else:
                        # Log other errors but don't fail the entire process for existing files
                        # as they may have been processed with older versions
                        logging.warning(
                            f"Could not extract database summary for existing file {existing_file_id}, "
                            f"continuing anyway"
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
        google_result,  # Kept for backwards compatibility
        azure_result,
        actual_file_id,
        temp_file_path,
        metadata,
    ):
        """Process image analysis for new or incomplete embeddings.

        Parameters:
        -----------
        google_result : dict
            Kept for backwards compatibility but not used in the unified Azure approach
        """
        # Add debugging logs
        logging.info(
            f"_process_image_analysis called with: is_image={is_image}, "
            f"existing_file_id={existing_file_id}, azure_result={azure_result}"
        )

        # Skip image analysis if:
        #   • file is not an image, or
        #   • we already have Azure embeddings for this image (unified embedding approach)
        embeddings_exist = (
            azure_result.get("embeddings_exist", False) if azure_result else False
        )
        should_skip = not is_image or (existing_file_id and embeddings_exist)

        logging.info(
            f"Image analysis decision: should_skip={should_skip} "
            f"(not is_image: {not is_image}, "
            f"existing_file_id and embeddings_exist: {existing_file_id and embeddings_exist})"
        )

        if should_skip:
            logging.info(
                f"Skipping image analysis: is_image={is_image}, "
                f"existing_file_id={existing_file_id}, embeddings_exist={embeddings_exist}"
            )
            return []

        logging.info(
            f"Proceeding with image analysis for file_id: {actual_file_id}, "
            f"temp_file_path: {temp_file_path}"
        )
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
        # With unified Azure embeddings used for both Azure GPT and Gemini models,
        # we only need to verify the Azure directory. Requiring the legacy
        # `google` directory falsely flags missing embeddings and triggers an
        # unnecessary GCS download.
        sqlite_path = os.path.join(azure_path, "chroma.sqlite3")
        return os.path.exists(sqlite_path)

    async def _handle_existing_file_encryption(
        self,
        encrypted_file_path,
        existing_file_id,
        original_filename,
        temp_file_path,
        is_tabular,
        is_database,
    ):
        """Check and handle encryption for existing files using the FileEncryptionManager."""
        # Delegate the encryption logic to our FileEncryptionManager
        await self.encryption_manager.ensure_file_encryption(
            existing_file_id, original_filename, temp_file_path, is_tabular, is_database
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
        google_result,  # Kept for backwards compatibility
        azure_result,
    ):
        """Handle processing for files that already exist in the system.

        Parameters:
        -----------
        google_result : dict
            Kept for backwards compatibility but not used in the unified Azure embedding approach
        azure_result : dict
            Contains information about Azure embeddings existence
        """
        # Early return if the file doesn't exist
        if not existing_file_id:
            return None

        logging.info(f"Found embeddings for: {original_filename}")

        # Update username list using the more comprehensive method
        # Use update_file_info which appends usernames (tracks frequency)
        # instead of update_username_list which deduplicates
        self.gcs_handler.update_file_info(existing_file_id, {"username": username})
        logging.info(
            f"Appended username '{username}' for existing file (tracks upload frequency)"
        )

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
        logging.info(f"Local embeddings exist: {local_exists}")

        # Download embeddings if they exist remotely but not locally
        # Download embeddings from GCS only if Azure embeddings exist remotely
        # but the local cache is missing.
        if azure_result["embeddings_exist"] and not local_exists:
            self.gcs_handler.download_files_from_folder_by_id(existing_file_id)

        # For images, we only need the embeddings to chat
        if is_image:
            # Get existing file metadata to ensure we have correct hash and other properties
            file_metadata = self.gcs_handler.get_file_info(existing_file_id) or {}

            # Ensure username is included
            if (
                "username" in file_metadata
                and username not in file_metadata["username"]
            ):
                file_metadata["username"].append(username)

            return {
                "file_id": existing_file_id,
                "is_image": is_image,
                "is_tabular": is_tabular,
                "message": "File already exists and has embeddings.",
                "status": "existing",
                "temp_file_path": temp_file_path,  # Keep original temp file for reference
                "metadata": file_metadata,  # Include file-specific metadata
            }

        # Copy the temp file to match the existing file ID path
        existing_temp_path = f"local_data/{existing_file_id}_{original_filename}"
        os.makedirs(os.path.dirname(existing_temp_path), exist_ok=True)
        shutil.copy2(temp_file_path, existing_temp_path)
        temp_file_path = existing_temp_path

        # Get existing file metadata to ensure we have correct hash and other properties
        file_metadata = self.gcs_handler.get_file_info(existing_file_id) or {}

        # Ensure username is included
        if "username" in file_metadata and username not in file_metadata["username"]:
            file_metadata["username"].append(username)

        return {
            "file_id": existing_file_id,
            "is_image": is_image,
            "is_tabular": is_tabular,
            "message": "File already exists. Processing database."
            if is_tabular
            else "File already exists and has embeddings.",
            "status": "existing",
            "temp_file_path": temp_file_path,
            "metadata": file_metadata,  # Include file-specific metadata
        }

    async def _cleanup_empty_directory(self, file_id: str):
        """Clean up empty directory that was created in error for existing files."""
        try:
            chroma_dir = f"./chroma_db/{file_id}"
            if os.path.exists(chroma_dir):
                # Check if directory is empty
                if not os.listdir(chroma_dir):
                    os.rmdir(chroma_dir)
                    logging.info(f"Cleaned up empty directory: {chroma_dir}")
                else:
                    logging.info(
                        f"Directory {chroma_dir} is not empty, skipping cleanup"
                    )
        except Exception as e:
            logging.warning(f"Error cleaning up directory {file_id}: {str(e)}")

    async def process_file(
        self, file: UploadFile, file_id: str, is_image: bool, username: str
    ) -> dict:
        """Process uploaded file including handling images, tabular data and existing files."""
        try:
            # Sanitize filename and read file content
            original_filename = self._sanitize_filename(file.filename)
            (
                file_content,
                file_hash,
                file_types,
            ) = await self._read_and_process_file_content(file, original_filename)
            is_tabular, is_database, is_text = file_types

            # Check for existing file
            (
                existing_file_id,
                azure_result,
            ) = await self._check_for_existing_file(
                file_hash, original_filename, file_id, is_tabular, is_database
            )
            # For compatibility with downstream code expecting google_result
            google_result = {
                "embeddings_exist": False
            }  # Always false with unified Azure approach

            # Check if we have an existing file with embeddings
            embeddings_exist = existing_file_id and azure_result.get(
                "embeddings_exist", False
            )

            # If we have an existing file with embeddings, handle it immediately without creating new directories
            if embeddings_exist:
                logging.info(f"Found embeddings for: {original_filename}")

                # Clean up any empty directory that might have been created for the new file_id
                await self._cleanup_empty_directory(file_id)

                # Download embeddings if they exist remotely but not locally
                # This ensures embeddings are available for immediate chat use
                local_exists = await self._check_local_embeddings(existing_file_id)
                if not local_exists:
                    logging.info(f"Downloading embeddings for file {existing_file_id}")
                    self.gcs_handler.download_files_from_folder_by_id(existing_file_id)

                # Update username list using the more comprehensive method
                self.gcs_handler.update_file_info(
                    existing_file_id, {"username": username}
                )
                logging.info(
                    f"Appended username '{username}' for existing file (tracks upload frequency)"
                )

                # Return early with existing file information
                return {
                    "status": "success",
                    "file_id": existing_file_id,  # Use existing file ID
                    "is_image": is_image,
                    "embeddings_exist": True,
                    "temp_file_path": None,
                }

            # Only create directories and save file if we don't have existing embeddings
            # Create directories and save file
            temp_file_path = await self._save_file_locally(
                file_id, original_filename, file_content
            )
            del file_content  # Free memory

            # Validate text content for document files before encryption
            file_extension = os.path.splitext(original_filename)[1].lower()
            if file_extension in [".pdf", ".doc", ".docx", ".txt"] and not is_image:
                try:
                    # Create a base handler for text extraction
                    base_handler = BaseRAGHandler(self.configs, self.gcs_handler)
                    extracted_text = base_handler.extract_text_from_file(temp_file_path)

                    # Check if extraction returned an error
                    if extracted_text.startswith("ERROR:"):
                        # Clean up temp file and directory before returning error
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                        await self._cleanup_empty_directory(file_id)

                        error_msg = extracted_text[7:]  # Remove "ERROR: " prefix
                        logging.error(
                            f"Text extraction failed for {original_filename}: {error_msg}"
                        )
                        return {
                            "status": "error",
                            "message": error_msg,
                            "file_id": file_id,
                            "is_image": is_image,
                            "embeddings_exist": False,
                            "temp_file_path": None,
                        }

                    # Calculate word count for extracted text
                    word_count = len(extracted_text.split())
                    logging.info(
                        f"Text extraction completed for {original_filename}: "
                        f"{len(extracted_text)} characters, {word_count} words"
                    )

                    # Store extracted text in a temporary variable for metadata
                    self.extracted_text_cache = extracted_text
                    # Store extracted text for embedding handler to avoid duplicate extraction
                    self.store_extracted_text(file_id, extracted_text)

                    # Validate text length for document files (minimum 100 characters)
                    cleaned_text = extracted_text.strip()
                    if len(cleaned_text) < 100:
                        # Clean up temp file and directory before returning error
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                        await self._cleanup_empty_directory(file_id)

                        error_msg = (
                            f"Insufficient text content extracted from {file_extension} file. "
                            f"Only {len(cleaned_text)} characters found (minimum 100 required). "
                            f"The document may be empty, corrupted, or contain only images/non-text content."
                        )
                        logging.error(
                            f"Text validation failed for {original_filename}: {error_msg}"
                        )
                        return {
                            "status": "error",
                            "message": error_msg,
                            "file_id": file_id,
                            "is_image": is_image,
                            "embeddings_exist": False,
                            "temp_file_path": None,
                        }

                    logging.info(
                        f"Text validation passed for {original_filename} ({len(extracted_text)} characters)"
                    )

                except Exception as e:
                    # Clean up temp file and directory before returning error
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    await self._cleanup_empty_directory(file_id)

                    error_msg = f"Failed to validate text content: {str(e)}"
                    logging.error(
                        f"Text validation failed for {original_filename}: {error_msg}"
                    )
                    return {
                        "status": "error",
                        "message": error_msg,
                        "file_id": file_id,
                        "is_image": is_image,
                        "embeddings_exist": False,
                        "temp_file_path": None,
                    }

            # Handle file encryption
            encrypted_file_path = await self._handle_file_encryption(
                is_tabular,
                is_database,
                existing_file_id,
                temp_file_path,
                original_filename,
                file_id,
            )

            # Determine final file_id and prepare metadata
            actual_file_id = existing_file_id if existing_file_id else file_id
            metadata = self._prepare_file_metadata(
                is_image,
                is_tabular,
                is_database,
                file_hash,
                username,
                original_filename,
                actual_file_id,
            )

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

            # Update metadata with encryption status
            metadata["is_encrypted"] = encrypted_file_path is not None
            self.gcs_handler.temp_metadata = metadata.copy()

            # Handle tabular data for new files
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

            # Handle potential file conflicts
            (
                actual_file_id,
                encrypted_file_path,
                metadata,
            ) = await self._handle_file_conflicts(
                is_tabular,
                encrypted_file_path,
                actual_file_id,
                original_filename,
                temp_file_path,
                metadata,
            )

            # Return final response
            return self._prepare_file_success_response(
                actual_file_id,
                is_image,
                is_tabular,
                temp_file_path,
                analysis_files,
                metadata,
            )

        except Exception as e:
            logging.error(f"Exception in process_file: {str(e)}")
            await self._cleanup_on_error(locals())
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

    async def _read_and_process_file_content(
        self, file: UploadFile, original_filename: str
    ):
        """Read file content and process it to get hash and file types"""
        file_content = await file.read()
        file_hash = self.calculate_file_hash(file_content)
        file_types = await self._determine_file_types(original_filename)
        return file_content, file_hash, file_types

    async def _check_for_existing_file(
        self,
        file_hash: str,
        original_filename: str,
        file_id: str,
        is_tabular: bool = False,
        is_database: bool = False,
    ):
        """Check if a file with the same hash already exists and verify it's not a hash collision"""
        azure_result = {"embeddings_exist": False}

        existing_file_id, _ = await self.find_existing_file_by_hash_async(file_hash)
        if not existing_file_id:
            return None, azure_result

        # Get embedding status for existing file
        embedding_handler = EmbeddingHandler(self.configs, self.gcs_handler)
        # Use only Azure embeddings for unified approach
        azure_result = await embedding_handler.check_embeddings_exist(
            existing_file_id, "gpt_4o_mini"
        )
        logging.info(f"Existing file found with hash: {existing_file_id}")

        return existing_file_id, azure_result

    async def _save_file_locally(
        self, file_id: str, original_filename: str, file_content: bytes
    ):
        """Save file content to local storage"""
        await self._prepare_file_directories(file_id, None)
        temp_file_path = f"local_data/{file_id}_{original_filename}"
        async with aiofiles.open(temp_file_path, "wb") as buffer:
            await buffer.write(file_content)
        return temp_file_path

    async def _handle_file_encryption(
        self,
        is_tabular: bool,
        is_database: bool,
        existing_file_id: str,
        temp_file_path: str,
        original_filename: str,
        file_id: str,
    ):
        """Handle encryption of the file if needed"""
        # Skip encryption if we have an existing file with embeddings
        if existing_file_id:
            return None

        # Encrypt the original uploaded file (tabular files are handled separately)
        if not is_tabular and not is_database:
            # Check if the temp_file_path actually exists (it might not if we skipped file saving)
            if os.path.exists(temp_file_path):
                # Use the FileEncryptionManager to handle encryption
                (
                    success,
                    encrypted_file_path,
                ) = await self.encryption_manager.encrypt_and_upload(
                    temp_file_path, original_filename, file_id
                )
                return encrypted_file_path

        return None

    def _prepare_file_metadata(
        self,
        is_image: bool,
        is_tabular: bool,
        is_database: bool,
        file_hash: str,
        username: str,
        original_filename: str,
        file_id: str,
    ):
        """Prepare metadata for the processed file"""
        metadata = {
            "is_image": is_image,
            "is_tabular": is_tabular or is_database,
            "file_hash": file_hash,
            "username": [username],  # Store username as an array
            "original_filename": original_filename,
            "file_id": file_id,  # Always use the actual file_id (existing or new)
            "migrated": False,  # New files are not migrated
        }

        # Add extracted text and word count if available and feature is enabled
        if (
            hasattr(self, "extracted_text_cache")
            and self.extracted_text_cache
            and getattr(self.configs, "save_extracted_text_in_metadata", False)
        ):
            metadata["extracted_text"] = self.extracted_text_cache
            metadata["word_count"] = len(self.extracted_text_cache.split())
            logging.info(
                "Saved extracted text in metadata to avoid duplicate extraction"
            )
            # Clear the cache after use
            self.extracted_text_cache = None
        elif hasattr(self, "extracted_text_cache") and self.extracted_text_cache:
            # Just save word count, not the full text
            metadata["word_count"] = len(self.extracted_text_cache.split())
            # Clear the cache after use
            self.extracted_text_cache = None

        return metadata

    async def _handle_file_conflicts(
        self,
        is_tabular: bool,
        encrypted_file_path,
        actual_file_id: str,
        original_filename: str,
        temp_file_path: str,
        metadata: dict,
    ):
        """Check for and handle potential file conflicts"""
        # Tabular files don't use this conflict detection since they're always saved as tabular_data.db.encrypted
        if is_tabular or not encrypted_file_path:
            return actual_file_id, encrypted_file_path, metadata

        # For non-tabular files, check if there's already a different file in this directory
        prefix = f"{self.configs.gcp_resource.gcp_embeddings_folder}/{actual_file_id}/"
        has_conflict = False

        blobs = await asyncio.to_thread(
            list, self.gcs_handler.bucket.list_blobs(prefix=prefix)
        )

        for blob in blobs:
            if blob.name.endswith(".encrypted") and not blob.name.endswith(
                f"{original_filename}.encrypted"
            ):
                has_conflict = True
                logging.warning(
                    f"Detected potential conflict: {blob.name} vs {original_filename}.encrypted"
                )
                break

        if has_conflict:
            # Generate a new file_id to avoid conflicts
            new_file_id = str(uuid.uuid4())
            logging.info(
                f"Avoiding conflict by using new file_id: {new_file_id} instead of {actual_file_id}"
            )
            actual_file_id = new_file_id
            # Update metadata with new file_id
            metadata["file_id"] = actual_file_id

            # Re-encrypt and upload with the new file_id
            (
                success,
                new_encrypted_file_path,
            ) = await self.encryption_manager.encrypt_and_upload(
                temp_file_path, original_filename, actual_file_id
            )
            if (
                new_encrypted_file_path
                and new_encrypted_file_path != encrypted_file_path
            ):
                encrypted_file_path = new_encrypted_file_path

        return actual_file_id, encrypted_file_path, metadata

    def _prepare_file_success_response(
        self,
        actual_file_id: str,
        is_image: bool,
        is_tabular: bool,
        temp_file_path: str,
        analysis_files: list,
        metadata: dict,
    ):
        """Prepare the success response for process_file"""
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
            "metadata": metadata,  # Include metadata explicitly with the result
        }

    async def _cleanup_on_error(self, context):
        """Clean up temporary files in case of an error"""
        if "temp_file_path" in context and os.path.exists(context["temp_file_path"]):
            await asyncio.to_thread(os.remove, context["temp_file_path"])

        if (
            "encrypted_file_path" in context
            and context["encrypted_file_path"]
            and os.path.exists(context["encrypted_file_path"])
        ):
            await asyncio.to_thread(os.remove, context["encrypted_file_path"])

        if (
            "analysis_files" in context
            and context["analysis_files"]
            and all(os.path.exists(path) for path in context["analysis_files"])
        ):
            await asyncio.to_thread(os.remove, context["analysis_files"][0])
            await asyncio.to_thread(os.remove, context["analysis_files"][1])

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
        if self.use_file_hash_db:
            try:
                # Use the database lookup function
                from rtl_rag_chatbot_api.app import get_db_session
                from rtl_rag_chatbot_api.common.db import find_file_by_hash_db

                with get_db_session() as db_session:
                    result = find_file_by_hash_db(db_session, file_hash)
                    if result:
                        file_id, embedding_type = result
                        logging.info(f"File found in database with ID: {file_id}")
                        return file_id, embedding_type
                    else:
                        logging.info(
                            f"No file found in database with hash: {file_hash}"
                        )
                        return None, None

            except Exception as db_e:
                logging.error(
                    f"Error during file_hash_db lookup: {str(db_e)}", exc_info=True
                )
                return None, None
        else:
            # Fallback to GCS lookup if not using file_hash_db
            return await asyncio.to_thread(
                self.gcs_handler.find_existing_file_by_hash, file_hash
            )

    async def process_single_url(
        self,
        url: str,
        username: str,
        background_tasks,
        embedding_handler,
        custom_file_id: str = None,
    ) -> dict:
        """Process a single URL and create embeddings for it.

        Args:
            url: The URL to process
            username: The username for this request
            background_tasks: FastAPI BackgroundTasks for async operations
            embedding_handler: Handler for creating embeddings

        Returns:
            Dict containing file_id and status information
        """
        # Use provided custom_file_id or generate a new unique ID for this URL
        url_file_id = custom_file_id if custom_file_id else str(uuid.uuid4())
        file_name = f"{url_file_id}_url_content.txt"
        temp_file_path = f"local_data/{file_name}"
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

        # Import website handler here to avoid circular imports
        from rtl_rag_chatbot_api.chatbot.website_handler import WebsiteHandler

        website_handler = WebsiteHandler()

        try:
            logging.info(
                f"Starting process_single_url for {url} with file_id {url_file_id}"
            )

            # Extract content from the URL
            (
                content,
                title,
                is_successful,
            ) = website_handler.extract_content_from_single_url(url)
            if not is_successful or not content:
                logging.warning(f"Failed to extract content from URL: {url}")
                return {
                    "file_id": None,
                    "status": "error",
                    "message": f"Could not extract content from URL: {url}",
                    "is_image": False,
                    "is_tabular": False,
                    "url": url,
                    "temp_file_path": None,
                }

            # Check content quality
            is_substantive, word_count = website_handler.check_content_quality(content)
            logging.info(
                f"Content quality check for {url}: substantive={is_substantive}, word_count={word_count}"
            )

            if not is_substantive:
                return {
                    "file_id": None,
                    "status": "error",
                    "message": (
                        f"Die Verarbeitung von {url} ist fehlgeschlagen. "
                        "Bitte versuchen Sie es erneut mit einer anderen Domain/einer anderen URL."
                    ),
                    "is_image": False,
                    "is_tabular": False,
                    "url": url,
                    "temp_file_path": None,
                }

            try:
                # Write content to file
                logging.info(f"Writing content to temporary file: {temp_file_path}")
                async with aiofiles.open(temp_file_path, "w", encoding="utf-8") as f:
                    await f.write(content)

                # Calculate hash and check existing embeddings
                content_hash = self.calculate_file_hash(content.encode("utf-8"))
                logging.info(f"Calculated hash for {url}: {content_hash[:16]}...")

                existing_file_id, _ = await self.find_existing_file_by_hash_async(
                    content_hash
                )

                if existing_file_id:
                    logging.info(
                        f"Found existing content with ID {existing_file_id} for URL {url}, reusing it"
                    )
                    # Update the file info with the new username
                    self.gcs_handler.update_file_info(
                        existing_file_id, {"username": username}
                    )
                    # Clean up the temporary file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                        logging.info(
                            f"Removed temporary file {temp_file_path} as content already exists"
                        )
                    return {
                        "file_id": existing_file_id,
                        "status": "existing",
                        "message": f"Content from URL {url} already processed, reusing existing file",
                        "is_image": False,
                        "is_tabular": False,
                        "url": url,
                        "temp_file_path": None,
                    }

                # Create metadata for the file
                metadata = {
                    "file_hash": content_hash,
                    "original_filename": title if title else "url_content.txt",
                    "username": [username],
                    "is_url": True,
                    "url": url,
                    "word_count": word_count,
                    "file_id": url_file_id,
                }

                # Save metadata
                logging.info(
                    f"Saving metadata for URL {url} with file_id {url_file_id}"
                )
                # Update GCS metadata but also prepare it directly
                # This ensures file-specific metadata is passed through the chain
                # without relying on shared state in self.gcs_handler.temp_metadata
                self.gcs_handler.temp_metadata = metadata
                # The initial file_info.json will be created in the background task,
                # consistent with PDF/document handling, to avoid redundant updates.
                # self.gcs_handler.update_file_info(url_file_id, metadata)

                # Import here to avoid circular imports
                from rtl_rag_chatbot_api.app import SessionLocal
                from rtl_rag_chatbot_api.chatbot.parallel_embedding_creator import (
                    create_embeddings_parallel,
                )

                # Wait for embedding creation to complete before returning response
                # This matches the behavior of multi-file uploads
                logging.info(
                    f"Creating embeddings for URL {url} with file_id {url_file_id}"
                )
                await create_embeddings_parallel(
                    file_ids=[url_file_id],
                    file_paths=[temp_file_path],
                    embedding_handler=embedding_handler,
                    configs=self.configs,
                    session_local=SessionLocal,
                    background_tasks=background_tasks,
                    username_lists=[
                        None
                    ],  # Username is in the metadata, no separate update needed
                    file_metadata_list=[metadata],
                    max_concurrent_tasks=1,
                )
                logging.info(
                    f"Completed embedding creation for URL {url} with file_id {url_file_id}"
                )

                return {
                    "file_id": url_file_id,
                    "status": "success",
                    "message": f"URL {url} processed successfully",
                    "is_image": False,
                    "is_tabular": False,
                    "url": url,
                    "temp_file_path": temp_file_path,
                    "word_count": word_count,
                    "title": title,
                }
            except Exception as file_error:
                logging.error(
                    f"Error during file operations for URL {url}: {str(file_error)}"
                )
                # Clean up temp file if it exists
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        logging.info(
                            f"Cleaned up temporary file {temp_file_path} after error"
                        )
                    except Exception as cleanup_error:
                        logging.error(
                            f"Failed to clean up temporary file {temp_file_path}: {str(cleanup_error)}"
                        )

                raise file_error  # Re-raise to be caught by the outer exception handler

        except Exception as e:
            logging.error(f"Error processing URL {url}: {str(e)}")
            import traceback

            logging.error(f"Traceback for URL {url}: {traceback.format_exc()}")
            # Clean up temp file if it exists
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass  # Already logging the main error, don't obscure it
            return {
                "file_id": None,
                "status": "error",
                "message": f"Error processing URL {url}: {str(e)}",
                "is_image": False,
                "is_tabular": False,
                "url": url,
                "temp_file_path": None,
            }

    # Helper methods for process_urls
    def _parse_url_list(self, urls_text: str) -> list[str]:
        """Parses a string of URLs (separated by newlines or commas) into a list."""
        if "\n" in urls_text:
            url_list = [url.strip() for url in urls_text.split("\n") if url.strip()]
        else:
            url_list = [url.strip() for url in urls_text.split(",") if url.strip()]
        return url_list

    def _cleanup_temp_files_from_results(self, results: list[dict]):
        """Cleans up temporary files from a list of URL processing results."""
        for result_item in results:
            if result_item.get("status") in ["success", "existing"] and result_item.get(
                "temp_file_path"
            ):
                temp_path = result_item.get("temp_file_path")
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                        logging.info(
                            f"Cleaned up temporary file {temp_path} after batch processing event"
                        )
                    except Exception as cleanup_error:
                        logging.error(
                            f"Failed to clean up temporary file {temp_path}: {str(cleanup_error)}"
                        )

    def _handle_batch_error(
        self, failed_url: str, error_message: str, current_results: list[dict]
    ) -> dict:
        """Handles errors during batch URL processing, cleans up, and returns error response."""
        logging.warning(
            f"Failing entire URL batch due to error with URL {failed_url}: {error_message}"
        )
        self._cleanup_temp_files_from_results(current_results)
        return {
            "status": "error",
            "message": error_message,
            "url_results": current_results,
            "is_image": False,
            "is_tabular": False,
        }

    def _prepare_success_response(
        self, url_list: list[str], file_ids: list[str], results: list[dict]
    ) -> dict:
        """Constructs the success response for process_urls."""
        # Optional: self._cleanup_temp_files_from_results(results) # If cleanup on success is also desired
        return {
            "status": "success",
            "message": f"Processed all {len(url_list)} URLs successfully",
            "file_id": file_ids[0] if file_ids else None,  # For backward compatibility
            "file_ids": file_ids,  # All file IDs for multi-document chat
            "url_results": results,
            "is_image": False,
            "is_tabular": False,
            "original_filename": "url_content.txt",
            "multi_file_mode": len(file_ids) > 1,
        }

    async def prepare_db_from_file(self, file_path: str, file_id: str, username: str):
        """
        Prepare SQLite database from a tabular file (CSV, Excel) and upload to GCS.

        This method is called from app.py's process_tabular_file function as a background task.
        It wraps the _handle_new_tabular_data method which does the actual work.

        Args:
            file_path: Path to the tabular file
            file_id: Unique identifier for the file
            username: Username for file ownership

        Returns:
            Dict containing processing status and file information
        """
        try:
            logging.info(
                f"Preparing database from file: {file_path} with ID: {file_id}"
            )

            # Calculate file hash
            with open(file_path, "rb") as f:
                file_content = f.read()
                file_hash = self.calculate_file_hash(file_content)

            # Get original filename from the path
            original_filename = os.path.basename(file_path)
            if file_id in original_filename:
                # Remove file_id prefix if present
                original_filename = original_filename.replace(f"{file_id}_", "")

            # Process the tabular file using the existing method
            result = await self._handle_new_tabular_data(
                file_id=file_id,
                temp_file_path=file_path,
                file_hash=file_hash,
                original_filename=original_filename,
                username=username,
            )

            logging.info(
                f"Successfully prepared database for {original_filename} with ID {file_id}"
            )
            return result
        except Exception as e:
            logging.error(
                f"Error preparing database from file {file_path}: {str(e)}",
                exc_info=True,
            )
            return {
                "file_id": file_id,
                "is_image": False,
                "is_tabular": True,
                "message": f"Error preparing database: {str(e)}",
                "status": "error",
                "temp_file_path": file_path,
            }

    # chat with URL is deprecated, but keeping it for backward compatibility
    async def process_urls(
        self, urls_text, username, temp_file_id, background_tasks, embedding_handler
    ):
        """Process multiple URLs and extract content from each in parallel.

        Args:
            urls_text: Text containing one or more URLs separated by commas or newlines
            username: Username processing the URLs
            temp_file_id: Temporary file ID (used as base for URL processing)
            background_tasks: FastAPI background tasks object
            embedding_handler: Handler for creating embeddings

        Returns:
            Dict containing status and results for all URLs processed in parallel
        """
        logging.info(f"Parsing URL input: {urls_text[:100]}...")
        url_list = self._parse_url_list(urls_text)

        if not url_list:
            logging.warning("No valid URLs were provided in the input")
            return {"status": "error", "message": "No valid URLs provided"}

        logging.info(
            f"Processing {len(url_list)} URLs in parallel with IDs: {temp_file_id}_*"
        )
        # Log detailed URL list at debug level
        logging.debug(f"URLs to process: {', '.join(url_list)}")
        start_time = time.time()

        # Create tasks for processing each URL concurrently
        tasks = []
        for url in url_list:
            # Generate a completely unique ID for each URL, just like with file uploads
            # This makes URL handling consistent with PDF/file handling
            url_specific_id = str(uuid.uuid4())
            tasks.append(
                self.process_single_url(
                    url, username, background_tasks, embedding_handler, url_specific_id
                )
            )

        # Process all URLs in parallel
        try:
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and handle any exceptions
            processed_results = []
            file_ids = []
            error_found = False

            for i, result in enumerate(results):
                url = url_list[i] if i < len(url_list) else "unknown"

                # Check if this result is an exception
                if isinstance(result, Exception):
                    logging.error(f"Error processing URL {url}: {str(result)}")
                    error_result = {
                        "file_id": None,
                        "status": "error",
                        "message": f"Error processing URL: {str(result)}",
                        "url": url,
                        "is_image": False,
                        "is_tabular": False,
                        "temp_file_path": None,
                    }
                    processed_results.append(error_result)
                    error_found = True
                    continue

                # Process successful result
                processed_results.append(result)
                logging.info(
                    f"URL processing result for {url}: {result.get('status', 'unknown')}"
                )

                # Check for error status in result
                if result.get("status") == "error":
                    error_found = True
                elif result.get("status") in ["success", "existing"] and result.get(
                    "file_id"
                ):
                    file_ids.append(result["file_id"])

            # If any errors were found, include them in the response but don't fail the whole batch
            if error_found and not file_ids:
                # All URLs failed - return error
                return self._handle_batch_error(
                    "multiple URLs", "Failed to process all URLs", processed_results
                )

            # At least some URLs succeeded

            # Calculate elapsed time for all URL processing
            elapsed = time.time() - start_time
            logging.info(f"Completed processing {len(url_list)} URLs in {elapsed:.2f}s")

            # If loop completes, all URLs were processed successfully
            return self._prepare_success_response(url_list, file_ids, processed_results)

        except Exception as e:
            # This outer try-except catches unexpected errors not tied to a specific URL processing step
            logging.error(f"Unexpected error in process_urls batch operation: {str(e)}")
            # Attempt cleanup if results list exists
            if "results" in locals():
                self._cleanup_temp_files_from_results(results)
            return {
                "status": "error",
                "message": f"An unexpected error occurred during batch URL processing: {str(e)}",
                "is_image": False,
                "is_tabular": False,
            }

    async def cleanup_temp_files(self, temp_file_paths):
        """Clean up temporary files after processing.

        Args:
            temp_file_paths: A list of temporary file paths to clean up

        Returns:
            None
        """
        for path in temp_file_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logging.info(f"Cleaned up temporary file: {path}")
                except Exception as e:
                    logging.error(f"Error cleaning up temporary file {path}: {str(e)}")

    def get_file_type(self, file_name):
        """Determine the file type from the file name.

        Args:
            file_name: Name of the file to check

        Returns:
            str: The file type ('pdf', 'image', 'tabular', etc.)
        """
        ext = os.path.splitext(file_name)[1].lower()

        if ext in [".pdf"]:
            return "pdf"
        elif ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
            return "image"
        elif ext in [".csv", ".xlsx", ".xls", ".db", ".sqlite"]:
            return "tabular"
        else:
            return "other"
