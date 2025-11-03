import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Import removed - gcs_handler should be passed as parameter to EmbeddingHandler constructor
from rtl_rag_chatbot_api.chatbot.chatbot_creator import AzureChatbot
from rtl_rag_chatbot_api.common.base_handler import BaseRAGHandler
from rtl_rag_chatbot_api.common.chroma_manager import ChromaDBManager
from rtl_rag_chatbot_api.common.errors import EmbeddingCreationError

logging.basicConfig(level=logging.INFO)


class EmbeddingHandler:
    """
    Handles the creation, storage, and uploading of embeddings for Azure and Gemini models.

    Attributes:
        configs: Configuration object containing necessary settings.
        gcs_handler: Handler for Google Cloud Storage operations.
    """

    def __init__(self, configs, gcs_handler, file_handler=None):
        self.configs = configs
        self.gcs_handler = gcs_handler
        self.file_handler = file_handler
        self.chroma_manager = ChromaDBManager()

    def create_base_handler(self):
        """
        Create a BaseRAGHandler instance for text extraction and processing.

        Returns:
            BaseRAGHandler: A handler for RAG operations
        """
        return BaseRAGHandler(self.configs, self.gcs_handler)

    def _prepare_file_info(
        self,
        file_id: str,
        azure_result: Any,
        gemini_result: Any,
        file_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Prepare file info dictionary with embeddings metadata.

        Args:
            file_id: Unique identifier for the file
            azure_result: Result of Azure embedding creation (can be dict or other type)
            gemini_result: Result of Gemini embedding creation (can be dict or other type)
            file_metadata: Additional metadata for the file

        Returns:
            Dict containing file metadata and embedding status
        """
        # Log what we received to help debug
        logging.info(
            f"Preparing file_info for {file_id} with azure_result type: {type(azure_result)},"
            f" gemini_result type: {type(gemini_result)}"
        )

        # Get metadata file_id if it exists to ensure consistency
        metadata_file_id = file_metadata.get("file_id") if file_metadata else None

        # If metadata has a file_id that differs from the passed file_id, log a warning but USE THE PASSED file_id
        if metadata_file_id and metadata_file_id != file_id:
            logging.warning(
                f"Metadata file_id {metadata_file_id} differs from passed file_id "
                f"{file_id}. Using passed file_id to avoid mix-ups."
            )

        # Ensure azure_result and gemini_result are dictionaries
        # If they're not, convert them to dictionaries with appropriate fields
        if not isinstance(azure_result, dict):
            azure_result = {
                "success": False,
                "error": str(azure_result) if azure_result else "Unknown error",
            }

        if not isinstance(gemini_result, dict):
            gemini_result = {
                "success": False,
                "error": str(gemini_result) if gemini_result else "Unknown error",
            }

        # Do an additional check to fix any incorrect values in the result dicts
        if azure_result.get("error") == "completed":
            azure_result["success"] = True
            azure_result["error"] = None

        if gemini_result.get("error") == "completed":
            gemini_result["success"] = True
            gemini_result["error"] = None

        # Create file_info with embeddings metadata
        file_info = {
            **(file_metadata or {}),  # Include original metadata
            "embeddings": {
                "azure": azure_result,
                # Keep gemini key for backward compatibility, but we use Azure embeddings for all models now
                "gemini": {
                    "success": False,
                    "error": "Deprecated. Using unified Azure embeddings",
                },
            },
            # Set status to ready_for_chat for local embeddings, will be updated to completed after GCS upload
            "embeddings_status": "ready_for_chat",
            "azure_ready": azure_result.get("success", False),
            # With unified approach, embeddings_ready depends only on Azure embeddings
            "embeddings_ready": azure_result.get("success", False),
            "file_id": file_id,  # Ensure file_id consistency
            "embeddings_created_at": datetime.now().isoformat(),  # Track when embeddings were created
            "embedding_type": self.configs.chatbot.default_embedding_type,  # Use configurable default embedding type
        }

        # Ensure critical fields are present
        if (
            "file_hash" not in file_info
            and file_metadata
            and "file_hash" in file_metadata
        ):
            file_info["file_hash"] = file_metadata["file_hash"]

        # Preserve original_filename if available
        if (
            "original_filename" not in file_info
            and file_metadata
            and "original_filename" in file_metadata
        ):
            file_info["original_filename"] = file_metadata["original_filename"]

        return file_info

    def embeddings_exist(self, file_id: str) -> tuple[bool, bool, bool]:
        """
        Check if Azure embeddings exist and are valid both in GCS and locally.
        With the unified embedding approach, we only need to check for Azure embeddings.

        Returns tuple of (gcs_status, local_status, all_valid)
        """
        try:
            # Check if file is ready for chat by checking embeddings_status in file_info.json
            local_file_info_path = os.path.join(
                "./chroma_db", file_id, "file_info.json"
            )
            if os.path.exists(local_file_info_path):
                try:
                    with open(local_file_info_path, "r") as f:
                        file_info = json.load(f)

                    # Check if embeddings_status is ready_for_chat or completed
                    if file_info.get("embeddings_status") in [
                        "ready_for_chat",
                        "completed",
                    ]:
                        return True, True, True  # GCS status, local status, all valid
                except Exception as e:
                    logging.warning(f"Error reading local file_info.json: {str(e)}")

            # If we didn't find valid status in file_info.json, check local Azure embeddings
            azure_path = f"./chroma_db/{file_id}/azure"

            # Now we only check for Azure embeddings, since we use them for all models
            local_files_exist = os.path.exists(azure_path) and os.path.exists(
                os.path.join(azure_path, "chroma.sqlite3")
            )

            # If local Azure files exist, we consider them valid for all models
            if local_files_exist:
                logging.info(
                    f"Found valid local Azure embeddings for file_id: {file_id}"
                )
                return True, True, True

            # Check GCS Azure embeddings only
            azure_gcs_prefix = (
                f"{self.configs.gcp_resource.gcp_embeddings_folder}/{file_id}/azure/"
            )

            azure_blobs = list(
                self.gcs_handler.bucket.list_blobs(prefix=azure_gcs_prefix)
            )

            # Now we only check for Azure embeddings in GCS
            gcs_files_exist = len(azure_blobs) > 0 and any(
                blob.name.endswith("chroma.sqlite3") for blob in azure_blobs
            )

            if gcs_files_exist:
                logging.info(f"Found valid GCS Azure embeddings for file_id: {file_id}")

            return gcs_files_exist, False, False

        except Exception as e:
            logging.error(f"Error checking embeddings existence: {str(e)}")
            return False, False, False

    async def ensure_embeddings_exist(
        self,
        file_id: str,
        temp_file_path: str = None,
        file_metadata: Dict[str, Any] = None,
    ):
        """
        Ensures embeddings exist and are valid, creates only if necessary.
        """
        try:
            # Use provided file_metadata if available, otherwise fallback to global temp_metadata
            # This prevents race conditions when processing multiple files in parallel
            temp_metadata = (
                file_metadata
                if file_metadata is not None
                else self.gcs_handler.temp_metadata
            )

            if not temp_metadata:
                # Try to get metadata from GCS directly as a last resort
                temp_metadata = self.gcs_handler.get_file_info(file_id)

            if not temp_metadata:
                from rtl_rag_chatbot_api.common.errors import (
                    BaseAppError,
                    ErrorRegistry,
                )

                raise BaseAppError(
                    ErrorRegistry.ERROR_FILE_NOT_FOUND,
                    f"No metadata found for file_id: {file_id}",
                    details={"file_id": file_id},
                )

            # Check embeddings status
            gcs_status, local_status, all_valid = self.embeddings_exist(file_id)

            # If everything is valid, return immediately
            if all_valid:
                logging.info(f"Valid embeddings exist for file_id: {file_id}")
                return {
                    "message": "Embeddings already exist and are valid",
                    "status": "existing",
                }

            # If embeddings exist in GCS but not locally, just download
            if gcs_status and not local_status:
                logging.info(f"Downloading existing embeddings for file_id: {file_id}")
                self.gcs_handler.download_files_from_folder_by_id(file_id)
                return {
                    "message": "Embeddings downloaded successfully",
                    "status": "downloaded",
                }

            # Create new embeddings
            logging.info(f"Creating new embeddings for file_id: {file_id}")

            # For images, pass both analysis files
            # To DO
            if temp_metadata.get("is_image"):
                gpt4_path = temp_metadata.get("gpt4_analysis_path")
                logging.info(
                    f"Creating new embeddings for image analysis file : {gpt4_path}"
                )
                if not gpt4_path:
                    raise EmbeddingCreationError(
                        "Missing analysis path for image", details={"file_id": file_id}
                    )

                # Only pass single analysis file – unified embeddings.
                return await self.create_and_upload_embeddings(
                    file_id, gpt4_path, is_image=True
                )
            else:
                # For non-images, use single file path
                return await self.create_and_upload_embeddings(
                    file_id, temp_file_path, is_image=False, file_metadata=temp_metadata
                )

        except Exception as e:
            logging.error(f"Error in ensure_embeddings_exist: {str(e)}")
            raise

    def _resolve_file_metadata(
        self, file_id: str, file_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Resolves file metadata from various sources, ensuring it's valid for the given file ID.

        Args:
            file_id: Unique identifier for the file
            file_metadata: Optional metadata provided directly

        Returns:
            Dictionary containing valid metadata for the file
        """
        # Check if provided file_metadata matches the requested file_id
        if file_metadata is not None and file_metadata.get("file_id") == file_id:
            logging.info(f"Using provided file-specific metadata for {file_id}")
            return file_metadata
        elif file_metadata is not None and file_metadata.get("file_id") != file_id:
            # The provided metadata doesn't match our file_id - this is a serious issue
            logging.warning(
                f"Provided file_metadata has file_id {file_metadata.get('file_id')}"
                f" which doesn't match requested file_id {file_id}"
            )
            # Reset file_metadata to None to force looking up the correct metadata
            file_metadata = None

        # Check temp_metadata if available
        if file_metadata is None and self.gcs_handler.temp_metadata:
            if self.gcs_handler.temp_metadata.get("file_id") == file_id:
                # Create a deep copy to avoid modifying shared state
                file_metadata = self.gcs_handler.temp_metadata.copy()
                file_metadata["_metadata_used_for"] = file_id
                logging.info(f"Using copied temp_metadata for {file_id}")
            else:
                # Wrong temp_metadata for this file_id
                logging.warning(
                    f"temp_metadata file_id {self.gcs_handler.temp_metadata.get('file_id')}"
                    f" doesn't match requested file_id {file_id}"
                )

        # Try stored file info as last resort
        if file_metadata is None:
            file_info = self.get_embeddings_info(file_id)
            if file_info:
                file_metadata = file_info
                logging.info(f"Using existing stored metadata for {file_id}")
            else:
                # Create minimal valid metadata
                file_metadata = {"file_id": file_id}
                logging.warning(
                    f"No complete metadata available for {file_id}, using minimal metadata"
                )

        # Ensure file_id in metadata is correct
        if file_metadata.get("file_id") != file_id:
            logging.warning(
                f"Correcting file_id in metadata from {file_metadata.get('file_id')} to {file_id}"
            )
            file_metadata["file_id"] = file_id

        if "file_hash" in file_metadata:
            logging.info(f"File hash for {file_id}: {file_metadata['file_hash']}")

        return file_metadata

    def _manage_temp_metadata_isolation(self, file_id: str) -> None:
        """
        Manages temp_metadata to prevent cross-file contamination.

        Args:
            file_id: The file ID to check against temp_metadata
        """
        # Skip if no temp_metadata exists
        if not self.gcs_handler.temp_metadata:
            return

        if self.gcs_handler.temp_metadata.get("file_id") == file_id:
            # Clear temp_metadata that belongs to current file after use
            self.gcs_handler.temp_metadata = None
            logging.info(f"Cleared global temp_metadata after using for {file_id}")
        else:
            # Add isolation marker for another file's metadata
            logging.warning(
                f"Not clearing global temp_metadata as it belongs to another file: "
                f"{self.gcs_handler.temp_metadata.get('file_id')} vs requested {file_id}"
            )

            # Add isolation markers if needed
            if (
                isinstance(self.gcs_handler.temp_metadata, dict)
                and "_isolation_markers" not in self.gcs_handler.temp_metadata
            ):
                self.gcs_handler.temp_metadata["_isolation_markers"] = []

            if isinstance(self.gcs_handler.temp_metadata, dict) and isinstance(
                self.gcs_handler.temp_metadata.get("_isolation_markers"), list
            ):
                self.gcs_handler.temp_metadata["_isolation_markers"].append(file_id)

    async def _process_embeddings_with_timeout(
        self,
        file_id: str,
        base_handler,
        temp_file_path: str,
        second_file_path: str = None,
        is_image: bool = False,
        extracted_text: str = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process embeddings with timeout handling.

        Args:
            file_id: Unique identifier for the file
            base_handler: BaseRAGHandler instance for processing
            temp_file_path: Path to the temporary file
            second_file_path: Optional path to second file (for images)
            is_image: Whether the file is an image
            extracted_text: Pre-extracted text to avoid duplicate extraction

        Returns:
            Tuple of (azure_result, gemini_result) dictionaries
        """
        try:
            # Use asyncio.wait_for to add timeouts to embedding creation
            if is_image:
                return await asyncio.wait_for(
                    self._process_image_file(
                        file_id, base_handler, temp_file_path, second_file_path
                    ),
                    timeout=900,  # 15 minute timeout
                )
            else:
                return await asyncio.wait_for(
                    self._process_regular_file(
                        file_id, base_handler, temp_file_path, extracted_text
                    ),
                    timeout=900,  # 15 minute timeout
                )
        except asyncio.TimeoutError:
            logging.error(f"Embedding creation timed out for file_id {file_id}")
            # Check which embeddings were created successfully before timeout
            azure_embeddings_check = await self.check_embeddings_exist(
                file_id, "gpt_4o_mini"
            )
            gemini_embeddings_check = await self.check_embeddings_exist(
                file_id, "gemini-2.5-flash"
            )

            # Convert check results to proper format
            azure_result = {
                "success": azure_embeddings_check.get("embeddings_exist", False),
                "error": None
                if azure_embeddings_check.get("embeddings_exist", False)
                else "Embedding creation timed out",
            }
            gemini_result = {
                "success": gemini_embeddings_check.get("embeddings_exist", False),
                "error": None
                if gemini_embeddings_check.get("embeddings_exist", False)
                else "Embedding creation timed out",
            }

            logging.info(
                f"Azure embeddings exist: {azure_result['success']}, "
                f"Gemini embeddings exist: {gemini_result['success']}"
            )

            return azure_result, gemini_result

    def _extract_embedding_exists_status(self, result) -> bool:
        """
        Safely extract embedding existence status from various result formats.

        Args:
            result: Result object from embedding creation (could be dict, bool, etc.)

        Returns:
            Boolean indicating if embeddings exist
        """
        if isinstance(result, dict):
            return result.get("embeddings_exist", False) or result.get("success", False)
        elif isinstance(result, bool):
            return result
        return False

    async def create_embeddings(
        self,
        file_id: str,
        temp_file_path: str,
        second_file_path: str = None,
        is_image: bool = False,
        file_metadata: Dict[str, Any] = None,
        extracted_text: str = None,
    ) -> Dict[str, Any]:
        """
        Creates embeddings for a file locally. This is a blocking operation.

        Args:
            file_id: Unique identifier for the file
            temp_file_path: Path to the temporary file
            second_file_path: Optional second file path (for images with separate analysis)
            is_image: Whether the file is an image
            file_metadata: Additional metadata for the file
            extracted_text: Pre-extracted text to avoid duplicate extraction

        Returns:
            Dictionary with embedding status information
        """
        try:
            # Resolve file metadata from various sources
            file_metadata = self._resolve_file_metadata(file_id, file_metadata)

            # Create base handler for processing
            base_handler = self.create_base_handler()

            # Manage temp_metadata to prevent cross-file contamination
            self._manage_temp_metadata_isolation(file_id)

            # Process embeddings with timeout handling
            azure_result, gemini_result = await self._process_embeddings_with_timeout(
                file_id,
                base_handler,
                temp_file_path,
                second_file_path,
                is_image,
                extracted_text,
            )

            # Extract embedding existence status
            azure_embeddings_exist = self._extract_embedding_exists_status(azure_result)
            gemini_embeddings_exist = self._extract_embedding_exists_status(
                gemini_result
            )

            # Update file info with embedding status before uploading
            # This makes chat available immediately
            file_info = self._prepare_file_info(
                file_id, azure_result, gemini_result, file_metadata
            )
            file_info["embeddings_status"] = "ready_for_chat"

            # Save file_info locally to make chat available immediately
            self._save_file_info_locally(file_id, file_info)

            return {
                "message": "Embeddings created successfully. Ready for chat.",
                "status": "ready_for_chat",
                "can_chat": True,
                "file_id": file_id,
                "azure_embeddings_exist": azure_embeddings_exist,
                "gemini_embeddings_exist": gemini_embeddings_exist,
                **file_info,
            }

        except Exception as e:
            logging.error(f"Error in create_embeddings: {str(e)}", exc_info=True)
            # Clean up any partial embeddings
            await self._handle_failed_embedding_cleanup(file_id)
            return self._build_error_response(file_id, str(e))

    async def upload_embeddings(self, file_id: str) -> None:
        """
        Uploads embeddings to GCS. This should be called as a non-blocking operation
        after create_embeddings has completed.

        Args:
            file_id: Unique identifier for the file
        """
        try:
            # Execute the upload function directly
            # When called via background_tasks.add_task, this will run in the background
            await self._upload_embeddings_to_gcs_background(file_id)
            logging.info(
                f"Completed background upload of embeddings to GCS for {file_id}"
            )
        except Exception as e:
            logging.error(f"Error during background upload: {str(e)}", exc_info=True)

    async def create_and_upload_embeddings(
        self,
        file_id: str,
        temp_file_path: str,
        second_file_path: str = None,
        is_image: bool = False,
        file_metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Creates embeddings for a file and uploads them to GCS.
        This is maintained for backward compatibility but uses the new separate methods.

        Args:
            file_id: Unique identifier for the file
            temp_file_path: Path to the temporary file
            second_file_path: Optional second file path (for images with separate analysis)
            is_image: Whether the file is an image
            file_metadata: Additional metadata for the file

        Returns:
            Dictionary with embedding status information
        """
        try:
            # Create embeddings first (blocking operation)
            result = await self.create_embeddings(
                file_id, temp_file_path, second_file_path, is_image, file_metadata
            )

            # If embeddings were created successfully, start the upload in the background
            if result.get("status") == "ready_for_chat":
                # Create a new task to handle the upload in the background
                asyncio.create_task(self.upload_embeddings(file_id))
                logging.info(
                    f"Started background upload via create_and_upload_embeddings for {file_id}"
                )

            return result

        except Exception as e:
            logging.error(
                f"Error in create_and_upload_embeddings: {str(e)}", exc_info=True
            )
            # Clean up any partial embeddings
            await self._handle_failed_embedding_cleanup(file_id)
            return self._build_error_response(file_id, str(e))

    def _save_file_info_locally(self, file_id: str, file_info: Dict[str, Any]) -> None:
        """
        Save file info locally to make embeddings immediately available for chat.

        Args:
            file_id: The ID of the file
            file_info: The file info dictionary to save
        """
        try:
            # Ensure the directory exists
            local_dir = os.path.join("./chroma_db", file_id)
            os.makedirs(local_dir, exist_ok=True)

            # Write the file info locally
            local_path = os.path.join(local_dir, "file_info.json")
            logging.info(f"Writing file_info.json locally to {local_path}")
            with open(local_path, "w") as f:
                json.dump(file_info, f, indent=2)

            # Update the embeddings status to ready_for_chat
            file_info["embeddings_status"] = "ready_for_chat"

        except Exception as e:
            logging.error(f"Error saving file_info locally: {str(e)}")

    async def _handle_failed_embedding_cleanup(self, file_id: str) -> None:
        """
        Safely clean up failed embedding artifacts.

        Args:
            file_id: The ID of the file whose embeddings failed
        """
        try:
            await self._cleanup_failed_embeddings(file_id)
        except Exception as cleanup_error:
            logging.error(f"Error during embeddings cleanup: {str(cleanup_error)}")

    def _build_error_response(self, file_id: str, error_message: str) -> Dict[str, Any]:
        """
        Build a standardized error response for embedding failures.

        Args:
            file_id: The ID of the file that failed
            error_message: The error message to include

        Returns:
            Dictionary with error information including code/key
        """
        from rtl_rag_chatbot_api.common.errors import (
            EmbeddingCreationError,
            build_error_result,
            map_exception_to_app_error,
        )

        # Try to map the error message to a specific error type
        # Create a temporary exception to use with map_exception_to_app_error
        temp_exc = Exception(error_message)
        app_error = map_exception_to_app_error(temp_exc)

        # If it's still a generic error, use EmbeddingCreationError
        if isinstance(app_error, Exception) and not hasattr(app_error, "spec"):
            app_error = EmbeddingCreationError(
                error_message, details={"file_id": file_id}
            )

        result = build_error_result(app_error, file_id=file_id)
        result.update(
            {
                "can_chat": False,
                "azure_embeddings_exist": False,
                "gemini_embeddings_exist": False,
            }
        )
        return result

    async def check_embeddings_exist(
        self, file_id: str, model_choice: str, embedding_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check if embeddings exist for a specific file and model.
        With our unified embedding approach, we always check for Azure embeddings.

        Args:
            file_id (str): The ID of the file to check
            model_choice (str): The chosen model (e.g., 'gpt-4', 'gemini-pro')

        Returns:
            Dict containing:
                - embeddings_exist (bool): Whether embeddings exist
                - model_type (str): Type of model (azure/google)
                - file_id (str): The checked file ID
                - status (str): The status of the embeddings (ready_for_chat, completed, etc.)
        """
        try:
            model_choice = model_choice.lower()
            # We now always use Azure embeddings regardless of model_choice
            model_type = "azure"  # Always use Azure embeddings for unified approach

            # Use provided embedding_type if available (from all_file_infos), otherwise check DB
            if embedding_type:
                logging.info(f"File {file_id} has embedding_type: '{embedding_type}'")
                return {
                    "embeddings_exist": True,
                    "model_type": embedding_type,
                    "file_id": file_id,
                    "status": "completed",
                }

            # Fallback to DB lookup only if embedding_type not provided
            from rtl_rag_chatbot_api.app import get_db_session
            from rtl_rag_chatbot_api.common.db import get_file_info_by_file_id

            try:
                with get_db_session() as db_session:
                    # AIP-1060, https://rtldata.atlassian.net/browse/AIP-1060
                    # take embedding type from database if available
                    record = get_file_info_by_file_id(db_session, file_id)

                    if record and record.embedding_type:
                        logging.info(
                            f"Database record for {file_id} shows embedding_type: {record.embedding_type}"
                        )

                        return {
                            "embeddings_exist": True,
                            "model_type": record.embedding_type,
                            # AIP-1066, https://rtldata.atlassian.net/browse/AIP-1066
                            "file_name": record.file_name,
                            "file_id": file_id,
                            "status": "completed",
                        }
                    else:
                        logging.warning(
                            f"No database record found for {file_id}. Checking local file_info.json next and then GCS."
                        )
            except Exception as db_error:
                logging.error(
                    f"Database error while checking embeddings: {str(db_error)}"
                )
                # we continue to check local and GCS as fallback
                pass

            # Second, check file_info.json for embedding status
            local_info_path = os.path.join("./chroma_db", file_id, "file_info.json")
            embeddings_status = "not_started"
            embeddings_exist = False
            file_name = None

            # Check local embeddings first (most reliable and up-to-date source)
            # If local file_info.json exists, check it first
            if os.path.exists(local_info_path):
                try:
                    with open(local_info_path, "r") as f:
                        file_info = json.load(f)
                        embeddings_status = file_info.get(
                            "embeddings_status", "not_started"
                        )
                        # AIP-1066, https://rtldata.atlassian.net/browse/AIP-1066
                        file_name = file_info.get("original_filename", None)
                        model_type = file_info.get("embedding_type", model_type)

                    logging.info(
                        f"Local file_info.json for {file_id} shows status: {embeddings_status}"
                    )

                    # Consider in_progress status as valid if embedding files exist
                    # This is needed because sometimes the status update from in_progress to ready_for_chat
                    # might not be immediate, but embeddings are already created
                    if embeddings_status in [
                        "in_progress",
                        "ready_for_chat",
                        "completed",
                    ]:
                        # Check if the actual embedding files exist
                        azure_dir = os.path.join("./chroma_db", file_id, "azure")
                        azure_db = os.path.join(azure_dir, "chroma.sqlite3")

                        # If we have the directory and either the DB or any files in the directory
                        if os.path.exists(azure_dir) and (
                            os.path.exists(azure_db)
                            or any(
                                os.path.isfile(os.path.join(azure_dir, f))
                                for f in os.listdir(azure_dir)
                                if f
                            )
                        ):
                            logging.info(
                                f"Found valid local Azure embeddings for file {file_id} (using with {model_choice})"
                            )
                            embeddings_exist = True
                            return {
                                "embeddings_exist": True,
                                "model_type": model_type,  # Return original model_type for compatibility
                                "file_id": file_id,
                                # AIP-1066, https://rtldata.atlassian.net/browse/AIP-1066
                                "file_name": file_name if file_name else None,
                                "status": embeddings_status,
                            }
                        else:
                            logging.warning(
                                f"File {file_id} has status {embeddings_status}"
                                " but no valid embedding files found locally"
                            )
                except Exception as e:
                    logging.warning(f"Error reading local file_info.json: {str(e)}")

            # If no valid local embeddings or status, check GCS as fallback
            gcs_prefix = (
                f"{self.configs.gcp_resource.gcp_embeddings_folder}/{file_id}/azure/"
            )
            blobs = list(self.gcs_handler.bucket.list_blobs(prefix=gcs_prefix))

            gcs_embeddings_exist = len(blobs) > 0 and any(
                blob.name.endswith("chroma.sqlite3") for blob in blobs
            )

            # Only update embeddings_exist if we found them in GCS and not locally
            if gcs_embeddings_exist and not embeddings_exist:
                embeddings_exist = True
                logging.info(
                    f"Found embeddings in GCS for {file_id} (not found locally)"
                )

                # Update status from GCS if needed
                if embeddings_status == "not_started":
                    file_info = self.gcs_handler.get_file_info(file_id)
                    if file_info:
                        embeddings_status = file_info.get(
                            "embeddings_status", "not_started"
                        )
                        file_name = file_info.get("original_filename", None)
                        model_type = file_info.get("embedding_type", model_type)

            logging.info(
                f"Final check for file {file_id} with {model_choice}:"
                f" exist={embeddings_exist}, status={embeddings_status}"
            )
            return {
                "embeddings_exist": embeddings_exist,
                "model_type": model_type,  # Still return original model_type for compatibility
                "file_id": file_id,
                # AIP-1066, https://rtldata.atlassian.net/browse/AIP-1066
                "file_name": file_name if file_name else None,
                "status": embeddings_status,
            }

        except Exception as e:
            logging.error(
                f"Error checking embeddings for {file_id} with {model_choice}: {str(e)}"
            )
            raise EmbeddingCreationError(
                f"Error checking embeddings: {str(e)}",
                details={"file_id": file_id, "model_choice": model_choice},
            )

    async def _create_azure_embeddings(self, file_id: str, chunks: List[str]):
        """Creates embeddings using Azure OpenAI."""
        logging.info(f"Generating Azure embeddings for file_id: {file_id}...")
        try:
            # Use a default model_choice like 'gpt_4o_mini' as AzureChatbot is used here
            # primarily for its configuration and BaseRAGHandler methods for embeddings.
            # The actual embedding model (e.g., text-embedding-ada-002) is handled by create_and_store_embeddings.
            azure_handler = AzureChatbot(
                configs=self.configs,
                gcs_handler=self.gcs_handler,
                model_choice="gpt_4o_mini",  # Or another default chat model from your config
                file_id=file_id,
                collection_name_prefix="rag_collection_",  # Ensure this matches how collection_name is formed
                chroma_manager=self.chroma_manager,  # Pass the shared manager
            )

            # Force re-embedding to always use Azure 3 Large regardless of legacy metadata
            # By providing embedding_type via all_file_infos, AzureChatbot will select the 3-large deployment
            azure_handler.all_file_infos = {
                file_id: {"embedding_type": "azure-3-large"}
            }

            # Use asyncio.to_thread for IO-bound operations
            # Always create new embeddings under the newer embedding type
            result = await asyncio.to_thread(
                azure_handler.create_and_store_embeddings,
                chunks,
                file_id,
                "azure",  # Always 'azure' in the unified embedding approach
                is_embedding=True,
            )

            logging.info(
                f"Azure embeddings generated successfully for file_id: {file_id}"
            )
            return result
        except Exception as e:
            logging.error(
                f"Error creating Azure embeddings for file_id {file_id}: {str(e)}",
                exc_info=True,
            )
            raise

    async def _create_gemini_embeddings(self, file_id: str, chunks: List[str]):
        """
        No-op method that returns success without creating any embeddings.
        We now use Azure embeddings for all LLM types.
        """
        logging.info(
            f"Skipping Gemini embedding creation for file_id: {file_id} (using Azure embeddings)"
        )
        return {"success": True, "status": "completed"}

    async def _upload_embeddings_to_gcs_background(self, file_id: str):
        """
        Upload embeddings to GCS in the background without blocking.
        This allows chat to start immediately after local embeddings are available.
        """
        try:
            logging.info(
                f"Background task: Uploading embeddings to GCS for file_id: {file_id}..."
            )

            # Use a timeout to prevent indefinite waiting
            await asyncio.wait_for(
                self._upload_embeddings_to_gcs(file_id),
                timeout=300,  # 5 minutes timeout
            )

            # After successful upload, update file_info.json status from ready_for_chat to completed
            await self._update_embedding_status_to_completed(file_id)

            return "completed"
        except asyncio.TimeoutError:
            logging.error(
                f"Background embedding upload timed out for file_id {file_id}"
            )
            return {"success": False, "error": "Upload timed out"}
        except Exception as e:
            logging.error(f"Error in background upload embeddings to GCS: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _update_embedding_status_to_completed(self, file_id: str):
        """
        Update file_info.json status from ready_for_chat to completed after successful GCS upload.
        """
        try:
            # Read the local file_info.json
            local_path = os.path.join("./chroma_db", file_id, "file_info.json")
            if not os.path.exists(local_path):
                logging.warning(f"Cannot update status: {local_path} does not exist")
                return

            with open(local_path, "r") as f:
                file_info = json.load(f)

            # Update status
            file_info["embeddings_status"] = "completed"

            # Save locally
            with open(local_path, "w") as f:
                json.dump(file_info, f, indent=2)

            # Update in GCS
            self.gcs_handler.upload_to_gcs(
                self.configs.gcp_resource.bucket_name,
                {
                    "metadata": (
                        file_info,
                        f"{self.configs.gcp_resource.gcp_embeddings_folder}/{file_id}/file_info.json",
                    )
                },
            )
            logging.info(f"Updated embedding status to 'completed' for {file_id}")
        except Exception as e:
            logging.error(f"Error updating embedding status: {str(e)}")

    async def _upload_embeddings_to_gcs(self, file_id: str):
        """
        Upload embeddings to GCS.
        """
        try:
            logging.info(f"Uploading embeddings to GCS for file_id: {file_id}...")

            # Get chroma DB directory path - only Azure embeddings are used in the unified approach
            azure_folder = os.path.join("./chroma_db", file_id, "azure")

            # Upload each file individually to avoid the dictionary unpacking issue
            upload_count = 0
            bucket_name = self.configs.gcp_resource.bucket_name

            # Process Azure folder
            if os.path.exists(azure_folder):
                for root, _, files in os.walk(azure_folder):
                    for file in files:
                        try:
                            local_path = os.path.join(root, file)
                            relative_path = os.path.relpath(local_path, "./chroma_db")
                            gcs_path = f"{self.configs.gcp_resource.gcp_embeddings_folder}/{relative_path}"

                            # Upload each file individually
                            self.gcs_handler.upload_to_gcs(
                                bucket_name, local_path, gcs_path
                            )
                            upload_count += 1
                        except Exception as inner_e:
                            logging.error(
                                f"Error uploading Azure file {file}: {str(inner_e)}"
                            )

            # Google folder processing removed as part of unified Azure embedding approach

            if upload_count > 0:
                logging.info(
                    f"Successfully uploaded {upload_count} embedding files to GCS"
                )
            else:
                logging.warning("No embedding files found to upload")

            return "completed"
        except Exception as e:
            logging.error(f"Error uploading embeddings to GCS: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def _cleanup_failed_embeddings(self, file_id: str):
        """Remove any partially created embeddings for failed operations."""
        try:
            logging.info(f"Cleaning up failed embeddings for file_id: {file_id}")
            # Clean up local files - only Azure embeddings in unified approach
            local_path = f"./chroma_db/{file_id}/azure"
            if os.path.exists(local_path):
                # Use asyncio.to_thread for IO-bound operations like file deletion
                await asyncio.to_thread(self._remove_directory, local_path)
                logging.info(f"Removed local embeddings at {local_path}")

            # We don't delete from GCS as it might be partial and the original data should be preserved
            # for debugging purposes. In production, you might want to add GCS cleanup as well.

            # Clean up ChromaDB instance if it exists
            try:
                if hasattr(self.chroma_manager, "delete_embeddings"):
                    await asyncio.to_thread(
                        self.chroma_manager.delete_embeddings, file_id
                    )
                    logging.info(f"Deleted ChromaDB instance for {file_id}")
            except Exception as chroma_e:
                logging.error(f"Error cleaning up ChromaDB: {str(chroma_e)}")

        except Exception as e:
            logging.error(f"Error cleaning up embeddings: {str(e)}", exc_info=True)
            # We don't re-raise as this is a cleanup operation
            # and we don't want it to affect the main error handling

    def _remove_directory(self, path: str):
        """Helper method to remove a directory safely."""
        import shutil

        shutil.rmtree(path, ignore_errors=True)

    def _get_metadata(
        self, file_id: str, file_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Get metadata for a file with proper fallback mechanisms."""
        # STEP 1: Check if provided file_metadata is valid for this file_id
        if file_metadata is not None:
            if file_metadata.get("file_id") == file_id:
                logging.info(f"Using provided file-specific metadata for {file_id}")
                return (
                    file_metadata.copy()
                )  # Return a copy to prevent modifications affecting the original
            else:
                logging.warning(
                    f"Provided file_metadata has incorrect file_id: {file_metadata.get('file_id')} vs {file_id}"
                )
                # Don't use this metadata as it belongs to another file
                file_metadata = None

        # STEP 2: Check if global temp_metadata is valid for this file_id
        if self.gcs_handler.temp_metadata:
            if self.gcs_handler.temp_metadata.get("file_id") == file_id:
                metadata_copy = self.gcs_handler.temp_metadata.copy()
                # Mark that we've accessed this to track potential cross-contamination
                metadata_copy["_accessed_by"] = file_id

                # Check if this file_id is in the isolation markers
                isolation_markers = self.gcs_handler.temp_metadata.get(
                    "_isolation_markers", []
                )
                if file_id in isolation_markers:
                    logging.warning(
                        f"Found file_id {file_id} in isolation markers, this metadata should not be used"
                    )
                    # Don't use this metadata as it's explicitly marked as not for this file
                    return self._get_metadata_from_gcs(file_id)

                # Clear the global temp_metadata after use to prevent it affecting other files
                self.gcs_handler.temp_metadata = None
                logging.info(
                    f"Cleared global temp_metadata after copying for {file_id}"
                )
                return metadata_copy
            else:
                logging.warning(
                    f"Global temp_metadata has incorrect file_id:"
                    f" {self.gcs_handler.temp_metadata.get('file_id')} vs {file_id}"
                )

        # STEP 3: Try to get metadata from GCS as a last resort
        return self._get_metadata_from_gcs(file_id)

    def _get_metadata_from_gcs(self, file_id: str) -> Dict[str, Any]:
        """Helper method to get metadata from GCS with error handling."""
        temp_metadata = self.gcs_handler.get_file_info(file_id)
        if not temp_metadata:
            from rtl_rag_chatbot_api.common.errors import BaseAppError, ErrorRegistry

            raise BaseAppError(
                ErrorRegistry.ERROR_FILE_NOT_FOUND,
                f"No metadata found for file_id: {file_id}",
                details={"file_id": file_id},
            )
        return temp_metadata

    def _extract_and_chunk_text(
        self, base_handler, file_path: str, extracted_text: str = None
    ) -> tuple[str, List[str]]:
        """Extract text from file and split into chunks.

        Args:
            base_handler: Base handler for text operations
            file_path: Path to the file (used if extracted_text is not provided)
            extracted_text: Pre-extracted text (if available from FileHandler)
        """
        # Use pre-extracted text if available, otherwise extract from file
        if extracted_text:
            logging.info(
                "Using pre-extracted text from FileHandler (avoiding duplicate extraction)"
            )
            text = extracted_text
        else:
            logging.info("No pre-extracted text found, extracting from file")
            text = base_handler.extract_text_from_file(file_path)
            # Text extraction errors now raise structured exceptions from base_handler
            # No need to check for ERROR: prefix anymore

        # Save extracted text to diagnostic file
        self._save_extracted_text_for_diagnosis(file_path, text)

        # Check if extracted text is too short
        if len(text.strip()) < 100:
            raise Exception(
                "Unable to extract sufficient text from this file (less than 100 characters). "
                "Please try using the 'Chat with Image' feature instead."
            )

        text_chunks = base_handler.split_text(text)
        if not text_chunks:
            raise Exception("No processable text found in the document")

        return text, text_chunks

    def _save_extracted_text_for_diagnosis(self, file_path: str, extracted_text: str):
        """Save extracted text to a diagnostic file for testing purposes."""
        # Check if diagnostic saving is enabled
        if not getattr(self.configs, "save_extracted_text_diagnostic", False):
            return

        try:
            # Create diagnostic directory if it doesn't exist
            diagnostic_dir = "diagnostic_extracted_texts"
            os.makedirs(diagnostic_dir, exist_ok=True)

            # Generate filename based on original file
            original_filename = os.path.basename(file_path)
            base_name = os.path.splitext(original_filename)[0]
            timestamp = int(time.time())
            diagnostic_filename = f"{base_name}_extracted_text_{timestamp}.txt"
            diagnostic_path = os.path.join(diagnostic_dir, diagnostic_filename)

            # Save the extracted text
            with open(diagnostic_path, "w", encoding="utf-8") as f:
                f.write("=== EXTRACTED TEXT DIAGNOSTIC FILE ===\n")
                f.write(f"Original file: {file_path}\n")
                f.write(f"Extraction timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Text length: {len(extracted_text)} characters\n")
                f.write(f"Word count: {len(extracted_text.split())} words\n")
                f.write(f"Line count: {len(extracted_text.splitlines())} lines\n")
                f.write(f"Contains markdown headers: {'#' in extracted_text}\n")
                f.write(
                    f"Contains markdown lists: {'* ' in extracted_text or '- ' in extracted_text}\n"
                )
                f.write(f"Contains markdown bold: {'**' in extracted_text}\n")
                f.write(f"Contains markdown italic: {'*' in extracted_text}\n")
                f.write(f"Contains markdown code blocks: {'```' in extracted_text}\n")
                f.write(
                    f"Contains markdown links: {'[' in extracted_text and '](' in extracted_text}\n"
                )
                f.write(f"Contains tables: {'|' in extracted_text}\n")
                newline_check = "\\n" in repr(extracted_text)
                tab_check = "\\t" in repr(extracted_text)
                f.write(f"Contains newlines: {newline_check}\n")
                f.write(f"Contains tabs: {tab_check}\n")
                f.write(f"First 200 characters: {repr(extracted_text[:200])}\n")
                f.write(f"Last 200 characters: {repr(extracted_text[-200:])}\n")
                f.write(f"\n{'=' * 50}\n")
                f.write("FULL EXTRACTED TEXT:\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(extracted_text)
                f.write(f"\n\n{'=' * 50}\n")
                f.write("END OF EXTRACTED TEXT\n")
                f.write(f"{'=' * 50}\n")

            logging.info(f"Saved extracted text diagnostic file: {diagnostic_path}")

        except Exception as e:
            logging.warning(f"Failed to save extracted text diagnostic file: {str(e)}")

    def _log_chunk_info(self, base_handler, chunks: List[str], prefix: str = ""):
        """Log information about chunks."""
        prefix_str = f"{prefix} " if prefix else ""
        logging.info(f"{prefix_str}Text split into {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            tokens = len(base_handler.simple_tokenize(chunk))
            logging.info(f"{prefix_str}Chunk {i}: {tokens} tokens")

    async def _process_image_file(
        self, file_id: str, base_handler, temp_file_path: str, second_file_path: str
    ) -> tuple[Dict, Dict]:
        """Process image file using only Azure embeddings for all models."""
        logging.info(f"Processing image file {file_id} with Azure-only embeddings")

        try:
            # Get pre-extracted text from metadata if available
            metadata = self._get_metadata(file_id)
            extracted_text = metadata.get("extracted_text") if metadata else None

            # Process only GPT-4 analysis for embeddings
            _, gpt4_chunks = self._extract_and_chunk_text(
                base_handler, temp_file_path, extracted_text
            )
            self._log_chunk_info(base_handler, gpt4_chunks, "GPT-4")
            azure_result = await self._create_azure_embeddings(file_id, gpt4_chunks)
            logging.info(
                f"Completed Azure embedding generation for image file_id: {file_id}"
            )

            # Update metadata to indicate unified embeddings
            metadata_file = os.path.join("./chroma_db", file_id, "file_info.json")
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)

                    # Update metadata to show unified embeddings approach
                    metadata["uses_unified_embeddings"] = True
                    metadata["embedding_provider"] = "azure"

                    with open(metadata_file, "w") as f:
                        json.dump(metadata, f, indent=2)
                except Exception as e:
                    logging.warning(
                        f"Failed to update unified embedding metadata for image: {str(e)}"
                    )

            # Return success for both without creating Gemini embeddings
            gemini_result = {"success": True, "status": "completed"}
            logging.info(
                f"Using Azure embeddings for all models for image file_id: {file_id}"
            )

            return azure_result, gemini_result

        except Exception as e:
            logging.error(f"Error processing image file {file_id}: {str(e)}")
            from rtl_rag_chatbot_api.common.errors import (
                EmbeddingCreationError,
                map_exception_to_app_error,
            )

            app_error = map_exception_to_app_error(e)
            if not hasattr(app_error, "spec"):
                app_error = EmbeddingCreationError(str(e), details={"file_id": file_id})
            error_result = {
                "status": "failed",
                "error": str(e),
                "code": app_error.spec.code,
                "key": app_error.spec.key,
            }
            return error_result, error_result

    async def _process_regular_file(
        self,
        file_id: str,
        base_handler,
        temp_file_path: str,
        extracted_text: str = None,
    ) -> tuple[Dict, Dict]:
        """Process regular (non-image) file using only Azure embeddings for all models."""
        logging.info(f"Processing non-image file with file_id: {file_id}")

        # Use provided extracted text, get from file handler, or get from metadata if available
        if not extracted_text and self.file_handler:
            extracted_text = self.file_handler.get_extracted_text(file_id)

        if not extracted_text:
            metadata = self._get_metadata(file_id)
            extracted_text = metadata.get("extracted_text") if metadata else None

        # Extract text and create chunks
        _, text_chunks = self._extract_and_chunk_text(
            base_handler, temp_file_path, extracted_text
        )
        self._log_chunk_info(base_handler, text_chunks)

        # Create only Azure embeddings for all use cases
        logging.info(f"Creating Azure-only embeddings for file_id: {file_id}")

        # Create Azure embeddings
        try:
            azure_result = await self._create_azure_embeddings(file_id, text_chunks)
            logging.info(f"Completed Azure embedding generation for file_id: {file_id}")

            # Set up metadata to indicate we're using Azure embeddings for all LLMs
            metadata_file = os.path.join("./chroma_db", file_id, "file_info.json")
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)

                    # Update metadata to show all models use Azure embeddings
                    metadata["uses_unified_embeddings"] = True
                    metadata["embedding_provider"] = "azure"

                    with open(metadata_file, "w") as f:
                        json.dump(metadata, f, indent=2)
                except Exception as e:
                    logging.warning(
                        f"Failed to update unified embedding metadata: {str(e)}"
                    )

            # Return the same result for both model types
            # This maintains API compatibility with existing code expecting both results
            gemini_result = {"success": True, "status": "completed"}

        except Exception as e:
            from rtl_rag_chatbot_api.common.errors import (
                EmbeddingCreationError,
                map_exception_to_app_error,
            )

            app_error = map_exception_to_app_error(e)
            if not hasattr(app_error, "spec"):
                app_error = EmbeddingCreationError(str(e), details={"file_id": file_id})

            azure_result = e
            gemini_result = {
                "status": "failed",
                "error": str(e),
                "code": app_error.spec.code,
                "key": app_error.spec.key,
            }
            logging.error(f"Azure embedding generation failed for {file_id}: {str(e)}")

        # Handle any exceptions that occurred during processing
        if isinstance(azure_result, Exception):
            logging.error(
                f"Azure embedding generation failed for {file_id}: {str(azure_result)}"
            )
            from rtl_rag_chatbot_api.common.errors import (
                EmbeddingCreationError,
                map_exception_to_app_error,
            )

            app_error = map_exception_to_app_error(azure_result)
            if not hasattr(app_error, "spec"):
                app_error = EmbeddingCreationError(
                    str(azure_result), details={"file_id": file_id}
                )

            azure_result = {
                "status": "failed",
                "error": str(azure_result),
                "code": app_error.spec.code,
                "key": app_error.spec.key,
            }
            gemini_result = {
                "status": "failed",
                "error": str(azure_result),
                "code": app_error.spec.code,
                "key": app_error.spec.key,
            }

        logging.info(
            f"Completed Azure-only embedding generation for file_id: {file_id}"
        )
        return azure_result, gemini_result

    async def _create_file_info(
        self,
        file_id: str,
        temp_metadata: Dict[str, Any],
        azure_result: Dict,
        gemini_result: Dict,
    ) -> Dict[str, Any]:
        """Create file info dictionary with embeddings metadata."""
        # Get metadata file_id if it exists to ensure consistency
        metadata_file_id = temp_metadata.get("file_id") if temp_metadata else None

        # If metadata has a file_id that differs from the passed file_id, log a warning but USE THE PASSED file_id
        # This prevents file_id mix-up during parallel processing
        if metadata_file_id and metadata_file_id != file_id:
            logging.warning(
                f"Metadata file_id {metadata_file_id} differs from passed file_id "
                f"{file_id}. Using passed file_id to avoid mix-ups."
            )

        # Always use the passed file_id to avoid mix-ups in concurrent processing
        consistent_file_id = file_id

        # Create file_info with embeddings metadata
        file_info = {
            **temp_metadata,  # Include original metadata
            "embeddings": {
                "azure": azure_result,
                "gemini": gemini_result,
            },
            "embeddings_status": "completed",
            "azure_ready": True,
            "file_id": consistent_file_id,  # Ensure file_id consistency
            "embeddings_created_at": datetime.now().isoformat(),  # Track when embeddings were created
        }

        # Ensure migrated flag is present (default to False if not in temp_metadata)
        if "migrated" not in file_info:
            file_info["migrated"] = False

        # Ensure critical fields are present
        # If we're missing file_hash but have it in GCS, retrieve it from there
        if "file_hash" not in file_info:
            logging.warning(
                f"file_hash missing in metadata for {file_id}, attempting to retrieve from GCS"
            )
            gcs_metadata = self.gcs_handler.get_file_info(file_id)
            if gcs_metadata and "file_hash" in gcs_metadata:
                file_info["file_hash"] = gcs_metadata["file_hash"]
                logging.info(f"Retrieved file_hash from GCS: {file_info['file_hash']}")
            else:
                logging.error(f"Unable to retrieve file_hash for {file_id} from GCS")

        # Preserve original_filename if available
        if (
            "original_filename" not in file_info
            and temp_metadata
            and "original_filename" in temp_metadata
        ):
            file_info["original_filename"] = temp_metadata["original_filename"]

        return file_info

    def has_local_embeddings(self, file_id: str) -> bool:
        """
        Check if local embeddings exist for immediate use without waiting for GCS uploads.

        Args:
            file_id: Unique identifier for the file

        Returns:
            bool: True if local embeddings exist and are ready for immediate use
        """
        # Check for the local embeddings directories and SQLite files
        azure_path = f"./chroma_db/{file_id}/azure/chroma.sqlite3"
        return os.path.exists(azure_path)

    def all_files_have_local_embeddings(self, file_ids: List[str]) -> bool:
        """
        Check if all files in a list have local embeddings available.

        Args:
            file_ids: List of file IDs to check

        Returns:
            bool: True if all files have local embeddings, False otherwise
        """
        if not file_ids:
            return False

        for file_id in file_ids:
            if not self.has_local_embeddings(file_id):
                return False
        return True

    def get_embeddings_info(self, file_id: str):
        # Retrieve embeddings info from GCS
        try:
            info_blob_name = f"{self.configs.gcp_resource.gcp_embeddings_folder}/{file_id}/file_info.json"
            blob = self.gcs_handler.bucket.blob(info_blob_name)
            if blob.exists():
                content = blob.download_as_text()
                return json.loads(content)
            else:
                return None
        except Exception as e:
            logging.error(
                f"Error retrieving embeddings info for file_id {file_id}: {str(e)}"
            )
            return None

    async def migrate_embeddings_to_new_format(
        self,
        file_id: str,
        username: str | None = None,
    ) -> Dict[str, Any]:
        """
        Encapsulated migration: delete old artifacts and recreate fresh ones for the given file_id.

        Steps:
        - Optionally append username to file's username list
        - Download and decrypt original source to a temp path (first!)
        - Delete previous embeddings/artifacts (local + GCS + DB when enabled)
        - Recreate: for tabular, rebuild SQLite DB and upload; for others, recreate embeddings
        - For embeddings, schedule background upload to GCS

        Returns the standard creation result dict (ready_for_chat/completed on success).
        """
        try:
            # 1) Fetch file info and basic flags
            file_info = self.gcs_handler.get_file_info(file_id) or {}
            original_filename = file_info.get("original_filename")
            file_type = file_info.get("file_type")
            is_image = bool(file_info.get("is_image", False))

            # 2) Update username list if provided
            if username:
                try:
                    self.gcs_handler.update_username_list(file_id, [username])
                    logging.info(
                        f"Updated username list for {file_id} with user {username}"
                    )
                except Exception as e:
                    logging.warning(
                        f"Failed to update username list for {file_id}: {str(e)}"
                    )

            # 3) Download and decrypt original file to temp path (must happen BEFORE deletion)
            import os as _os

            base_name = (
                _os.path.basename(original_filename) if original_filename else None
            )
            preferred_temp_path = (
                f"temp_files/{file_id}_{base_name}"
                if base_name
                else f"temp_files/{file_id}"
            )
            decrypted_path = self.gcs_handler.download_encrypted_file_by_id(
                file_id, destination_path=preferred_temp_path
            )
            temp_file_path = (
                decrypted_path
                if decrypted_path and _os.path.exists(decrypted_path)
                else None
            )
            if not temp_file_path:
                legacy_temp = f"temp_files/{file_id}"
                if _os.path.exists(legacy_temp):
                    temp_file_path = legacy_temp
            if not temp_file_path:
                raise RuntimeError(
                    f"Unable to obtain decrypted source file for migration of {file_id}"
                )

            # 4) Delete old embeddings/artifacts (local + GCS + DB when enabled)
            try:
                from rtl_rag_chatbot_api.common.cleanup_coordinator import (
                    CleanupCoordinator,
                )

                cleanup = CleanupCoordinator(self.configs, None, self.gcs_handler)
                await asyncio.to_thread(cleanup.cleanup_chroma_instance, file_id, True)
                logging.info(f"Deleted old embeddings for {file_id}")
            except Exception as e:
                logging.error(f"Error deleting old embeddings for {file_id}: {str(e)}")
                raise

            # 5) Determine handling based on type: tabular vs non-tabular
            ext = _os.path.splitext(base_name)[1].lower() if base_name else ""
            is_tabular = file_type in ["tabular", "database"] or ext in [
                ".csv",
                ".xlsx",
                ".xls",
                ".db",
                ".sqlite",
                ".sqlite3",
            ]

            if is_tabular:
                # Rebuild SQLite DB for tabular files (same as new-file path)
                try:
                    from rtl_rag_chatbot_api.chatbot.utils.encryption import (
                        encrypt_file,
                    )
                    from rtl_rag_chatbot_api.common.prepare_sqlitedb_from_csv_xlsx import (
                        PrepareSQLFromTabularData,
                    )

                    data_dir = f"./chroma_db/{file_id}"
                    os.makedirs(data_dir, exist_ok=True)

                    # Create/refresh SQLite DB
                    preparer = PrepareSQLFromTabularData(temp_file_path, data_dir)
                    success = preparer.run_pipeline()
                    if not success:
                        raise ValueError("Failed to prepare database from input file")

                    # Encrypt and upload DB
                    db_path = os.path.join(data_dir, "tabular_data.db")
                    encrypted_db_path = encrypt_file(db_path)
                    try:
                        self.gcs_handler.upload_to_gcs(
                            self.configs.gcp_resource.bucket_name,
                            source=encrypted_db_path,
                            destination_blob_name=(
                                f"{self.configs.gcp_resource.gcp_embeddings_folder}/"
                                f"{file_id}/tabular_data.db.encrypted"
                            ),
                        )
                    finally:
                        if os.path.exists(encrypted_db_path):
                            os.remove(encrypted_db_path)

                    # Update file_info.json
                    tab_file_type = (
                        "database" if ext in [".db", ".sqlite"] else "tabular"
                    )
                    metadata = {
                        "embeddings_status": "completed",
                        "file_type": tab_file_type,
                        "processing_status": "success",
                        "migrated": True,
                        "file_id": file_id,
                    }
                    # Preserve usernames if present
                    usernames = file_info.get("username", [])
                    if usernames:
                        metadata["username"] = usernames

                    self.gcs_handler.upload_to_gcs(
                        self.configs.gcp_resource.bucket_name,
                        {
                            "metadata": (
                                metadata,
                                f"{self.configs.gcp_resource.gcp_embeddings_folder}/{file_id}/file_info.json",
                            )
                        },
                    )

                    return {
                        "file_id": file_id,
                        "status": "completed",
                        "is_tabular": True,
                        "message": "Tabular database prepared and uploaded",
                    }
                except Exception as e:
                    logging.error(f"Tabular migration failed for {file_id}: {str(e)}")
                    return {
                        "file_id": file_id,
                        "status": "error",
                        "message": str(e),
                        "is_tabular": True,
                    }

            # 6) For non-tabular: recreate embeddings and schedule upload
            file_metadata = (
                dict(file_info) if isinstance(file_info, dict) else {"file_id": file_id}
            )
            file_metadata["file_id"] = file_id
            file_metadata["migrated"] = True
            file_metadata[
                "embedding_type"
            ] = self.configs.chatbot.default_embedding_type

            result = await self.create_embeddings(
                file_id=file_id,
                temp_file_path=temp_file_path,
                is_image=is_image,
                file_metadata=file_metadata,
            )
            if result.get("status") == "ready_for_chat":
                asyncio.create_task(self.upload_embeddings(file_id))
                logging.info(f"Scheduled background upload for migrated file {file_id}")
            return result

        except Exception as e:
            logging.error(f"Migration failed for {file_id}: {str(e)}", exc_info=True)
            return {
                "file_id": file_id,
                "status": "error",
                "message": str(e),
                "can_chat": False,
            }


# Only run when this file is executed directly
if __name__ == "__main__":
    import sys

    # Add parent directory to path to import configs
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    async def test_migration():
        """Test the migration function when run directly"""
        try:
            # Import configs
            from configs.app_config import Config
            from rtl_rag_chatbot_api.chatbot.gcs_handler import GCSHandler

            # Initialize configs and handlers
            configs = Config()
            gcs_handler = GCSHandler(configs)

            # Create embedding handler
            embedding_handler = EmbeddingHandler(configs, gcs_handler)

            # Test parameters (you can modify these)
            test_file_id = "test_file_123"
            test_username = "test_user"

            print(f"Testing migration for file_id: {test_file_id}")
            result = await embedding_handler.migrate_embeddings_to_new_format(
                test_file_id, test_username
            )
            print(f"Migration result: {result}")

        except Exception as e:
            print(f"Error testing migration: {str(e)}")
            import traceback

            traceback.print_exc()

    # Run the test
    asyncio.run(test_migration())
