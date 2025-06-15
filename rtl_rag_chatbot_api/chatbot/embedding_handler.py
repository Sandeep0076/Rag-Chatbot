import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

from rtl_rag_chatbot_api.chatbot.chatbot_creator import AzureChatbot
from rtl_rag_chatbot_api.chatbot.gemini_handler import GeminiHandler
from rtl_rag_chatbot_api.common.base_handler import BaseRAGHandler
from rtl_rag_chatbot_api.common.chroma_manager import ChromaDBManager

logging.basicConfig(level=logging.INFO)


class EmbeddingHandler:
    """
    Handles the creation, storage, and uploading of embeddings for Azure and Gemini models.

    Attributes:
        configs: Configuration object containing necessary settings.
        gcs_handler: Handler for Google Cloud Storage operations.
    """

    def __init__(self, configs, gcs_handler):
        self.configs = configs
        self.gcs_handler = gcs_handler
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
            logging.warning(f"azure_result is not a dictionary: {azure_result}")
            azure_result = {
                "success": False,
                "error": str(azure_result) if azure_result else "Unknown error",
            }

        if not isinstance(gemini_result, dict):
            logging.warning(f"gemini_result is not a dictionary: {gemini_result}")
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
                "gemini": gemini_result,
            },
            "embeddings_status": "completed",
            "azure_ready": azure_result.get("success", False),
            "file_id": file_id,  # Ensure file_id consistency
            "embeddings_created_at": datetime.now().isoformat(),  # Track when embeddings were created
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
        Check if embeddings exist and are valid both in GCS and locally.
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

            # If we didn't find valid status in file_info.json, check GCS status
            azure_path = f"./chroma_db/{file_id}/azure"
            gemini_path = f"./chroma_db/{file_id}/google"

            local_files_exist = (
                os.path.exists(azure_path)
                and os.path.exists(os.path.join(azure_path, "chroma.sqlite3"))
                and os.path.exists(gemini_path)
                and os.path.exists(os.path.join(gemini_path, "chroma.sqlite3"))
            )

            # If local files exist, we consider them valid
            if local_files_exist:
                return True, True, True

            # Check GCS embeddings
            azure_gcs_prefix = f"file-embeddings/{file_id}/azure/"
            gemini_gcs_prefix = f"file-embeddings/{file_id}/google/"

            azure_blobs = list(
                self.gcs_handler.bucket.list_blobs(prefix=azure_gcs_prefix)
            )
            gemini_blobs = list(
                self.gcs_handler.bucket.list_blobs(prefix=gemini_gcs_prefix)
            )

            gcs_files_exist = (
                len(azure_blobs) > 0
                and len(gemini_blobs) > 0
                and any(blob.name.endswith("chroma.sqlite3") for blob in azure_blobs)
                and any(blob.name.endswith("chroma.sqlite3") for blob in gemini_blobs)
            )

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
                raise ValueError(f"No metadata found for file_id: {file_id}")

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
                gemini_path = temp_metadata.get("gemini_analysis_path")
                logging.info(f"Creating new embeddings for : {gpt4_path}")
                logging.info(f"Creating new embeddings for : {gemini_path}")
                if not gpt4_path or not gemini_path:
                    raise ValueError("Missing analysis paths for image")

                return await self.create_and_upload_embeddings(
                    file_id, gpt4_path, second_file_path=gemini_path, is_image=True
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
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process embeddings with timeout handling.

        Args:
            file_id: Unique identifier for the file
            base_handler: BaseRAGHandler instance for processing
            temp_file_path: Path to the temporary file
            second_file_path: Optional path to second file (for images)
            is_image: Whether the file is an image

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
                    self._process_regular_file(file_id, base_handler, temp_file_path),
                    timeout=900,  # 15 minute timeout
                )
        except asyncio.TimeoutError:
            logging.error(f"Embedding creation timed out for file_id {file_id}")
            # Check which embeddings were created successfully before timeout
            azure_embeddings_check = await self.check_embeddings_exist(
                file_id, "gpt_4o_mini"
            )
            gemini_embeddings_check = await self.check_embeddings_exist(
                file_id, "gemini-flash"
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
            # Resolve file metadata from various sources
            file_metadata = self._resolve_file_metadata(file_id, file_metadata)

            # Create base handler for processing
            base_handler = self.create_base_handler()

            # Manage temp_metadata to prevent cross-file contamination
            self._manage_temp_metadata_isolation(file_id)

            # Log final metadata being used
            logging.info(f"Using metadata for file_id {file_id}: {file_metadata}")

            # Process embeddings with timeout handling
            azure_result, gemini_result = await self._process_embeddings_with_timeout(
                file_id, base_handler, temp_file_path, second_file_path, is_image
            )

            # Extract embedding existence status first - needed for file_info
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

            # Start background upload of embeddings to GCS - after local save
            # This decouples the upload from the embedding creation
            try:
                asyncio.create_task(self._upload_embeddings_to_gcs_background(file_id))
                logging.info(
                    f"Started background upload of embeddings to GCS for {file_id}"
                )
            except Exception as e:
                logging.error(f"Error starting background upload: {str(e)}")

            # File info has already been updated and saved locally earlier
            # This is now a no-op to maintain code structure

            # Save file_info locally first to make chat available immediately
            self._save_file_info_locally(file_id, file_info)

            # Upload file info to GCS in the background
            try:
                self.gcs_handler.upload_to_gcs(
                    self.configs.gcp_resource.bucket_name,
                    {
                        "metadata": (
                            file_info,
                            f"file-embeddings/{file_id}/file_info.json",
                        )
                    },
                )
                logging.info(f"Successfully uploaded file_info.json for {file_id}")
            except Exception as upload_error:
                logging.error(f"Error uploading file_info.json: {str(upload_error)}")

            # Extract embedding existence status
            azure_embeddings_exist = self._extract_embedding_exists_status(azure_result)
            gemini_embeddings_exist = self._extract_embedding_exists_status(
                gemini_result
            )

            return {
                "message": "Embeddings created and uploaded successfully. Ready for chat.",
                "status": "ready_for_chat",
                "can_chat": True,
                "file_id": file_id,
                "azure_embeddings_exist": azure_embeddings_exist,
                "gemini_embeddings_exist": gemini_embeddings_exist,
                **file_info,
            }

        except Exception as e:
            logging.error(
                f"Error in create_and_upload_embeddings: {str(e)}", exc_info=True
            )
            # Clean up any partial embeddings
            await self._handle_failed_embedding_cleanup(file_id)

            # Return error information
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
            Dictionary with error information
        """
        return {
            "file_id": file_id,
            "status": "error",
            "message": "An error occurred while processing your file.",
            "error_message": error_message,
            "can_chat": False,
            "azure_embeddings_exist": False,
            "gemini_embeddings_exist": False,
        }

    async def check_embeddings_exist(
        self, file_id: str, model_choice: str
    ) -> Dict[str, Any]:
        """
        Check if embeddings exist for a specific file and model.
        With the decoupled approach, checks local files first before GCS.

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
            model_type = "azure" if "gpt" in model_choice else "google"
            folder_name = "azure" if model_type == "azure" else "google"

            # First check file_info.json for embedding status
            local_info_path = os.path.join("./chroma_db", file_id, "file_info.json")
            embeddings_status = "not_started"

            # If local file_info.json exists, check it first
            if os.path.exists(local_info_path):
                try:
                    with open(local_info_path, "r") as f:
                        file_info = json.load(f)
                        embeddings_status = file_info.get(
                            "embeddings_status", "not_started"
                        )

                    # Check if embeddings are ready for chat or completed
                    if embeddings_status in ["ready_for_chat", "completed"]:
                        # Now check if the specific model type folder exists locally
                        model_path = os.path.join(
                            "./chroma_db", file_id, folder_name, "chroma.sqlite3"
                        )
                        if os.path.exists(model_path):
                            return {
                                "embeddings_exist": True,
                                "model_type": model_type,
                                "file_id": file_id,
                                "status": embeddings_status,
                            }
                except Exception as e:
                    logging.warning(f"Error reading local file_info.json: {str(e)}")

            # If no valid local embeddings, check GCS
            gcs_prefix = f"file-embeddings/{file_id}/{folder_name}/"
            blobs = list(self.gcs_handler.bucket.list_blobs(prefix=gcs_prefix))

            embeddings_exist = len(blobs) > 0 and any(
                blob.name.endswith("chroma.sqlite3") for blob in blobs
            )

            # If we found embeddings in GCS but not locally, we might need to download them
            if embeddings_exist:
                # Check GCS file_info.json for status
                file_info = self.gcs_handler.get_file_info(file_id)
                if file_info:
                    embeddings_status = file_info.get(
                        "embeddings_status", "not_started"
                    )

            return {
                "embeddings_exist": embeddings_exist,
                "model_type": model_type,
                "file_id": file_id,
                "status": embeddings_status,
            }

        except Exception as e:
            logging.error(f"Error checking embeddings: {str(e)}")
            raise Exception(f"Error checking embeddings: {str(e)}")

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
            )

            # Use asyncio.to_thread for IO-bound operations
            await asyncio.to_thread(
                azure_handler.create_and_store_embeddings,
                chunks,
                file_id,
                "azure",
                is_embedding=True,
            )

            logging.info(
                f"Azure embeddings generated successfully for file_id: {file_id}"
            )
            return "completed"
        except Exception as e:
            logging.error(
                f"Error creating Azure embeddings for file_id {file_id}: {str(e)}",
                exc_info=True,
            )
            raise

    async def _create_gemini_embeddings(self, file_id: str, chunks: List[str]):
        """Creates embeddings using Gemini model."""
        logging.info(f"Generating Gemini embeddings for file_id: {file_id}...")
        try:
            collection_name = f"rag_collection_{file_id}"

            gemini_handler = GeminiHandler(self.configs, self.gcs_handler)
            gemini_handler.initialize(
                model="gemini-pro",
                file_id=file_id,
                embedding_type="google",
                collection_name=collection_name,
            )

            # Use asyncio.to_thread for IO-bound operations
            await asyncio.to_thread(
                gemini_handler.create_and_store_embeddings,
                chunks,
                file_id,
                "google",
                is_embedding=True,
            )

            logging.info(
                f"Gemini embeddings generated successfully for file_id: {file_id}"
            )
            return "completed"
        except Exception as e:
            logging.error(
                f"Error creating Gemini embeddings for file_id {file_id}: {str(e)}",
                exc_info=True,
            )
            raise

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
                        f"file-embeddings/{file_id}/file_info.json",
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

            # Get chroma DB directory path
            azure_folder = os.path.join("./chroma_db", file_id, "azure")
            google_folder = os.path.join("./chroma_db", file_id, "google")

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
                            gcs_path = f"file-embeddings/{relative_path}"

                            # Upload each file individually
                            self.gcs_handler.upload_to_gcs(
                                bucket_name, local_path, gcs_path
                            )
                            upload_count += 1
                        except Exception as inner_e:
                            logging.error(
                                f"Error uploading Azure file {file}: {str(inner_e)}"
                            )

            # Process Google folder
            if os.path.exists(google_folder):
                for root, _, files in os.walk(google_folder):
                    for file in files:
                        try:
                            local_path = os.path.join(root, file)
                            relative_path = os.path.relpath(local_path, "./chroma_db")
                            gcs_path = f"file-embeddings/{relative_path}"

                            # Upload each file individually
                            self.gcs_handler.upload_to_gcs(
                                bucket_name, local_path, gcs_path
                            )
                            upload_count += 1
                        except Exception as inner_e:
                            logging.error(
                                f"Error uploading Google file {file}: {str(inner_e)}"
                            )

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
            # Clean up local files
            for model in ["azure", "google"]:
                local_path = f"./chroma_db/{file_id}/{model}"
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
            raise ValueError(f"No metadata found for file_id: {file_id}")

        logging.info(
            f"Retrieved metadata from GCS for file_id {file_id}: {temp_metadata}"
        )
        return temp_metadata

    def _extract_and_chunk_text(
        self, base_handler, file_path: str
    ) -> tuple[str, List[str]]:
        """Extract text from file and split into chunks."""
        text = base_handler.extract_text_from_file(file_path)
        if text.startswith("ERROR:"):
            raise Exception(f"Error extracting text: {text[7:]}")

        text_chunks = base_handler.split_text(text)
        if not text_chunks:
            raise Exception("No processable text found in the document")

        return text, text_chunks

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
        """Process image file with separate GPT-4 and Gemini analysis files."""
        logging.info("Processing image analyses separately")

        # Process GPT-4 analysis
        _, gpt4_chunks = self._extract_and_chunk_text(base_handler, temp_file_path)
        self._log_chunk_info(base_handler, gpt4_chunks, "GPT-4")
        azure_result = self._create_azure_embeddings(file_id, gpt4_chunks)

        # Process Gemini analysis
        if not second_file_path:
            raise Exception("Second analysis file path not provided for image")

        _, gemini_chunks = self._extract_and_chunk_text(base_handler, second_file_path)
        self._log_chunk_info(base_handler, gemini_chunks, "Gemini")
        gemini_result = self._create_gemini_embeddings(file_id, gemini_chunks)

        return azure_result, gemini_result

    async def _process_regular_file(
        self, file_id: str, base_handler, temp_file_path: str
    ) -> tuple[Dict, Dict]:
        """Process regular (non-image) file with same chunks for both models."""
        logging.info(f"Processing non-image file with file_id: {file_id}")

        # Extract text and create chunks
        _, text_chunks = self._extract_and_chunk_text(base_handler, temp_file_path)
        self._log_chunk_info(base_handler, text_chunks)

        # Create embeddings using both models with same chunks SEQUENTIALLY
        logging.info(f"Starting sequential embedding generation for file_id: {file_id}")

        # First create Azure embeddings
        logging.info(f"Starting Azure embedding generation for file_id: {file_id}")
        try:
            azure_result = await self._create_azure_embeddings(file_id, text_chunks)
            logging.info(f"Completed Azure embedding generation for file_id: {file_id}")
        except Exception as e:
            azure_result = e
            logging.error(f"Azure embedding generation failed for {file_id}: {str(e)}")

        # Then create Gemini embeddings
        logging.info(f"Starting Gemini embedding generation for file_id: {file_id}")
        try:
            gemini_result = await self._create_gemini_embeddings(file_id, text_chunks)
            logging.info(
                f"Completed Gemini embedding generation for file_id: {file_id}"
            )
        except Exception as e:
            gemini_result = e
            logging.error(f"Gemini embedding generation failed for {file_id}: {str(e)}")

        # Handle any exceptions that occurred during parallel processing
        if isinstance(azure_result, Exception):
            logging.error(
                f"Azure embedding generation failed for {file_id}: {str(azure_result)}"
            )
            azure_result = {"status": "failed", "error": str(azure_result)}

        if isinstance(gemini_result, Exception):
            logging.error(
                f"Gemini embedding generation failed for {file_id}: {str(gemini_result)}"
            )
            gemini_result = {"status": "failed", "error": str(gemini_result)}

        logging.info(f"Completed parallel embedding generation for file_id: {file_id}")
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

    def get_embeddings_info(self, file_id: str):
        # Retrieve embeddings info from GCS
        try:
            info_blob_name = f"file-embeddings/{file_id}/file_info.json"
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
