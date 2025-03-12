import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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

    def embeddings_exist(self, file_id: str) -> tuple[bool, bool, bool]:
        """
        Check if embeddings exist and are valid both in GCS and locally.
        Returns tuple of (gcs_status, local_status, all_valid)
        """
        try:
            # Check local files first
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

    async def ensure_embeddings_exist(self, file_id: str, temp_file_path: str = None):
        """
        Ensures embeddings exist and are valid, creates only if necessary.
        """
        try:
            # Check temp_metadata first
            temp_metadata = self.gcs_handler.temp_metadata
            if not temp_metadata:
                raise ValueError(f"No metadata found for file_id: {file_id}")
            logging.info(f"temp_metadata: {temp_metadata}")

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
                    file_id, temp_file_path, is_image=False
                )

        except Exception as e:
            logging.error(f"Error in ensure_embeddings_exist: {str(e)}")
            raise

    async def create_and_upload_embeddings(
        self,
        file_id: str,
        temp_file_path: str,
        second_file_path: str = None,
        is_image: bool = False,
    ) -> Dict[str, Any]:
        """
        Create and upload embeddings for a file.
        For images, creates separate embeddings for GPT-4 and Gemini analyses.

        Args:
            file_id: Unique identifier for the file
            temp_file_path: Path to the first analysis file (GPT-4 for images)
            second_file_path: Path to the second analysis file (Gemini for images)
            is_image: Whether the file is an image
        """
        try:
            # Create base handler for text extraction

            base_handler = BaseRAGHandler(self.configs, self.gcs_handler)
            temp_metadata = self.gcs_handler.temp_metadata
            # username = temp_metadata.get("username", "Unknown")

            if is_image:
                # For images, create separate chunks for each analysis
                logging.info("Processing image analyses separately")

                # Extract text from GPT-4 analysis
                gpt4_text = base_handler.extract_text_from_file(temp_file_path)
                if gpt4_text.startswith("ERROR:"):
                    raise Exception(
                        f"Error extracting text from GPT-4 analysis: {gpt4_text[7:]}"
                    )
                gpt4_chunks = base_handler.split_text(gpt4_text)
                if not gpt4_chunks:
                    return {
                        "message": "No processable text found in the document.",
                        "status": "error",
                    }

                # Log chunk information
                logging.info(f"GPT-4 analysis split into {len(gpt4_chunks)} chunks")
                for i, chunk in enumerate(gpt4_chunks):
                    tokens = len(base_handler.simple_tokenize(chunk))
                    logging.info(f"GPT-4 Chunk {i}: {tokens} tokens")
                # Create embeddings using respective models
                azure_result = self._create_azure_embeddings(file_id, gpt4_chunks)

                # Extract text from Gemini analysis
                if not second_file_path:
                    raise Exception("Second analysis file path not provided for image")
                gemini_text = base_handler.extract_text_from_file(second_file_path)
                if gemini_text.startswith("ERROR:"):
                    raise Exception(
                        f"Error extracting text from Gemini analysis: {gemini_text[7:]}"
                    )
                gemini_chunks = base_handler.split_text(gemini_text)
                if not gemini_chunks:
                    return {
                        "message": "No processable text found in the document.",
                        "status": "error",
                    }
                logging.info(f"Gemini analysis split into {len(gemini_chunks)} chunks")
                for i, chunk in enumerate(gemini_chunks):
                    tokens = len(base_handler.simple_tokenize(chunk))
                    logging.info(f"Gemini Chunk {i}: {tokens} tokens")

                gemini_result = self._create_gemini_embeddings(file_id, gemini_chunks)

            else:
                # For non-image files, process normally with same chunks for both models
                logging.info("Processing non-image file")
                text = base_handler.extract_text_from_file(temp_file_path)
                if text.startswith("ERROR:"):
                    raise Exception(f"Error extracting text: {text[7:]}")

                text_chunks = base_handler.split_text(text)
                if not text_chunks:
                    raise Exception("No processable text found in the document")

                # Log chunk information
                logging.info(f"Text split into {len(text_chunks)} chunks")
                for i, chunk in enumerate(text_chunks):
                    tokens = len(base_handler.simple_tokenize(chunk))
                    logging.info(f"Chunk {i}: {tokens} tokens")

                # Create embeddings using both models with same chunks
                azure_result = self._create_azure_embeddings(file_id, text_chunks)
                gemini_result = self._create_gemini_embeddings(file_id, text_chunks)

            try:
                # Upload embeddings to GCS
                await self._upload_embeddings_to_gcs(file_id)

                # Get metadata file_id if it exists to ensure consistency
                metadata_file_id = (
                    temp_metadata.get("file_id") if temp_metadata else None
                )

                # If metadata has a file_id that differs from the passed file_id, use the metadata one
                # This ensures we always use the same file_id that's in the metadata
                consistent_file_id = metadata_file_id if metadata_file_id else file_id

                # Only create file_info.json after successful upload
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

                # Save file_info.json using the consistent file_id for the path
                self.gcs_handler.upload_to_gcs(
                    self.configs.gcp_resource.bucket_name,
                    {
                        "metadata": (
                            file_info,
                            f"file-embeddings/{consistent_file_id}/file_info.json",
                        )
                    },
                )

                # Clear temporary metadata
                self.gcs_handler.temp_metadata = None

                return {
                    "message": "Embeddings created and uploaded successfully. Ready for chat.",
                    "status": "completed",
                    "can_chat": True,
                }

            except Exception as e:
                logging.error(f"Error uploading embeddings to GCS: {str(e)}")
                # Clean up any partially created embeddings
                await self._cleanup_failed_embeddings(file_id)
                raise

        except Exception as e:
            logging.error(
                f"Error in create_and_upload_embeddings: {str(e)}", exc_info=True
            )
            return {
                "message": "An error occurred while processing your file.",
                "status": "error",
                "can_chat": False,
            }

    async def check_embeddings_exist(
        self, file_id: str, model_choice: str
    ) -> Dict[str, Any]:
        """
        Check if embeddings exist for a specific file and model.

        Args:
            file_id (str): The ID of the file to check
            model_choice (str): The chosen model (e.g., 'gpt-4', 'gemini-pro')

        Returns:
            Dict containing:
                - embeddings_exist (bool): Whether embeddings exist
                - model_type (str): Type of model (azure/google)
                - file_id (str): The checked file ID
        """
        try:
            model_choice = model_choice.lower()
            model_type = "azure" if "gpt" in model_choice else "google"
            folder_name = "azure" if model_type == "azure" else "google"

            # Check GCS for embeddings
            gcs_prefix = f"file-embeddings/{file_id}/{folder_name}/"
            blobs = list(self.gcs_handler.bucket.list_blobs(prefix=gcs_prefix))

            embeddings_exist = len(blobs) > 0 and any(
                blob.name.endswith("chroma.sqlite3") for blob in blobs
            )

            return {
                "embeddings_exist": embeddings_exist,
                "model_type": model_type,
                "file_id": file_id,
            }

        except Exception as e:
            logging.error(f"Error checking embeddings: {str(e)}")
            raise Exception(f"Error checking embeddings: {str(e)}")

    def _create_azure_embeddings(self, file_id: str, chunks: List[str]):
        """Creates embeddings using Azure OpenAI."""
        logging.info("Generating Azure embeddings...")
        try:
            collection_name = f"rag_collection_{file_id}"

            azure_handler = AzureChatbot(self.configs, self.gcs_handler)
            azure_handler.initialize(
                model_choice="gpt_4o_mini",
                file_id=file_id,
                embedding_type="azure",
                collection_name=collection_name,
            )

            azure_handler.create_and_store_embeddings(
                chunks, file_id, "azure", is_embedding=True
            )

            logging.info("Azure embeddings generated successfully")
            return "completed"
        except Exception as e:
            logging.error(f"Error creating Azure embeddings: {str(e)}", exc_info=True)
            raise

    def _create_gemini_embeddings(self, file_id: str, chunks: List[str]):
        """Creates embeddings using Gemini model."""
        logging.info("Generating Gemini embeddings...")
        try:
            collection_name = f"rag_collection_{file_id}"

            gemini_handler = GeminiHandler(self.configs, self.gcs_handler)
            gemini_handler.initialize(
                model="gemini-pro",
                file_id=file_id,
                embedding_type="google",
                collection_name=collection_name,
            )

            gemini_handler.create_and_store_embeddings(
                chunks, file_id, "google", is_embedding=True
            )

            logging.info("Gemini embeddings generated successfully")
            return "completed"
        except Exception as e:
            logging.error(f"Error creating Gemini embeddings: {str(e)}", exc_info=True)
            raise

    async def _upload_embeddings_to_gcs(self, file_id: str):
        logging.info("Uploading embeddings to GCS...")
        try:
            # Get the temp metadata which should already have the correct file_id
            temp_metadata = self.gcs_handler.temp_metadata

            # If we have temp_metadata and a file_id in it, use that to ensure consistency
            if temp_metadata and "file_id" in temp_metadata:
                metadata_file_id = temp_metadata.get("file_id")
                if metadata_file_id != file_id:
                    logging.warning(
                        f"Metadata file_id {metadata_file_id} differs from passed file_id "
                        f"{file_id}. Using metadata file_id."
                    )
                    file_id = metadata_file_id

            for model in ["azure", "google"]:
                chroma_db_path = f"./chroma_db/{file_id}/{model}"
                gcs_subfolder = f"file-embeddings/{file_id}/{model}"

                files_to_upload = {}
                for file in Path(chroma_db_path).rglob("*"):
                    if file.is_file():
                        relative_path = file.relative_to(chroma_db_path)
                        gcs_object_name = f"{gcs_subfolder}/{relative_path}"
                        files_to_upload[str(relative_path)] = (
                            str(file),
                            gcs_object_name,
                        )

                self.gcs_handler.upload_to_gcs(
                    self.configs.gcp_resource.bucket_name, files_to_upload
                )
            logging.info("Embeddings uploaded to GCS successfully")
        except Exception as e:
            logging.error(f"Error uploading embeddings to GCS: {str(e)}", exc_info=True)
            raise

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
