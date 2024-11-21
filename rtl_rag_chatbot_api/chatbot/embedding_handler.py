import json
import logging
import os
from pathlib import Path
from typing import List

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
            # Instead of looking for file_info, check temp_metadata first
            temp_metadata = self.gcs_handler.temp_metadata
            if not temp_metadata:
                raise ValueError(f"No metadata found for file_id: {file_id}")

            # Check embeddings status using the existing embeddings_exist function
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

            # Only create new embeddings if they don't exist anywhere
            if not gcs_status:
                if not temp_file_path:
                    original_filename = temp_metadata.get("original_filename")
                    temp_file_path = f"local_data/{file_id}_{original_filename}"

                if not os.path.exists(temp_file_path):
                    raise FileNotFoundError(
                        f"Source file not found at {temp_file_path}"
                    )

                logging.info(f"Creating new embeddings for file_id: {file_id}")
                return await self.create_and_upload_embeddings(
                    file_id, temp_metadata.get("is_image", False), temp_file_path
                )

            return {
                "message": "Embeddings status check completed",
                "status": "checked",
            }

        except Exception as e:
            logging.error(f"Error in ensure_embeddings_exist: {str(e)}")
            raise

    async def create_and_upload_embeddings(
        self, file_id: str, is_image: bool, temp_file_path: str
    ):
        try:
            # Get temporary metadata
            temp_metadata = self.gcs_handler.temp_metadata
            username = temp_metadata.get("username", "Unknown")

            base_handler = BaseRAGHandler(self.configs, self.gcs_handler)
            text = base_handler.extract_text_from_file(temp_file_path)
            if text.startswith("ERROR:"):
                return {"message": text[7:], "status": "error"}

            chunks = base_handler.split_text(text)
            if not chunks:
                return {
                    "message": "No processable text found in the document.",
                    "status": "error",
                }

            # Log the chunk sizes for debugging
            for i, chunk in enumerate(chunks):
                tokens = len(base_handler.simple_tokenize(chunk))
                logging.info(f"Chunk {i}: {tokens} tokens")

            logging.info(f"Text extracted and split into {len(chunks)} chunks")

            # Create embeddings
            azure_result = self._create_azure_embeddings(
                file_id,
                chunks,
                self.configs.azure_embedding.azure_embedding_api_key,
                username,
            )
            gemini_result = self._create_gemini_embeddings(file_id, chunks, username)

            try:
                # Upload embeddings to GCS
                await self._upload_embeddings_to_gcs(file_id)

                # Only create file_info.json after successful upload
                file_info = {
                    **temp_metadata,  # Include original metadata
                    "embeddings": {
                        "azure": azure_result,
                        "gemini": gemini_result,
                    },
                    "embeddings_status": "completed",
                    "azure_ready": True,
                }

                # Save file_info.json
                self.gcs_handler.upload_to_gcs(
                    self.configs.gcp_resource.bucket_name,
                    {
                        "metadata": (
                            file_info,
                            f"file-embeddings/{file_id}/file_info.json",
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

    def _create_azure_embeddings(
        self, file_id: str, chunks: List[str], api_key: str, username: str
    ):
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

    def _create_gemini_embeddings(self, file_id: str, chunks: List[str], username: str):
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
