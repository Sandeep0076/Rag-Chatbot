# import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chromadb
from chromadb.config import Settings

from rtl_rag_chatbot_api.chatbot.gemini_handler import GeminiHandler
from rtl_rag_chatbot_api.common.embeddings import run_preprocessor


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

    def embeddings_exist(self, file_id: str) -> tuple[bool, bool, bool]:
        """
        Check if embeddings exist and are valid both in GCS and locally.
        Returns tuple of (gcs_status, local_status, all_valid)
        """
        # Check GCS embeddings status
        file_info = self.gcs_handler.get_file_info(file_id)
        gcs_status = file_info.get("embeddings_status") == "completed"

        # Check local embeddings
        azure_path = f"./chroma_db/{file_id}/azure"
        gemini_path = f"./chroma_db/{file_id}/google"

        local_status = (
            os.path.exists(azure_path)
            and len(os.listdir(azure_path)) > 0
            and os.path.exists(gemini_path)
            and len(os.listdir(gemini_path)) > 0
        )

        return gcs_status, local_status, (gcs_status and local_status)

    async def ensure_embeddings_exist(self, file_id: str, temp_file_path: str = None):
        """
        Ensures embeddings exist and are valid by:
        1. Checking in chroma_db/{file_id}/ first
        2. If not found but hash matches, download from GCP
        3. If not in GCP, create and upload new embeddings
        """
        try:
            file_info = self.gcs_handler.get_file_info(file_id)
            if not file_info:
                raise ValueError(f"No file info found for file_id: {file_id}")

            # Check embeddings status
            gcs_status, local_status, all_valid = self.embeddings_exist(file_id)

            if all_valid:
                return {
                    "message": "Embeddings already exist and are valid",
                    "status": "existing",
                }

            # If embeddings don't exist locally but exist in GCS, download them
            if gcs_status and not local_status:
                self.gcs_handler.download_files_from_folder_by_id(file_id)
                return {
                    "message": "Embeddings downloaded successfully",
                    "status": "downloaded",
                }

            # Create new embeddings
            result = await self.create_and_upload_embeddings(
                file_id, file_info.get("is_image", False), temp_file_path
            )

            return result

        except Exception as e:
            logging.error(f"Error in ensure_embeddings_exist: {str(e)}", exc_info=True)
            raise

    async def create_and_upload_embeddings(
        self, file_id: str, is_image: bool, temp_file_path: str
    ):
        """Create embeddings for both Azure and Gemini models and upload them to GCS"""
        try:
            # Get username from file info
            file_info = self.gcs_handler.get_file_info(file_id)
            username = file_info.get("username", "Unknown")

            # Create new embeddings for both Azure and Gemini
            with ThreadPoolExecutor(max_workers=2) as executor:
                azure_future = executor.submit(
                    self._create_azure_embeddings,
                    file_id,
                    temp_file_path,
                    self.configs.azure_embedding.azure_embedding_api_key,
                    username,  # Pass username here
                )
                gemini_future = executor.submit(
                    self._create_gemini_embeddings,
                    file_id,
                    temp_file_path,
                    username,  # Pass same username here
                )

                azure_result = azure_future.result()
                gemini_result = gemini_future.result()

            # Upload embeddings to GCS
            self._upload_embeddings_to_gcs(file_id)

            # Update file info with embedding status
            new_info = {
                "embeddings": {"azure": azure_result, "google": gemini_result},
                "embeddings_status": "completed",
            }
            self.gcs_handler.update_file_info(file_id, new_info)

            return {
                "message": "Embeddings created and uploaded successfully",
                "status": "completed",
            }

        except Exception as e:
            logging.error(
                f"Error in create_and_upload_embeddings: {str(e)}", exc_info=True
            )
            raise

    def _create_azure_embeddings(
        self, file_id: str, file_path: str, api_key: str, username: str
    ):
        logging.info("Generating Azure embeddings...")
        try:
            chroma_db_path = f"./chroma_db/{file_id}/azure"
            os.makedirs(chroma_db_path, exist_ok=True)

            chroma_db = chromadb.PersistentClient(
                path=chroma_db_path,
                settings=Settings(allow_reset=True, is_persistent=True),
            )
            collection_name = f"rag_collection_{file_id}"
            if collection_name in [col.name for col in chroma_db.list_collections()]:
                chroma_db.delete_collection(collection_name)

            run_preprocessor(
                configs=self.configs,
                text_data_folder_path=os.path.dirname(file_path),
                file_id=file_id,
                chroma_db_path=chroma_db_path,
                chroma_db=chroma_db,
                is_image=False,
                gcs_handler=self.gcs_handler,
                username=username,
                collection_name=collection_name,
            )
            logging.info("Azure embeddings generated successfully")
            logging.info(f"{collection_name} collection is being used")
            return "completed"
        except Exception as e:
            logging.error(f"Error creating Azure embeddings: {str(e)}", exc_info=True)
            raise

    def _create_gemini_embeddings(self, file_id: str, file_path: str, username: str):
        logging.info("Generating Gemini embeddings...")
        try:
            chroma_db_path = f"./chroma_db/{file_id}/google"
            os.makedirs(chroma_db_path, exist_ok=True)
            collection_name = f"rag_collection_{file_id}"
            gemini_handler = GeminiHandler(self.configs, self.gcs_handler)
            gemini_handler.process_file(
                file_id, file_path, subfolder="google", collection_name=collection_name
            )
            logging.info("Gemini embeddings generated successfully")
            logging.info(f"{collection_name} collection is being used")
            return "completed"
        except Exception as e:
            logging.error(f"Error creating Gemini embeddings: {str(e)}", exc_info=True)
            raise

    def _upload_embeddings_to_gcs(self, file_id: str):
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
