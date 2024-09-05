# import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chromadb
from chromadb.config import Settings

from rtl_rag_chatbot_api.chatbot.gemini_handler import GeminiHandler
from rtl_rag_chatbot_api.chatbot.image_reader import analyze_images
from rtl_rag_chatbot_api.common.embeddings import run_preprocessor


class EmbeddingHandler:
    def __init__(self, configs, gcs_handler):
        self.configs = configs
        self.gcs_handler = gcs_handler

    def embeddings_exist(self, file_id: str) -> bool:
        azure_path = f"./chroma_db/{file_id}/azure"
        gemini_path = f"./chroma_db/{file_id}/gemini"
        return (os.path.exists(azure_path) and len(os.listdir(azure_path)) > 0) and (
            os.path.exists(gemini_path) and len(os.listdir(gemini_path)) > 0
        )

    async def create_and_upload_embeddings(self, file_id: str, is_image: bool):
        if self.embeddings_exist(file_id):
            logging.info(f"Embeddings already exist for file_id: {file_id}")
            return {"message": "Embeddings already exist for this file"}

        chroma_db_path = f"./chroma_db/{file_id}"
        os.makedirs(os.path.join(chroma_db_path, "azure"), exist_ok=True)
        os.makedirs(os.path.join(chroma_db_path, "gemini"), exist_ok=True)

        destination_file_path = f"local_data/{file_id}/"
        os.makedirs(destination_file_path, exist_ok=True)

        decrypted_file_path = self.gcs_handler.download_and_decrypt_file(
            file_id, destination_file_path
        )

        try:
            with open(decrypted_file_path, "rb") as test_file:
                test_content = test_file.read(1024)
            if not test_content:
                raise IOError(
                    f"Decrypted file at {decrypted_file_path} appears to be empty or unreadable"
                )

            logging.info(f"Successfully verified decrypted file: {decrypted_file_path}")

            if is_image:
                logging.info("Processing image file...")
                image_analysis_result = analyze_images(decrypted_file_path)
                analysis_json_path = os.path.join(
                    destination_file_path, f"{file_id}_analysis.json"
                )
                with open(analysis_json_path, "w") as f:
                    json.dump(image_analysis_result, f)
                logging.info(f"Image analysis result saved to {analysis_json_path}")

            # Create embeddings for both Azure and Gemini in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                azure_future = executor.submit(
                    self._create_azure_embeddings,
                    file_id,
                    decrypted_file_path if not is_image else analysis_json_path,
                )
                gemini_future = executor.submit(
                    self._create_gemini_embeddings,
                    file_id,
                    decrypted_file_path if not is_image else analysis_json_path,
                )

                azure_result = azure_future.result()
                gemini_result = gemini_future.result()

            # Upload embeddings to GCS
            self._upload_embeddings_to_gcs(file_id)

            file_info = {
                "embeddings": {"azure": azure_result, "gemini": gemini_result},
                "is_image": is_image,
            }
            self.gcs_handler.upload_to_gcs(
                self.configs.gcp_resource.bucket_name,
                {
                    "file_info": (
                        file_info,
                        f"file-embeddings/{file_id}/file_info.json",
                    )
                },
            )

            return {
                "message": "Embeddings created and uploaded successfully for both Azure and Gemini"
            }

        finally:
            if os.path.exists(decrypted_file_path):
                os.remove(decrypted_file_path)
            if is_image:
                analysis_json_path = os.path.join(
                    destination_file_path, f"{file_id}_analysis.json"
                )
                if os.path.exists(analysis_json_path):
                    os.remove(analysis_json_path)

    def _create_azure_embeddings(self, file_id, file_path):
        logging.info("Generating Azure embeddings...")
        chroma_db = chromadb.PersistentClient(
            path=f"./chroma_db/{file_id}/azure",
            settings=Settings(allow_reset=True, is_persistent=True),
        )
        run_preprocessor(
            configs=self.configs,
            text_data_folder_path=os.path.dirname(file_path),
            file_id=file_id,
            chroma_db_path=f"./chroma_db/{file_id}/azure",
            chroma_db=chroma_db,
            is_image=False,
            gcs_handler=self.gcs_handler,
        )
        logging.info("Azure embeddings generated successfully")
        return "completed"

    def _create_gemini_embeddings(self, file_id, file_path):
        logging.info("Generating Gemini embeddings...")
        gemini_handler = GeminiHandler(self.configs, self.gcs_handler)
        gemini_handler.process_file(file_id, file_path, subfolder="gemini")
        logging.info("Gemini embeddings generated successfully")
        return "completed"

    def _upload_embeddings_to_gcs(self, file_id):
        logging.info("Uploading embeddings to GCS...")
        for model in ["azure", "gemini"]:
            chroma_db_path = f"./chroma_db/{file_id}/{model}"
            gcs_subfolder = f"file-embeddings/{file_id}/{model}"

            files_to_upload = {}
            for file in Path(chroma_db_path).rglob("*"):
                if file.is_file():
                    relative_path = file.relative_to(chroma_db_path)
                    gcs_object_name = f"{gcs_subfolder}/{relative_path}"
                    files_to_upload[str(relative_path)] = (str(file), gcs_object_name)

            self.gcs_handler.upload_to_gcs(
                self.configs.gcp_resource.bucket_name, files_to_upload
            )
        logging.info("Embeddings uploaded to GCS successfully")

    def get_embeddings_info(self, file_id: str):
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
