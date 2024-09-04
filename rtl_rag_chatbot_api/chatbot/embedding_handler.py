import json
import logging
import os

import chromadb
from chromadb.config import Settings

from rtl_rag_chatbot_api.chatbot.gemini_handler import GeminiHandler
from rtl_rag_chatbot_api.chatbot.image_reader import analyze_images
from rtl_rag_chatbot_api.common.embeddings import run_preprocessor


class EmbeddingHandler:
    """
    Initialize the EmbeddingHandler with configurations and GCS handler.

    Create and upload embeddings for a specified file based on the model choice and whether it's an image file.

    Args:
        file_id (str): Unique identifier for the file.
        model_choice (str): Choice of the model for processing.
        is_image (bool): Indicates if the file is an image.

    Returns:
        dict: Contains information about the processed file, model choice, and whether it's an image.
    """

    def __init__(self, configs, gcs_handler):
        self.configs = configs
        self.gcs_handler = gcs_handler

    def create_and_upload_embeddings(
        self, file_id: str, model_choice: str, is_image: bool
    ):
        chroma_db_path = f"./chroma_db/{file_id}"
        os.makedirs(chroma_db_path, exist_ok=True)

        chroma_db = chromadb.PersistentClient(
            path=chroma_db_path,
            settings=Settings(allow_reset=True, is_persistent=True),
        )

        destination_file_path = f"local_data/{file_id}/"
        os.makedirs(destination_file_path, exist_ok=True)

        decrypted_file_path = self.gcs_handler.download_and_decrypt_file(
            file_id, destination_file_path
        )

        try:
            # Verify that the file is readable
            with open(decrypted_file_path, "rb") as test_file:
                test_content = test_file.read(1024)  # Read first 1KB to test
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

            if model_choice.lower() in ["gemini-flash", "gemini-pro"]:
                gemini_handler = GeminiHandler(self.configs, self.gcs_handler)
                gemini_handler.process_file(
                    file_id, decrypted_file_path if not is_image else analysis_json_path
                )
                gemini_handler.upload_embeddings_to_gcs(file_id)
                logging.info("Embeddings generated via Gemini")
            else:
                # Use run_preprocessor for Azure models
                run_preprocessor(
                    configs=self.configs,
                    text_data_folder_path=destination_file_path,
                    file_id=file_id,
                    chroma_db_path=chroma_db_path,
                    chroma_db=chroma_db,
                    is_image=is_image,
                    gcs_handler=self.gcs_handler,
                )
                logging.info("Embeddings generated via Azure")

        finally:
            # Clean up the decrypted file and analysis JSON if they exist
            if os.path.exists(decrypted_file_path):
                os.remove(decrypted_file_path)
            if is_image:
                analysis_json_path = os.path.join(
                    destination_file_path, f"{file_id}_analysis.json"
                )
                if os.path.exists(analysis_json_path):
                    os.remove(analysis_json_path)

        return {"file_id": file_id, "model_choice": model_choice, "is_image": is_image}
