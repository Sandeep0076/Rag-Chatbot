import logging
import os

from rtl_rag_chatbot_api.chatbot.chatbot_creator import AzureChatbot
from rtl_rag_chatbot_api.chatbot.gemini_handler import GeminiHandler
from rtl_rag_chatbot_api.common.chroma_manager import ChromaDBManager


class ModelHandler:
    def __init__(self, configs, gcs_handler):
        self.configs = configs
        self.gcs_handler = gcs_handler
        self.chroma_manager = ChromaDBManager()

    def initialize_model(
        self, model_choice: str, file_id: str, embedding_type: str, user_id: str = None
    ):
        """Initialize and return a model based on the specified model choice."""
        try:
            chroma_db_path = f"./chroma_db/{file_id}/{embedding_type}"
            os.makedirs(chroma_db_path, exist_ok=True)

            if not os.path.exists(chroma_db_path):
                self.gcs_handler.download_files_from_folder_by_id(file_id)

            collection_name = f"rag_collection_{file_id}"

            if model_choice.lower() in ["gemini-flash", "gemini-pro"]:
                model = GeminiHandler(self.configs, self.gcs_handler)
                model.initialize(
                    model=model_choice,
                    file_id=file_id,
                    embedding_type=embedding_type,
                    collection_name=collection_name,
                    user_id=user_id,
                )
            else:
                model = AzureChatbot(self.configs, self.gcs_handler)
                model.initialize(
                    model_choice=model_choice,
                    file_id=file_id,
                    embedding_type=embedding_type,
                    collection_name=collection_name,
                    user_id=user_id,
                )

            return model
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            raise
