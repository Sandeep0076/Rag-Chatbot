import os

from rtl_rag_chatbot_api.chatbot.chatbot_creator import Chatbot
from rtl_rag_chatbot_api.chatbot.gemini_handler import GeminiHandler


class ModelHandler:
    """
    Initializes and returns a model based on the specified model choice, file ID, and embedding type.

    Args:
        model_choice (str): The choice of the model to initialize.
        file_id (str): Identifier for the file being processed.
        embedding_type (str): Type of embedding to use.

    Returns:
        GeminiHandler or Chatbot: An instance of GeminiHandler or Chatbot based on the embedding type.
    """

    def __init__(self, configs, gcs_handler):
        self.configs = configs
        self.gcs_handler = gcs_handler

    def initialize_model(self, model_choice: str, file_id: str, embedding_type: str):
        """
        Initializes and returns a model based on the specified model choice, file ID, and embedding type.

        Args:
            model_choice (str): The choice of the model to initialize.
            file_id (str): Identifier for the file being processed.
            embedding_type (str): Type of embedding to use.

        Returns:
            GeminiHandler or Chatbot: An instance of GeminiHandler or Chatbot based on the embedding type.
        """
        chroma_db_path = f"./chroma_db/{file_id}/{embedding_type}"

        if not os.path.exists(chroma_db_path):
            os.makedirs(chroma_db_path)
            self.gcs_handler.download_files_from_folder_by_id(file_id)

        # Create unique collection name for this file
        collection_name = f"rag_collection_{file_id}"

        if model_choice.lower() in ["gemini-flash", "gemini-pro"]:
            gemini_model = (
                self.configs.gemini.model_flash
                if model_choice.lower() == "gemini-flash"
                else self.configs.gemini.model_pro
            )
            model = GeminiHandler(self.configs, self.gcs_handler)
            model.initialize(
                model=gemini_model,
                file_id=file_id,
                embedding_type=embedding_type,
                collection_name=collection_name,
            )
        else:
            model = Chatbot(
                self.configs,
                file_id=file_id,
                model_choice=model_choice,
                embedding_type=embedding_type,
                collection_name=collection_name,
            )

        return model
