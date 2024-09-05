from rtl_rag_chatbot_api.chatbot.chatbot_creator import Chatbot
from rtl_rag_chatbot_api.chatbot.gemini_handler import GeminiHandler


class ModelHandler:
    def __init__(self, configs, gcs_handler):
        self.configs = configs
        self.gcs_handler = gcs_handler

    def initialize_model(self, model_choice: str, file_id: str, embedding_type: str):
        if embedding_type == "gemini":
            gemini_model = (
                self.configs.gemini.model_flash
                if model_choice.lower() == "gemini-flash"
                else self.configs.gemini.model_pro
            )
            model = GeminiHandler(self.configs, self.gcs_handler)
            model.initialize(
                model=gemini_model, file_id=file_id, embedding_type=embedding_type
            )
        else:
            model = Chatbot(
                self.configs,
                file_id=file_id,
                model_choice=model_choice,
                embedding_type=embedding_type,
            )

        return model
