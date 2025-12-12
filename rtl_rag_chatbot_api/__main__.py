# Disable ChromaDB telemetry before any imports
import warnings

import uvicorn

import configs.chromadb_telemetry_fix  # noqa: F401
from rtl_rag_chatbot_api.app import app

# Suppress warnings from google-cloud-aiplatform and vertexai
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="google.cloud.aiplatform"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="vertexai._model_garden._model_garden_models"
)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
