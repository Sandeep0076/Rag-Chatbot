import sys
import warnings

import streamlit.web.cli as stcli

# Suppress warnings from google-cloud-aiplatform and vertexai
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="google.cloud.aiplatform"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="vertexai._model_garden._model_garden_models"
)


def run():
    sys.argv = ["streamlit", "run", "streamlit_app.py"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    run()
