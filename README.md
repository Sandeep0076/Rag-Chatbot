# RAG PDF API

## Description
RAG PDF API is a FastAPI-based application that implements a Retrieval-Augmented Generation (RAG) system for processing and querying PDF documents and images. The system uses Azure OpenAI services, Google Cloud Vertex AI (Gemini), and GPT-4 Vision for various AI functionalities, Google Cloud Storage for file management, and Chroma as the vector database.

## Features
- PDF and image preprocessing and embedding generation
- Vector database creation and management using Chroma
- Chat-based querying of processed documents
- Integration with Azure OpenAI for embeddings and language models
- Integration with Google Cloud Vertex AI (Gemini) for advanced language processing
- GPT-4-Omni integration for image analysis
- Google Cloud Storage for file management
- FastAPI backend with Prometheus metrics
- (Optional) Streamlit-based user interface for easy interaction

## Installation

### Prerequisites
- Python 3.10 or higher
- Poetry for dependency management
- Google Cloud SDK
- Azure account with OpenAI services
- Google Cloud account with Vertex AI enabled

### Setup
1. Install Python 3.10 or higher if not already installed.

2. Install Poetry:
   ```
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. Clone the repository:
   ```
   git clone [repository-url]
   cd rtl-rag-chatbot-api
   ```

4. Configure Poetry:
   ```
   poetry config http-basic.python-packages gitlab-ci-token xxx
   ```

5. Install dependencies:
   ```
   poetry install
   ```

6. Lock dependencies:
   ```
   poetry lock
   ```
   Note: You may need to reset/checkout the poetry.lock file again from the repository if there are conflicts.

7. Add Poetry to your PATH:
   ```
   export PATH="$HOME/.local/bin:$PATH"
   ```
   Consider adding this line to your shell configuration file (e.g., .bashrc, .zshrc) for persistence.

8. Set up environment variables (see Environment Variables section)

9. Install pre-commit hooks:
   ```
   pre-commit install
   ```

## Configuration

### Environment Variables
The application requires several environment variables to be set. These are detailed in the `env-variables.md` file. Key variables include:

- Azure OpenAI Configuration
- Google Cloud Storage Configuration
- Google Cloud Vertex AI Configuration
- Application-specific variables (prefix with `RAG_PDF_API__`)

To set these variables, you can either export them in your shell or create a `.env` file in the project root.

The main configuration file is `configs/app_config.py`. You can override these configurations using the environment variables.

## Usage

### Running the Application
To start the FastAPI server:

```
poetry run start
```

The server will start on `http://0.0.0.0:8080`.

To run the Streamlit interface:

```
streamlit run streamlit_app.py
```

To run the Version logger interface:

```
python version_doc/version_logger.py
```
To run unit tests:

```
pytest tests/api_unit_tests.py
```
### API Endpoints

1. **Health Check**: GET `/health`
2. **Application Info**: GET `/info`
3. **Preprocess PDF/Image**: POST `/file/upload`
4. **Chat with PDF/Image**: POST `/file/chat`
5. **Get Nearest Neighbors**: POST `/file/neighbors`
6. **Available Models**: GET `/available-models`
7. **Cleanup Files**: POST `/file/cleanup`
8. **Initialize Model**: POST `/model/initialize`
9. **Analyze Image**: POST `/image/analyze`
10. **Metrics**: GET `/metrics`

## Architecture

The application is structured as follows:

- `app.py`: Main FastAPI application
- `streamlit_app.py`: Streamlit user interface
- `chatbot/chatbot_creator.py`: Implements the Chatbot class for RAG functionality
- `chatbot/gcs_handler.py`: Handles interactions with Google Cloud Storage
- `chatbot/gemini_handler.py`: Manages Gemini model interactions
- `chatbot/image_reader.py`: Handles image analysis using GPT-4 Omni
- `common/embeddings.py`: Manages the creation of embeddings
- `common/vector_db_creator.py`: Handles the creation and management of the Chroma vector database

## Development

### Running Tests
To run the test suite:

```
make test
```

## Deployment

The project includes Helm charts for Kubernetes deployment. Deployment configurations can be found in the `helm` directory.

## Contact

[Sandeep Pathania] - [AI-Products Team]
