# RAG PDF API

## Description
RAG PDF API is a FastAPI-based application that implements a Retrieval-Augmented Generation (RAG) system for processing and querying PDF documents. The system uses Azure OpenAI services for embeddings and language models, Google Cloud Storage for file management, and Chroma as the vector database.

## Features
- PDF preprocessing and embedding generation
- Vector database creation and management
- Chat-based querying of processed documents
- Integration with Azure OpenAI for embeddings and language models
- Google Cloud Storage for file management
- FastAPI backend with Prometheus metrics

## Installation

### Prerequisites
- Python 3.10 or higher
- Poetry for dependency management
- Google Cloud SDK
- Azure account with OpenAI services

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
The application requires several environment variables to be set. These are detailed in the `env-variables.md` file. Here's a summary of the required variables:

- Azure OpenAI Configuration:
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_API_VERSION`

- Google Cloud Storage Configuration:
  - `GOOGLE_APPLICATION_CREDENTIALS` (path to your GCP credentials file)

- Application-specific variables (prefix with `RAG_PDF_API__`):
  - `RAG_PDF_API__AZURE_EMBEDDING__AZURE_EMBEDDING_API_KEY`
  - `RAG_PDF_API__AZURE_EMBEDDING__AZURE_EMBEDDING_ENDPOINT`
  - `RAG_PDF_API__AZURE_EMBEDDING__AZURE_EMBEDDING_API_VERSION`
  - `RAG_PDF_API__AZURE_EMBEDDING__AZURE_EMBEDDING_DEPLOYMENT`
  - `RAG_PDF_API__AZURE_EMBEDDING__AZURE_EMBEDDING_MODEL_NAME`
  - (Add other variables as specified in env-variables.md)

To set these variables, you can either export them in your shell or create a `.env` file in the project root.

The main configuration file is `configs/app_config.py`. You can override these configurations using the environment variables listed above.

## Usage

### Running the Application
To start the FastAPI server:

```
poetry run start
```

The server will start on `http://0.0.0.0:8080`.

### API Endpoints

1. **Health Check**
   - GET `/health`
   - Returns the health status of the application

2. **Application Info**
   - GET `/info`
   - Returns basic information about the application

3. **Preprocess PDF**
   - POST `/pdf/preprocess`
   - Preprocesses a PDF file and creates embeddings
   - Request body: `{"file_id": "string"}`

4. **Chat with PDF**
   - POST `/pdf/chat`
   - Allows querying the processed PDF using natural language
   - Request body: `{"text": "string", "file_id": "string", "model_choice": "string"}`

5. **Available Models**
   - GET `/available-models`
   - Returns a list of available language models

6. **Metrics**
   - GET `/metrics`
   - Exposes Prometheus metrics

## Architecture

The application is structured as follows:

- `app.py`: Main FastAPI application
- `chatbot/chatbot_creator.py`: Implements the Chatbot class for RAG functionality
- `chatbot/gcs_handler.py`: Handles interactions with Google Cloud Storage
- `common/embeddings.py`: Manages the creation of embeddings
- `common/vector_db_creator.py`: Handles the creation and management of the vector database

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
