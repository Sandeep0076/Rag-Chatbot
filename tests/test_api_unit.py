import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from rtl_rag_chatbot_api.app import app

# Initialize test client
client = TestClient(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# Configure logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('test_delete_files.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("test_delete_files.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
client = TestClient(app)


@pytest.fixture
def mock_config():
    mock_config = MagicMock()
    mock_config.gemini = MagicMock()
    mock_config.gemini.model_pro = "gemini-pro"
    mock_config.gemini.model_flash = "gemini-flash"
    return mock_config


@pytest.fixture
def mock_gcs():
    with patch("rtl_rag_chatbot_api.app.gcs_handler") as mock:
        mock.get_file_info.return_value = {
            "original_filename": "test.pdf",
            "embeddings_status": "pending",
        }
        yield mock


@pytest.fixture
def test_file_setup():
    # Create a simple test file
    test_content = "This is a test document for chat testing."
    file_path = "local_data/test_file.txt"
    os.makedirs("local_data", exist_ok=True)
    with open(file_path, "w") as f:
        f.write(test_content)
    return file_path


@pytest.fixture
def mock_gcs_handler():
    with patch("rtl_rag_chatbot_api.chatbot.gcs_handler.GCSHandler") as mock:
        # Mock file info response
        mock.return_value.get_file_info.return_value = {
            "file_id": "test_file_123",
            "embeddings_status": "completed",
            "is_image": False,
        }
        yield mock


@pytest.fixture
def mock_chroma_manager():
    with patch("rtl_rag_chatbot_api.common.chroma_manager.ChromaDBManager") as mock:
        # Mock ChromaDB responses
        mock.return_value.get_collection.return_value.query.return_value = {
            "documents": [["This is a relevant context for the query."]]
        }
        yield mock


def test_chat(test_file_setup, mock_gcs_handler, mock_chroma_manager):
    # First upload a file
    with open(test_file_setup, "rb") as f:
        files = {"file": ("test_file.txt", f, "text/plain")}
        upload_response = client.post(
            "/file/upload",
            files=files,
            data={"is_image": "false", "username": "test_user"},
        )
        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]

    # Now test the chat endpoint
    test_data = {
        "text": ["Hello, can you summarize the document?"],
        "file_id": file_id,
        "model_choice": "gpt_4o_mini",
        "user_id": "test_user",
    }

    response = client.post("/file/chat", json=test_data)

    assert response.status_code == 200
    assert "response" in response.json()

    # Cleanup
    if os.path.exists(test_file_setup):
        os.remove(test_file_setup)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "up"}


def test_info():
    response = client.get("/info")
    assert response.status_code == 200
    assert "title" in response.json()
    assert "description" in response.json()


@patch("rtl_rag_chatbot_api.app.uuid.uuid4")
@patch("rtl_rag_chatbot_api.app.file_handler.process_file")
async def test_file_upload(mock_process_file, mock_uuid, mock_chroma_manager):
    mock_uuid.return_value = "test_file_id"
    mock_process_file.return_value = {
        "file_id": "test_file_id",
        "status": "new",
        "message": "File processed successfully",
        "is_image": False,
        "temp_file_path": "local_data/test_file_id_test.pdf",
    }

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/file/upload",
            files={"file": ("test.pdf", b"test content", "application/pdf")},
            data={"is_image": "false", "username": "testuser"},
        )

    assert response.status_code == 200
    assert response.json()["file_id"] == "test_file_id"


@pytest.mark.asyncio
async def test_create_embeddings(mock_gcs):
    with patch("rtl_rag_chatbot_api.app.os.path.exists", return_value=True), patch(
        "rtl_rag_chatbot_api.app.EmbeddingHandler"
    ) as MockEmbeddingHandler:
        mock_handler = MockEmbeddingHandler.return_value
        mock_handler.create_and_upload_embeddings = AsyncMock(
            return_value={"message": "Embeddings created successfully"}
        )

        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post(
                "/embeddings/create",
                json={"file_id": "test_file_id", "is_image": False},
            )

        assert response.status_code == 200
        assert response.json() == {"message": "Embeddings created successfully"}


# @pytest.mark.asyncio
# async def test_chat_with_tabular_data():
#     with patch("rtl_rag_chatbot_api.app.os.path.exists") as mock_exists, patch(
#         "rtl_rag_chatbot_api.app.TabularDataHandler"
#     ) as MockTabularHandler, patch("rtl_rag_chatbot_api.app.gcs_handler") as mock_gcs:
#         mock_exists.return_value = True  # Make it find the tabular data file
#         mock_handler = MockTabularHandler.return_value
#         mock_handler.get_answer.return_value = "SQL query response"
#         mock_gcs.get_file_info.return_value = {"embeddings_status": "completed"}

#         async with AsyncClient(app=app, base_url="http://test") as ac:
#             response = await ac.post(
#                 "/file/chat",
#                 json={
#                     "text": "Show me sales data",
#                     "file_id": "test_csv_id",
#                     "model_choice": "gpt-4o-mini",
#                 },
#             )

#         assert response.status_code == 200
#         assert response.json() == {"response": "SQL query response"}


# @pytest.mark.asyncio
# async def test_get_neighbors(mock_chroma_manager):
#     with patch("rtl_rag_chatbot_api.app.initialized_models") as mock_models:
#         mock_model = MagicMock()
#         mock_model.get_n_nearest_neighbours.return_value = ["Neighbor 1", "Neighbor 2"]
#         mock_models.__getitem__.return_value = mock_model
#         mock_models.__contains__.return_value = True

#         async with AsyncClient(app=app, base_url="http://test") as ac:
#             response = await ac.post(
#                 "/file/neighbors",
#                 json={
#                     "text": "Test query",
#                     "file_id": "test_file_id",
#                     "n_neighbors": 2,
#                 },
#             )

#         assert response.status_code == 200
#         assert response.json() == {"neighbors": ["Neighbor 1", "Neighbor 2"]}


@pytest.mark.asyncio
async def test_cleanup(mock_chroma_manager):
    with patch("rtl_rag_chatbot_api.app.CleanupCoordinator") as MockCleanupCoordinator:
        mock_coordinator = MockCleanupCoordinator.return_value
        mock_coordinator.cleanup = MagicMock()

        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/file/cleanup")

        assert response.status_code == 200
        assert response.json() == {"status": "Cleanup completed successfully"}
        mock_coordinator.cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_gemini_chat():
    async def mock_stream():
        yield "Test response"

    with patch("rtl_rag_chatbot_api.app.ModelHandler") as MockModelHandler:
        mock_model = MagicMock()
        mock_model.get_gemini_response_stream.return_value = mock_stream()
        MockModelHandler.return_value.initialize_model.return_value = mock_model

        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post(
                "/chat/gemini", json={"model": "gemini-pro", "message": "Test message"}
            )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
