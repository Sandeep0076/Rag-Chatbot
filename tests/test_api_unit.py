import logging
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from rtl_rag_chatbot_api.app import app

# Initialize test client
client = TestClient(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Mock files directory
MOCK_FILES_DIR = os.path.join(os.path.dirname(__file__), "mock_files")


@pytest.fixture
def mock_files() -> dict:
    """Fixture providing paths to mock files for testing."""
    return {
        "pdf": os.path.join(MOCK_FILES_DIR, "mock_file.pdf"),
        "csv": os.path.join(MOCK_FILES_DIR, "mock_file.csv"),
        "excel": os.path.join(MOCK_FILES_DIR, "mock_file.xlsx"),
        "image": os.path.join(MOCK_FILES_DIR, "mock_file.png"),
        "db": os.path.join(MOCK_FILES_DIR, "mock_file.sqlite"),
    }


@pytest.fixture
def mock_config():
    mock_config = MagicMock()
    mock_config.gemini = MagicMock()
    mock_config.gemini.model_pro = "gemini-pro"
    mock_config.gemini.model_pro_vision = "gemini-pro-vision"
    mock_config.gemini.api_key = "test-api-key"
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


@pytest.fixture
def mock_pdf_processor():
    with patch("rtl_rag_chatbot_api.chatbot.file_handler.FileHandler") as mock:
        mock.return_value.process_pdf.return_value = True
        yield mock


@pytest.fixture
def mock_csv_processor():
    with patch("rtl_rag_chatbot_api.chatbot.csv_handler.CSVHandler") as mock:
        mock.return_value.process_csv.return_value = {
            "columns": ["pregnancies", "glucose"],
            "preview": "data preview",
        }
        yield mock


@pytest.fixture
def mock_image_processor():
    with patch("rtl_rag_chatbot_api.chatbot.image_reader.ImageReader") as mock:
        mock.return_value.process_image.return_value = "Image analysis result"
        yield mock


@pytest.fixture
def mock_tabular_handler():
    with patch("rtl_rag_chatbot_api.chatbot.csv_handler.TabularDataHandler") as mock:
        mock.return_value.initialize_database.return_value = None
        mock.return_value.get_answer.return_value = (
            "The average number of pregnancies is 3.845"
        )
        yield mock


@pytest.fixture
def mock_image_analyzer():
    with patch("rtl_rag_chatbot_api.chatbot.image_reader.analyze_images") as mock:
        mock.return_value = {"response": "Image analysis result", "is_table": False}
        yield mock


def test_chat_with_pdf(
    mock_files: dict,
    mock_gcs_handler: MagicMock,
    mock_chroma_manager: MagicMock,
    mock_pdf_processor: MagicMock,
) -> None:
    """Test chat functionality with PDF files."""
    # Upload PDF file
    with open(mock_files["pdf"], "rb") as f:
        files = {"file": ("mock_file.pdf", f, "application/pdf")}
        upload_response = client.post(
            "/file/upload",
            files=files,
            data={"is_image": "false", "username": "test_user"},
        )
        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]

    # Test chat with PDF using GPT-4
    chat_data = {
        "text": ["Who is main character of the story"],
        "file_id": file_id,
        "model_choice": "gpt_4o_mini",
        "user_id": "test_user",
    }
    response = client.post("/file/chat", json=chat_data)
    assert response.status_code == 200
    assert "response" in response.json()
    assert "Aladdin" in response.json()["response"]
    assert not response.json().get("is_table", False)

    # Test chat with PDF using Gemini Pro
    chat_data_gemini = {
        "text": ["Who is main character of the story"],
        "file_id": file_id,
        "model_choice": "gemini-pro",
        "user_id": "test_user",
    }
    response_gemini = client.post("/file/chat", json=chat_data_gemini)
    assert response_gemini.status_code == 200
    assert "response" in response_gemini.json()
    assert "Aladdin" in response_gemini.json()["response"]
    assert not response_gemini.json().get("is_table", False)


def test_chat_with_csv(
    mock_files: dict, mock_gcs_handler: MagicMock, mock_tabular_handler: MagicMock
) -> None:
    """Test chat functionality with CSV files."""
    # Upload CSV file
    with open(mock_files["csv"], "rb") as f:
        files = {"file": ("mock_file.csv", f, "text/csv")}
        upload_response = client.post(
            "/file/upload",
            files=files,
            data={"is_image": "false", "username": "test_user"},
        )
        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]

    # Test chat with CSV using GPT-4
    chat_data = {
        "text": ["Show me the average of pregnancies"],
        "file_id": file_id,
        "model_choice": "gpt_4o_mini",
        "user_id": "test_user",
    }
    response = client.post("/file/chat", json=chat_data)
    assert response.status_code == 200
    assert "response" in response.json()
    response_text = response.json()["response"]
    assert any(
        value in response_text for value in ["3.85", "3.845"]
    ), f"Expected value '3.85' or '3.845' not found in response: {response_text}"
    assert not response.json().get("is_table", False)

    # Test chat with CSV using Gemini Pro
    chat_data_gemini = {
        "text": ["Show me the average of pregnancies"],
        "file_id": file_id,
        "model_choice": "gemini-pro",
        "user_id": "test_user",
    }
    response_gemini = client.post("/file/chat", json=chat_data_gemini)
    assert response_gemini.status_code == 200
    assert "response" in response_gemini.json()
    response_text_gemini = response_gemini.json()["response"]
    assert any(
        value in response_text_gemini for value in ["3.85", "3.845"]
    ), f"Expected value '3.85' or '3.845' not found in response: {response_text_gemini}"
    assert not response_gemini.json().get("is_table", False)


def test_chat_with_excel(
    mock_files: dict, mock_gcs_handler: MagicMock, mock_tabular_handler: MagicMock
) -> None:
    """Test chat functionality with Excel files."""
    # Upload Excel file
    with open(mock_files["excel"], "rb") as f:
        files = {
            "file": (
                "mock_file.xlsx",
                f,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        }
        upload_response = client.post(
            "/file/upload",
            files=files,
            data={"is_image": "false", "username": "test_user"},
        )
        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]

    # Test chat with Excel
    chat_data = {
        "text": ["What is the maximum value in col1?"],
        "file_id": file_id,
        "model_choice": "gpt_4o_mini",
        "user_id": "test_user",
    }
    response = client.post("/file/chat", json=chat_data)
    assert response.status_code == 200
    assert "response" in response.json()
    assert response.json().get("is_table") is True


def test_chat_with_db(
    mock_files: dict, mock_gcs_handler: MagicMock, mock_tabular_handler: MagicMock
) -> None:
    """Test chat functionality with database files."""
    # Upload DB file
    with open(mock_files["db"], "rb") as f:
        files = {"file": ("mock_file.sqlite", f, "application/x-sqlite3")}
        upload_response = client.post(
            "/file/upload",
            files=files,
            data={"is_image": "false", "username": "test_user"},
        )
        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]

    # Test chat with DB
    chat_data = {
        "text": ["Show me all records from the test table"],
        "file_id": file_id,
        "model_choice": "gpt_4o_mini",
        "user_id": "test_user",
    }
    response = client.post("/file/chat", json=chat_data)
    assert response.status_code == 200
    assert "response" in response.json()
    assert response.json().get("is_table") is True


def test_chat_with_image(
    mock_files: dict,
    mock_gcs_handler: MagicMock,
    mock_image_analyzer: MagicMock,
    mock_config: MagicMock,
) -> None:
    """Test chat functionality with image files."""
    with patch("rtl_rag_chatbot_api.app.Config", return_value=mock_config):
        # Upload image file
        with open(mock_files["image"], "rb") as f:
            files = {"file": ("mock_file.png", f, "image/png")}
            upload_response = client.post(
                "/file/upload",
                files=files,
                data={"is_image": "true", "username": "test_user"},
            )
            assert upload_response.status_code == 200
            file_id = upload_response.json()["file_id"]

        # Test chat with image
        chat_data = {
            "text": ["What can you see in this image?"],
            "file_id": file_id,
            "model_choice": "gemini-pro-vision",
            "user_id": "test_user",
        }
        response = client.post("/file/chat", json=chat_data)
        assert response.status_code == 200
        assert "response" in response.json()
        assert response.json().get("is_table") is False


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


# @pytest.mark.asyncio
# async def test_create_embeddings(mock_gcs):
#     with patch("rtl_rag_chatbot_api.app.os.path.exists", return_value=True), patch(
#         "rtl_rag_chatbot_api.app.EmbeddingHandler"
#     ) as MockEmbeddingHandler:
#         mock_handler = MockEmbeddingHandler.return_value
#         mock_handler.create_and_upload_embeddings = AsyncMock(
#             return_value={"message": "Embeddings created successfully"}
#         )

#         async with AsyncClient(app=app, base_url="http://test") as ac:
#             response = await ac.post(
#                 "/embeddings/create",
#                 json={"file_id": "test_file_id", "is_image": False},
#             )

#         assert response.status_code == 200
#         assert response.json() == {"message": "Embeddings created successfully"}


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
