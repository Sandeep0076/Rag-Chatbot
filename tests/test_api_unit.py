import logging
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from rtl_rag_chatbot_api.app import app

from .test_resources import TestResourceManager

# Initialize test client
client = TestClient(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Mock files directory
MOCK_FILES_DIR = os.path.join(os.path.dirname(__file__), "mock_files")


@pytest.fixture(autouse=True, scope="session")
def clear_test_resources():
    """Fixture to clear test resources before any tests run."""
    resource_manager = TestResourceManager()
    resource_manager.clear_file_ids()  # Clear any existing file IDs before tests start
    yield
    # Optionally clear again after all tests complete
    resource_manager.clear_file_ids()


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
        # Create a mock get_file_info that returns info for any file_id
        def get_file_info(file_id):
            return {
                "file_id": file_id,  # Use the actual file_id
                "embeddings_status": "completed",
                "is_image": False,
            }

        mock.return_value.get_file_info.side_effect = get_file_info
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


@pytest.fixture
def resource_manager():
    """Fixture providing TestResourceManager instance."""
    return TestResourceManager()


def test_chat_with_pdf(
    mock_files: dict,
    mock_gcs_handler: MagicMock,
    mock_chroma_manager: MagicMock,
    mock_pdf_processor: MagicMock,
    resource_manager: TestResourceManager,
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
        resource_manager.store_file_id(file_id)

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
    mock_files: dict,
    mock_gcs_handler: MagicMock,
    mock_tabular_handler: MagicMock,
    resource_manager: TestResourceManager,
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
        resource_manager.store_file_id(file_id)

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
    mock_files: dict,
    mock_gcs_handler: MagicMock,
    mock_tabular_handler: MagicMock,
    resource_manager: TestResourceManager,
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
        resource_manager.store_file_id(file_id)

    # Test chat with Excel using GPT-4
    chat_data = {
        "text": ["Employee Krista Orcutt is from which location?"],
        "file_id": file_id,
        "model_choice": "gpt_4o_mini",
        "user_id": "test_user",
    }
    response = client.post("/file/chat", json=chat_data)
    assert response.status_code == 200
    assert "response" in response.json()
    response_text = response.json()["response"]
    assert (
        "Pennsylvania" in response_text
    ), f"Expected 'Pennsylvania' in response: {response_text}"
    assert not response.json().get("is_table", False)

    # Test chat with Excel using Gemini Pro
    chat_data_gemini = {
        "text": ["Employee Krista Orcutt is from which location?"],
        "file_id": file_id,
        "model_choice": "gemini-pro",
        "user_id": "test_user",
    }
    response_gemini = client.post("/file/chat", json=chat_data_gemini)
    assert response_gemini.status_code == 200
    assert "response" in response_gemini.json()
    response_text_gemini = response_gemini.json()["response"]
    assert (
        "Pennsylvania" in response_text_gemini
    ), f"Expected 'Pennsylvania' in response: {response_text_gemini}"
    assert not response_gemini.json().get("is_table", False)


def test_chat_with_db(
    mock_files: dict,
    mock_gcs_handler: MagicMock,
    mock_tabular_handler: MagicMock,
    resource_manager: TestResourceManager,
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
        resource_manager.store_file_id(file_id)

    # Test chat with DB using GPT-4
    chat_data = {
        "text": ["What is the address of customer Maria Anders?"],
        "file_id": file_id,
        "model_choice": "gpt_4o_mini",
        "user_id": "test_user",
    }
    response = client.post("/file/chat", json=chat_data)
    assert response.status_code == 200
    assert "response" in response.json()
    response_text = response.json()["response"]
    assert (
        "Obere Str. 57" in response_text
    ), f"Expected 'Obere Str. 57' in response: {response_text}"
    assert not response.json().get("is_table", False)

    # Test chat with DB using Gemini Pro
    chat_data_gemini = {
        "text": ["What is the address of customer Maria Anders?"],
        "file_id": file_id,
        "model_choice": "gemini-pro",
        "user_id": "test_user",
    }
    response_gemini = client.post("/file/chat", json=chat_data_gemini)
    assert response_gemini.status_code == 200
    assert "response" in response_gemini.json()
    response_text_gemini = response_gemini.json()["response"]
    assert (
        "Obere Str. 57" in response_text_gemini
    ), f"Expected 'Obere Str. 57' in response: {response_text_gemini}"
    assert not response_gemini.json().get("is_table", False)


def test_chat_with_image(
    mock_files: dict,
    mock_gcs_handler: MagicMock,
    mock_image_analyzer: MagicMock,
    mock_config: MagicMock,
    resource_manager: TestResourceManager,
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
            resource_manager.store_file_id(file_id)

        # Test chat with image using GPT-
        chat_data = {
            "text": ["Who has highest gdp per capita?"],
            "file_id": file_id,
            "model_choice": "gpt_4o_mini",
            "user_id": "test_user",
        }
        response = client.post("/file/chat", json=chat_data)
        assert response.status_code == 200
        assert "response" in response.json()
        response_text = response.json()["response"]
        assert any(
            country in response_text for country in ["United States", "US", "USA"]
        ), f"Expected US reference in response: {response_text}"
        assert not response.json().get("is_table", False)

        # Test chat with image using Gemini Pro Vision
        chat_data_gemini = {
            "text": ["Who has highest gdp per capita?"],
            "file_id": file_id,
            "model_choice": "gemini-pro",
            "user_id": "test_user",
        }
        response_gemini = client.post("/file/chat", json=chat_data_gemini)
        assert response_gemini.status_code == 200
        assert "response" in response_gemini.json()
        response_text_gemini = response_gemini.json()["response"]
        assert any(
            country in response_text_gemini
            for country in ["United States", "US", "USA"]
        ), f"Expected US reference in response: {response_text_gemini}"
        assert not response_gemini.json().get("is_table", False)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "up"}


def test_info():
    response = client.get("/info")
    assert response.status_code == 200
    assert "title" in response.json()
    assert "description" in response.json()


def test_file_upload(mock_files: dict, resource_manager: TestResourceManager) -> None:
    """Test file upload functionality with a real CSV file."""
    # Upload CSV file
    with open(mock_files["csv"], "rb") as f:
        files = {"file": ("mock_file.csv", f, "text/csv")}
        response = client.post(
            "/file/upload",
            files=files,
            data={"is_image": "false", "username": "test_user"},
        )

        # Check basic response structure
        assert response.status_code == 200
        assert "file_id" in response.json()
        assert "message" in response.json()
        assert "original_filename" in response.json()
        assert "is_image" in response.json()
        assert response.json()["is_image"] is False

        # Store file ID for cleanup
        resource_manager.store_file_id(response.json()["file_id"])


def test_cleanup_test_resources(resource_manager: TestResourceManager):
    """Cleanup test resources after all tests have completed."""
    file_ids = resource_manager.get_all_file_ids()
    if not file_ids:
        return

    # Delete both ChromaDB embeddings and GCS resources
    response = client.request(
        "DELETE",
        "/delete",
        json={
            "file_ids": file_ids,
            "include_gcs": True,  # Explicitly set to True to clean up GCS resources
        },
    )
    assert response.status_code == 200
    resource_manager.clear_file_ids()


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


# @pytest.mark.asyncio
# async def test_gemini_chat():
#     async def mock_stream():
#         yield "Test response"

#     with patch("rtl_rag_chatbot_api.app.ModelHandler") as MockModelHandler:
#         mock_model = MagicMock()
#         mock_model.get_gemini_response_stream.return_value = mock_stream()
#         MockModelHandler.return_value.initialize_model.return_value = mock_model

#         async with AsyncClient(app=app, base_url="http://test") as ac:
#             response = await ac.post(
#                 "/chat/gemini", json={"model": "gemini-pro", "message": "Test message"}
#             )

#         assert response.status_code == 200
#         assert response.headers["content-type"].startswith("text/plain")
