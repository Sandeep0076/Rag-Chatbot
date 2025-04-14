import logging
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rtl_rag_chatbot_api.app import app

from . import delete_test_embeddings
from .test_utils import ResourceManager


@pytest.fixture(scope="session", autouse=True)
def run_delete_embeddings_first():
    """Run delete_test_embeddings.py before any tests to clean up test embeddings."""
    logging.info("Running delete_test_embeddings.py before tests")
    delete_test_embeddings.main()
    yield
    # Optionally run again after all tests
    logging.info("Running delete_test_embeddings.py after tests")
    delete_test_embeddings.main()


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
    resource_manager = ResourceManager()
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
        "text": os.path.join(MOCK_FILES_DIR, "mock_txt.txt"),
    }


@pytest.fixture
def mock_config():
    mock_config = MagicMock()
    mock_config.gemini = MagicMock()
    mock_config.gemini.model_pro = "gemini-pro"
    mock_config.gemini.model_pro_vision = "gemini-pro-vision"
    mock_config.gemini.api_key = "test-api-key"

    # Add chatbot configuration
    mock_config.chatbot = MagicMock()
    mock_config.chatbot.system_prompt_plain_llm = (
        "You are a helpful assistant that provides accurate and relevant "
        "information based on the given data."
    )
    mock_config.chatbot.system_prompt_rag_llm = (
        "You are a helpful assistant that provides accurate and relevant "
        "information based on the retrieved context."
    )
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


@pytest.fixture(scope="session")
def resource_manager():
    """Fixture providing ResourceManager instance."""
    return ResourceManager()


@pytest.fixture
def mock_chart_pdf():
    """Fixture providing mock chart PDF file path."""
    return os.path.join(MOCK_FILES_DIR, "mock_chart.pdf")


def test_chat_with_pdf(
    mock_files: dict,
    mock_gcs_handler: MagicMock,
    mock_chroma_manager: MagicMock,
    mock_pdf_processor: MagicMock,
    resource_manager: ResourceManager,
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
        resource_manager.add_file_id(file_id)

    # Test chat with PDF using GPT-4
    chat_data = {
        "text": ["From which country this paper is from ?"],
        "file_id": file_id,
        "model_choice": "gpt_4o_mini",
        "user_id": "test_user",
    }
    response = client.post("/file/chat", json=chat_data)
    assert response.status_code == 200
    assert "response" in response.json()
    assert "Uzbekistan" in response.json()["response"]
    assert not response.json().get("is_table", False)

    # Test chat with PDF using Gemini Pro
    chat_data_gemini = {
        "text": ["From which country this paper is from ?"],
        "file_id": file_id,
        "model_choice": "gemini-pro",
        "user_id": "test_user",
    }
    response_gemini = client.post("/file/chat", json=chat_data_gemini)
    assert response_gemini.status_code == 200
    assert "response" in response_gemini.json()
    assert "Uzbekistan" in response_gemini.json()["response"]
    assert not response_gemini.json().get("is_table", False)


def test_chat_with_csv(
    mock_files: dict,
    mock_gcs_handler: MagicMock,
    mock_tabular_handler: MagicMock,
    resource_manager: ResourceManager,
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
        resource_manager.add_file_id(file_id)

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
    resource_manager: ResourceManager,
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
        resource_manager.add_file_id(file_id)

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
    resource_manager: ResourceManager,
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
        resource_manager.add_file_id(file_id)

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
    resource_manager: ResourceManager,
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
            resource_manager.add_file_id(file_id)

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


@pytest.mark.asyncio
async def test_chat_with_image_visualization(
    mock_files: dict,
    mock_gcs_handler: MagicMock,
    mock_image_analyzer: MagicMock,
    mock_config: MagicMock,
    resource_manager: ResourceManager,
):
    """Test chat with image functionality when visualization is auto-detected from query text.
    Verifies that image analysis queries containing visualization keywords trigger chart generation.
    """
    # Mock authentication and config
    with patch(
        "rtl_rag_chatbot_api.app.get_current_user",
        return_value={"username": "test_user"},
    ), patch("rtl_rag_chatbot_api.app.Config", return_value=mock_config):
        # Upload the test image file
        with open(mock_files["image"], "rb") as f:
            response = client.post(
                "/file/upload",
                files={"file": ("test.png", f, "image/png")},
                data={"is_image": "true", "username": "test_user"},
            )
        assert response.status_code == 200
        file_id = response.json()["file_id"]
        resource_manager.add_file_id(file_id)

        # Test chat with visualization request (visualization need is auto-detected from query text)
        query = {
            "text": ["Generate a chart comparing gdp of china and india"],
            "file_id": file_id,
            "model_choice": "gemini-pro",
            "user_id": "test_user",
        }

        chat_response = client.post("/file/chat", json=query)
        assert chat_response.status_code == 200
        response_data = chat_response.json()

        # Verify chart configuration
        assert "chart_config" in response_data
        chart_config = response_data["chart_config"]

        # Verify basic chart structure without checking specific values
        assert isinstance(chart_config, dict)

        # Convert the entire response to string to search for countries
        response_str = str(response_data)
        assert any(
            country in response_str for country in ["China", "India"]
        ), "Response should contain either China or India"


@pytest.mark.asyncio
async def test_chat_with_pdf_visualization(
    mock_chart_pdf: str,
    mock_gcs_handler: MagicMock,
    mock_chroma_manager: MagicMock,
    mock_pdf_processor: MagicMock,
    resource_manager: ResourceManager,
):
    """Test chat with PDF functionality when visualization is auto-detected from query text.
    Verifies that queries containing visualization keywords trigger chart generation.
    """
    # Mock authentication
    with patch(
        "rtl_rag_chatbot_api.app.get_current_user",
        return_value={"username": "test_user"},
    ):
        # Upload the test PDF file
        with open(mock_chart_pdf, "rb") as f:
            response = client.post(
                "/file/upload",
                files={"file": ("test.pdf", f, "application/pdf")},
                data={"is_image": "false", "username": "test_user"},
            )
        assert response.status_code == 200
        file_id = response.json()["file_id"]
        resource_manager.add_file_id(file_id)

        # No need to mock ChromaDB response - we'll use the actual PDF content

        # Test chat with visualization request (visualization need is auto-detected from query text)
        query = {
            "text": ["Create a pie chart for distribution of operating systems"],
            "file_id": file_id,
            "model_choice": "gpt_4o_mini",
            "user_id": "test_user",
        }

        chat_response = client.post("/file/chat", json=query)
        assert chat_response.status_code == 200
        response_data = chat_response.json()

        # Verify chart configuration
        assert "chart_config" in response_data
        chart_config = response_data["chart_config"]

        # Verify basic chart structure without checking specific values
        assert isinstance(chart_config, dict)

        # Verify data exists and has a valid structure
        assert "data" in chart_config
        data = chart_config["data"]

        # Check that data contains some form of dataset structure
        # but don't validate specific keys or values
        assert isinstance(data, dict)
        assert any(
            key in ["datasets", "categories", "series", "data"] for key in data.keys()
        )

        # If datasets exist, verify they have the minimum required structure
        if "datasets" in data:
            assert isinstance(data["datasets"], list)
            if data["datasets"]:  # if not empty
                dataset = data["datasets"][0]
                assert all(key in dataset for key in ["x", "y"])


@pytest.mark.asyncio
async def test_chat_with_csv_visualization(
    mock_files: dict,
    mock_gcs_handler: MagicMock,
    mock_tabular_handler: MagicMock,
    resource_manager: ResourceManager,
):
    """Test chat with CSV functionality when visualization is auto-detected from query text.
    Verifies that queries containing visualization keywords work correctly with tabular data.
    """
    # Mock authentication
    with patch(
        "rtl_rag_chatbot_api.app.get_current_user",
        return_value={"username": "test_user"},
    ):
        # Upload the test CSV file
        with open(mock_files["csv"], "rb") as f:
            response = client.post(
                "/file/upload",
                files={"file": ("mock_file.csv", f, "text/csv")},
                data={"is_image": "false", "username": "test_user"},
            )
        assert response.status_code == 200
        file_id = response.json()["file_id"]
        resource_manager.add_file_id(file_id)

        # Test chat with visualization request (visualization need is auto-detected from query text)
        query = {
            "text": [
                "Generate a chart for Relationship between pregnancies and age for first 10 entries"
            ],
            "file_id": file_id,
            "model_choice": "gemini-pro",
            "user_id": "test_user",
        }

        chat_response = client.post("/file/chat", json=query)
        assert chat_response.status_code == 200
        response_data = chat_response.json()

        # Verify chart configuration
        assert "chart_config" in response_data
        chart_config = response_data["chart_config"]

        # Verify basic chart structure without checking specific values
        assert isinstance(chart_config, dict)

        # Verify data exists and has a valid structure
        assert "data" in chart_config
        data = chart_config["data"]

        # Check that data contains some form of dataset structure
        # but don't validate specific keys or values
        assert isinstance(data, dict)
        assert any(
            key in ["datasets", "categories", "series", "data"] for key in data.keys()
        )

        # If datasets exist, verify they have the minimum required structure
        if "datasets" in data:
            assert isinstance(data["datasets"], list)
            if data["datasets"]:  # if not empty
                dataset = data["datasets"][0]
                assert all(key in dataset for key in ["x", "y"])


def test_chat_with_doc(
    mock_files: dict,
    mock_gcs_handler: MagicMock,
    mock_chroma_manager: MagicMock,
    mock_pdf_processor: MagicMock,
    resource_manager: ResourceManager,
) -> None:
    """Test chat functionality with text document files."""
    # Upload text file
    with open(mock_files["text"], "rb") as f:
        files = {"file": ("mock_txt.txt", f, "text/plain")}
        upload_response = client.post(
            "/file/upload",
            files=files,
            data={"is_image": "false", "username": "test_user"},
        )
        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]
        resource_manager.add_file_id(file_id)

    # Test chat with text document using GPT-4
    chat_data = {
        "text": ["How many mangoes are there in Garden"],
        "file_id": file_id,
        "model_choice": "gpt_4o_mini",
        "user_id": "test_user",
    }
    response = client.post("/file/chat", json=chat_data)
    assert response.status_code == 200
    assert "response" in response.json()
    assert "50" in response.json()["response"]
    assert not response.json().get("is_table", False)

    # Test chat with text document using Gemini Pro
    chat_data_gemini = {
        "text": ["How many mangoes are there in Garden"],
        "file_id": file_id,
        "model_choice": "gemini-pro",
        "user_id": "test_user",
    }
    response_gemini = client.post("/file/chat", json=chat_data_gemini)
    assert response_gemini.status_code == 200
    assert "response" in response_gemini.json()
    assert "50" in response_gemini.json()["response"]
    assert not response_gemini.json().get("is_table", False)


def test_chat_with_url(
    mock_files: dict,
    resource_manager: ResourceManager,
) -> None:
    """Test chat functionality with URL content."""
    # Upload URL content
    upload_response = client.post(
        "/file/upload",
        data={
            "urls": "https://en.wikipedia.org/wiki/Elon_Musk",
            "username": "test_user",
        },
    )

    # Verify upload response
    assert upload_response.status_code == 200
    file_id = upload_response.json()["file_id"]
    resource_manager.add_file_id(file_id)

    # Test chat with URL content
    chat_data = {
        "text": ["In which month Elon musk was born"],
        "file_id": file_id,
        "model_choice": "gpt_4o_mini",
        "user_id": "test_user",
    }

    # Send chat request
    response = client.post("/file/chat", json=chat_data)
    assert response.status_code == 200
    assert "response" in response.json()

    # Check if response contains June or 6
    response_text = response.json()["response"]
    assert any(
        month in response_text for month in ["June", "6"]
    ), f"Response '{response_text}' does not contain 'June' or '6'"
    assert not response.json().get("is_table", False)


def test_health():
    response = client.get("/internal/healthy")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_info():
    response = client.get("/info")
    assert response.status_code == 200
    assert "title" in response.json()
    assert "description" in response.json()


def test_file_upload(mock_files: dict, resource_manager: ResourceManager) -> None:
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
        file_id = response.json()["file_id"]
        resource_manager.add_file_id(file_id)


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
