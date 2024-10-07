from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from configs.app_config import Config
from rtl_rag_chatbot_api.app import app
from rtl_rag_chatbot_api.chatbot.embedding_handler import EmbeddingHandler
from rtl_rag_chatbot_api.chatbot.gemini_handler import GeminiHandler

client = TestClient(app)


# Mock the Config class
@pytest.fixture
def mock_config():
    mock_config = MagicMock(spec=Config)
    mock_config.gemini = MagicMock()
    mock_config.gemini.model_pro = "gemini-pro"
    mock_config.gemini.model_flash = "gemini-flash"
    return mock_config


class MockNode:
    def __init__(self, text):
        self.node = MagicMock()
        self.node.text = text
        self.text = text


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
def test_file_upload(mock_process_file, mock_uuid):
    mock_uuid.return_value = "test_file_id"
    mock_process_file.return_value = {
        "file_id": "test_file_id",
        "status": "new",
        "message": "File uploaded successfully",
        "is_image": False,
    }

    response = client.post(
        "/file/upload",
        files={"file": ("test.pdf", b"test content", "application/pdf")},
        data={"is_image": "false", "username": "testuser"},
    )

    assert response.status_code == 200
    assert response.json()["file_id"] == "test_file_id"
    assert (
        response.json()["message"]
        == "File uploaded, encrypted, and processed successfully"
    )
    assert response.json()["original_filename"] == "test.pdf"
    assert response.json()["is_image"] is False

    mock_process_file.assert_called_once()
    mock_uuid.assert_called_once()


@pytest.mark.asyncio
@patch("rtl_rag_chatbot_api.app.embedding_handler.get_embeddings_info")
@patch("rtl_rag_chatbot_api.app.os.path.exists")
@patch("rtl_rag_chatbot_api.app.TabularDataHandler")
@patch("rtl_rag_chatbot_api.app.model_handler.initialize_model")
async def test_initialize_model(
    mock_initialize_model,
    mock_tabular_handler,
    mock_path_exists,
    mock_get_embeddings_info,
):
    mock_get_embeddings_info.return_value = {"embeddings": {"azure": "completed"}}
    mock_initialize_model.return_value = MagicMock()
    mock_tabular_handler.return_value = MagicMock()

    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Test for regular file (PDF/Image)
        mock_path_exists.return_value = False
        response = await ac.post(
            "/model/initialize",
            json={"model_choice": "gpt-3.5-turbo", "file_id": "test_file_id"},
        )
        assert response.status_code == 200
        assert "initialized successfully" in response.json()["message"]
        mock_initialize_model.assert_called_once()

        # Reset mocks
        mock_initialize_model.reset_mock()
        mock_path_exists.reset_mock()

        # Test for CSV/Excel file
        mock_path_exists.return_value = True
        response = await ac.post(
            "/model/initialize",
            json={"model_choice": "gpt-3.5-turbo", "file_id": "csv_file_id"},
        )
        assert response.status_code == 200
        assert "initialized successfully" in response.json()["message"]
        mock_tabular_handler.assert_called_once()

        # Test for file not found
        mock_get_embeddings_info.return_value = None
        mock_path_exists.return_value = False
        response = await ac.post(
            "/model/initialize",
            json={"model_choice": "gpt-3.5-turbo", "file_id": "nonexistent_file_id"},
        )
        assert response.status_code == 404
        assert "Embeddings not found for this file" in response.json()["detail"]


def test_available_models():
    response = client.get("/available-models")
    assert response.status_code == 200
    assert "models" in response.json()
    assert isinstance(response.json()["models"], list)


@patch("rtl_rag_chatbot_api.app.GCSHandler")
def test_cleanup_files(mock_gcs_handler):
    mock_gcs_handler.return_value.cleanup_local_files.return_value = None
    response = client.post("/file/cleanup")
    assert response.status_code == 200
    assert response.json() == {"status": "Cleanup completed successfully"}


@patch("rtl_rag_chatbot_api.app.analyze_images")
def test_analyze_image(mock_analyze_images):
    """
    Test the functionality of analyzing an image by mocking the 'analyze_images' function.
    Ensure that the API endpoint '/image/analyze' returns a status code of 200 and
    includes the keys 'message' and 'analysis' in the JSON response.
    """
    mock_analyze_images.return_value = [{"analysis": "Test analysis"}]

    response = client.post(
        "/image/analyze",
        files={"file": ("test.jpg", b"test image content", "image/jpeg")},
    )
    assert response.status_code == 200
    assert "message" in response.json()
    assert "analysis" in response.json()


@patch("rtl_rag_chatbot_api.app.initialized_models")
def test_chat(mock_initialized_models):
    """
    Test the chat functionality by mocking initialized models and checking
    responses for different scenarios.
    """
    mock_model = MagicMock()
    mock_model.get_answer.return_value = "Test response"
    mock_initialized_models.__getitem__.return_value = mock_model
    mock_initialized_models.__contains__.return_value = True

    response = client.post(
        "/file/chat",
        json={
            "text": "Test query",
            "file_id": "test_file_id",
            "model_choice": "gpt-3.5-turbo",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"response": "Test response"}

    # Test for TabularDataHandler (CSV) case
    mock_tabular_handler = MagicMock()
    mock_tabular_handler.get_answer.return_value = "CSV Test response"
    mock_initialized_models.__getitem__.return_value = mock_tabular_handler

    response = client.post(
        "/file/chat",
        json={
            "text": "CSV Test query",
            "file_id": "csv_file_id",
            "model_choice": "gpt-3.5-turbo",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"response": "CSV Test response"}


@pytest.mark.asyncio
async def test_create_embeddings():
    """
    Asynchronous test function to verify the creation of embeddings.
    Mocks the EmbeddingHandler methods to simulate successful creation and upload of embeddings.
    Sends a POST request to test the creation of embeddings with specified file_id and image status.
    Checks the response status code and content for successful creation.
    """
    async with AsyncClient(app=app, base_url="http://test") as ac:
        with patch.object(
            EmbeddingHandler, "embeddings_exist", return_value=False
        ), patch.object(
            EmbeddingHandler,
            "create_and_upload_embeddings",
            new_callable=AsyncMock,
            return_value={"message": "Embeddings created successfully"},
        ):
            response = await ac.post(
                "/embeddings/create",
                json={"file_id": "test_file_id", "is_image": False},
            )

    print(f"Response status: {response.status_code}")
    print(f"Response content: {response.content}")
    assert response.status_code == 200
    assert response.json() == {"message": "Embeddings created successfully"}


@pytest.mark.asyncio
async def test_get_neighbors_model_not_initialized():
    with patch("rtl_rag_chatbot_api.app.initialized_models", {}):
        response = client.post(
            "/file/neighbors",
            json={"text": "Test query", "file_id": "test_file_id", "n_neighbors": 3},
        )

    print(f"Response status code: {response.status_code}")
    print(f"Response content: {response.content}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Model not initialized for this file"}


@pytest.mark.asyncio
async def test_get_neighbors():
    """
    Asynchronous unit test for the 'test_get_neighbors' function.
    Mocks a model to return nearest neighbors and tests the API endpoint '/file/neighbors'.
    Asserts the response status code and content against expected values.
    """
    mock_model = MagicMock()
    mock_model.get_n_nearest_neighbours.return_value = [
        MagicMock(node=MagicMock(text="Neighbor 1")),
        MagicMock(node=MagicMock(text="Neighbor 2")),
        MagicMock(node=MagicMock(text="Neighbor 3")),
    ]

    with patch(
        "rtl_rag_chatbot_api.app.initialized_models", {"test_file_id": mock_model}
    ):
        response = client.post(
            "/file/neighbors",
            json={"text": "Test query", "file_id": "test_file_id", "n_neighbors": 3},
        )

    print(f"Response status code: {response.status_code}")
    print(f"Response content: {response.content}")

    if response.status_code != 200:
        print(f"Error response: {response.json()}")

    assert response.status_code == 200
    assert response.json() == {"neighbors": ["Neighbor 1", "Neighbor 2", "Neighbor 3"]}


@pytest.mark.asyncio
async def test_get_neighbors_gemini():
    """
    Asynchronous unit test for the function that retrieves Gemini neighbors.
    Mocks the GeminiHandler to return a list of neighbors.
    Sends a POST request to test the endpoint for retrieving neighbors.
    Checks the response status code and content for correctness.
    """
    mock_gemini_handler = MagicMock(spec=GeminiHandler)
    mock_gemini_handler.get_n_nearest_neighbours.return_value = [
        "Gemini Neighbor 1",
        "Gemini Neighbor 2",
        "Gemini Neighbor 3",
    ]

    with patch(
        "rtl_rag_chatbot_api.app.initialized_models",
        {"test_file_id": mock_gemini_handler},
    ):
        response = client.post(
            "/file/neighbors",
            json={"text": "Test query", "file_id": "test_file_id", "n_neighbors": 3},
        )

    print(f"Response status code: {response.status_code}")
    print(f"Response content: {response.content}")

    if response.status_code != 200:
        print(f"Error response: {response.json()}")

    assert response.status_code == 200
    assert response.json() == {
        "neighbors": ["Gemini Neighbor 1", "Gemini Neighbor 2", "Gemini Neighbor 3"]
    }


@pytest.mark.asyncio
async def test_delete_files():
    """
    Asynchronous unit test for deleting files from the server.
    Mocks GCSHandler methods to simulate file deletion.
    Verifies the response status code, content, and method calls.
    Asserts the expected response data and method calls.
    """
    file_ids = ["file1", "file2", "file3"]

    # Mock GCSHandler and its methods
    with patch("rtl_rag_chatbot_api.app.GCSHandler") as MockGCSHandler, patch(
        "rtl_rag_chatbot_api.app.os.path.exists", return_value=True
    ) as mock_exists, patch("rtl_rag_chatbot_api.app.shutil.rmtree") as mock_rmtree:
        mock_gcs_handler = MockGCSHandler.return_value

        # Set up spies on all potentially relevant methods
        mock_gcs_handler.delete_file_and_embeddings = MagicMock()
        mock_gcs_handler.delete_folder = MagicMock()

        response = client.request("DELETE", "/files", json={"file_ids": file_ids})

    print(f"Response status code: {response.status_code}")
    print(f"Response content: {response.json()}")

    print(f"GCSHandler initialization: {MockGCSHandler.call_args_list}")
    print(
        f"delete_file_and_embeddings call count: {mock_gcs_handler.delete_file_and_embeddings.call_count}"
    )
    print(
        f"delete_file_and_embeddings call args: {mock_gcs_handler.delete_file_and_embeddings.call_args_list}"
    )
    print(f"delete_folder call count: {mock_gcs_handler.delete_folder.call_count}")
    print(f"delete_folder call args: {mock_gcs_handler.delete_folder.call_args_list}")
    print(f"os.path.exists call count: {mock_exists.call_count}")
    print(f"os.path.exists call args: {mock_exists.call_args_list}")
    print(f"shutil.rmtree call count: {mock_rmtree.call_count}")
    print(f"shutil.rmtree call args: {mock_rmtree.call_args_list}")

    assert response.status_code == 200
    response_data = response.json()

    assert "message" in response_data
    assert "deleted_files" in response_data

    # Check if any delete method was called
    delete_method_called = (
        mock_gcs_handler.delete_file_and_embeddings.call_count > 0
        or mock_gcs_handler.delete_folder.call_count > 0
    )

    assert (
        delete_method_called
    ), "Either delete_file_and_embeddings or delete_folder should be called"

    # Verify that all file_ids are in the deleted_files list
    assert set(response_data["deleted_files"]) == set(file_ids)


# Create a new fixture for Gemini chat tests
@pytest.fixture
def gemini_chat_client(mock_config):
    """
    Fixture for creating a TestClient instance for Gemini chat tests.
    Mocks the Config and GCSHandler classes for testing purposes.
    """
    with patch("rtl_rag_chatbot_api.app.Config", return_value=mock_config):
        with patch("rtl_rag_chatbot_api.app.GCSHandler"):
            with TestClient(app) as test_client:
                yield test_client


@pytest.mark.asyncio
async def test_get_gemini_response_stream(gemini_chat_client):
    """
    Test the endpoint for getting a streaming response from the Gemini chatbot.
    Mocks the ModelHandler class and asserts the expected behavior of the endpoint.
    """
    test_response = "Test response"

    async def mock_stream():
        for word in test_response.split():
            yield word + " "

    mock_model = MagicMock()
    mock_model.get_gemini_response_stream.return_value = mock_stream()

    with patch("rtl_rag_chatbot_api.app.ModelHandler") as MockModelHandler:
        MockModelHandler.return_value.initialize_model.return_value = mock_model

        response = gemini_chat_client.post(
            "/chat/gemini", json={"model": "gemini-pro", "message": "Test message"}
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")

        # Read the streaming response
        content = response.content.decode("utf-8")

        assert content.strip() == test_response

        MockModelHandler.assert_called_once()
        MockModelHandler.return_value.initialize_model.assert_called_once_with(
            "gemini-pro", file_id=None, embedding_type="gemini"
        )
        mock_model.get_gemini_response_stream.assert_called_once_with("Test message")
