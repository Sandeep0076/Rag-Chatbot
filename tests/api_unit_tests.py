from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rtl_rag_chatbot_api.app import app, initialized_chatbots

client = TestClient(app)


@pytest.fixture
def mock_gcs_handler():
    """
    Fixture to mock the GCSHandler class.
    """
    with patch("rtl_rag_chatbot_api.app.GCSHandler") as mock:
        yield mock.return_value


@pytest.fixture
def mock_chatbot():
    """
    Fixture to mock the Chatbot class.
    """
    with patch("rtl_rag_chatbot_api.app.Chatbot") as mock:
        yield mock.return_value


def test_chat_endpoint(mock_chatbot):
    """
    Test the /file/chat endpoint for a successful chat interaction.

    This test verifies that:
    1. The endpoint returns a 200 status code for a valid request.
    2. The response contains the expected chat response.
    """
    file_id = "test_file_id"
    initialized_chatbots[file_id] = mock_chatbot
    mock_chatbot.get_answer.return_value = "Test response"

    response = client.post(
        "/file/chat",
        json={
            "text": "Test query",
            "file_id": file_id,
            "model_choice": "gpt_3_5_turbo",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"response": "Test response"}


def test_chat_endpoint_not_initialized():
    """
    Test the /file/chat endpoint when the chatbot is not initialized.

    This test verifies that:
    1. The endpoint returns a 404 status code when the chatbot is not initialized.
    2. The response contains an appropriate error message.
    """
    response = client.post(
        "/file/chat",
        json={
            "text": "Test query",
            "file_id": "non_existent_file_id",
            "model_choice": "gpt_3_5_turbo",
        },
    )

    assert response.status_code == 404
    assert "Chatbot not initialized" in response.json()["detail"]


def test_available_models():
    """
    Test the /available-models endpoint.

    This test verifies that:
    1. The endpoint returns a 200 status code.
    2. The response contains a list of available models.
    """
    response = client.get("/available-models")
    assert response.status_code == 200
    assert "models" in response.json()
    assert isinstance(response.json()["models"], list)


def test_cleanup_files(mock_gcs_handler):
    """
    Test the /file/cleanup endpoint for successful cleanup.

    This test verifies that:
    1. The endpoint returns a 200 status code when cleanup is successful.
    2. The cleanup_local_files method of GCSHandler is called.
    """
    mock_gcs_handler.cleanup_local_files.return_value = None

    response = client.post("/file/cleanup")

    assert response.status_code == 200
    assert response.json() == {"status": "Cleanup completed successfully"}
    mock_gcs_handler.cleanup_local_files.assert_called_once()


def test_cleanup_files_error(mock_gcs_handler):
    """
    Test the /file/cleanup endpoint when an error occurs during cleanup.

    This test verifies that:
    1. The endpoint returns a 500 status code when an error occurs.
    2. The response contains an appropriate error message.
    """
    mock_gcs_handler.cleanup_local_files.side_effect = Exception("Test error")

    response = client.post("/file/cleanup")

    assert response.status_code == 500
    assert "An error occurred during cleanup" in response.json()["detail"]


@pytest.fixture
def mock_file():
    """
    Fixture to create a mock file object.
    """
    return MagicMock(filename="test.pdf")


def test_file_upload(mock_gcs_handler, mock_file, mock_chatbot):
    mock_gcs_handler.find_existing_file.return_value = None
    mock_gcs_handler.upload_to_gcs.return_value = None
    mock_gcs_handler.download_and_decrypt_file.return_value = "decrypted_file_path"

    with patch(
        "rtl_rag_chatbot_api.app.run_preprocessor"
    ) as mock_run_preprocessor, patch(
        "rtl_rag_chatbot_api.app.initialize_chatbot"
    ) as mock_initialize_chatbot:
        mock_run_preprocessor.return_value = None
        mock_initialize_chatbot.return_value = None

        response = client.post(
            "/file/upload",
            files={"file": ("test.pdf", b"test content", "application/pdf")},
            data={"is_image": "false"},
        )

    assert response.status_code == 200
    assert "file_id" in response.json()
    assert response.json()["original_filename"] == "test.pdf"
    assert response.json()["is_image"] is False


def test_file_upload_existing(mock_gcs_handler, mock_file):
    mock_gcs_handler.find_existing_file.return_value = "existing_file_id"
    mock_gcs_handler.download_files_from_folder_by_id.return_value = None

    with patch(
        "rtl_rag_chatbot_api.app.initialize_chatbot"
    ) as mock_initialize_chatbot, patch(
        "rtl_rag_chatbot_api.app.gcs_handler", mock_gcs_handler
    ):
        mock_initialize_chatbot.return_value = None

        response = client.post(
            "/file/upload",
            files={"file": ("test.pdf", b"test content", "application/pdf")},
            data={"is_image": "false"},
        )

    assert response.status_code == 200
    assert response.json()["file_id"] == "existing_file_id"
    assert "File already exists" in response.json()["message"]
    assert response.json()["original_filename"] == "test.pdf"
    assert not response.json()["is_image"]

    # Verify that the mocked methods were called
    mock_gcs_handler.find_existing_file.assert_called_once_with("test.pdf")
    mock_gcs_handler.download_files_from_folder_by_id.assert_called_once_with(
        "existing_file_id"
    )


def test_neighbors(mock_chatbot):
    """
    Test the /file/neighbors endpoint for successful retrieval of neighbors.

    This test verifies that:
    1. The endpoint returns a 200 status code for a valid request.
    2. The response contains the expected list of neighbors.
    """
    file_id = "test_file_id"
    initialized_chatbots[file_id] = mock_chatbot
    mock_chatbot.get_n_nearest_neighbours.return_value = [
        MagicMock(node=MagicMock(text="Neighbor 1")),
        MagicMock(node=MagicMock(text="Neighbor 2")),
    ]

    response = client.post(
        "/file/neighbors",
        json={"text": "Test query", "file_id": file_id, "n_neighbors": 2},
    )

    assert response.status_code == 200
    assert response.json() == {"neighbors": ["Neighbor 1", "Neighbor 2"]}


def test_neighbors_not_initialized():
    """
    Test the /file/neighbors endpoint when the chatbot is not initialized.

    This test verifies that:
    1. The endpoint returns a 404 status code when the chatbot is not initialized.
    2. The response contains an appropriate error message.
    """
    response = client.post(
        "/file/neighbors",
        json={
            "text": "Test query",
            "file_id": "non_existent_file_id",
            "n_neighbors": 2,
        },
    )

    assert response.status_code == 404
    assert "Chatbot not initialized" in response.json()["detail"]


@patch("rtl_rag_chatbot_api.app.analyze_images")
def test_image_analyze(mock_analyze_images, mock_file):
    """
    Test the /image/analyze endpoint for successful image analysis.

    This test verifies that:
    1. The endpoint returns a 200 status code for a successful analysis.
    2. The response contains the expected analysis results.
    """
    mock_analyze_images.return_value = {"analysis": "Test analysis"}

    response = client.post(
        "/image/analyze",
        files={"file": ("test.jpg", b"test image content", "image/jpeg")},
    )

    assert response.status_code == 200
    assert "message" in response.json()
    assert "result_file" in response.json()
    assert response.json()["analysis"] == {"analysis": "Test analysis"}


def test_image_analyze_error(mock_file):
    """
    Test the /image/analyze endpoint when an error occurs during analysis.

    This test verifies that:
    1. The endpoint returns a 500 status code when an error occurs.
    2. The response contains an appropriate error message.
    """
    with patch(
        "rtl_rag_chatbot_api.app.analyze_images", side_effect=Exception("Test error")
    ):
        response = client.post(
            "/image/analyze",
            files={"file": ("test.jpg", b"test image content", "image/jpeg")},
        )

    assert response.status_code == 500
    assert "An error occurred during image analysis" in response.json()["detail"]
