from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rtl_rag_chatbot_api.app import app, initialized_models

client = TestClient(app)


@pytest.fixture
def mock_configs():
    with patch("rtl_rag_chatbot_api.app.configs") as mock:
        mock.azure_llm.models = {"gpt-3.5-turbo": {}, "gpt-4": {}}
        yield mock


@pytest.fixture
def mock_gemini_handler():
    with patch("rtl_rag_chatbot_api.app.GeminiHandler") as mock:
        instance = mock.return_value
        instance.get_answer.return_value = "Gemini response"
        instance.get_n_nearest_neighbours.return_value = ["Neighbor 1", "Neighbor 2"]
        yield instance


@pytest.fixture
def mock_gcs_handler():
    with patch("rtl_rag_chatbot_api.app.GCSHandler") as mock:
        yield mock.return_value


@pytest.fixture
def mock_chatbot():
    with patch("rtl_rag_chatbot_api.app.Chatbot") as mock:
        yield mock.return_value


@pytest.fixture
def mock_file():
    return MagicMock(filename="test.pdf")


def test_available_models(mock_configs):
    response = client.get("/available-models")
    assert response.status_code == 200
    assert "models" in response.json()
    assert isinstance(response.json()["models"], list)
    assert set(response.json()["models"]) == {
        "gpt-3.5-turbo",
        "gpt-4",
        "gemini-flash",
        "gemini-pro",
    }


def test_chat_endpoint_gemini(mock_gemini_handler, mock_configs, mock_gcs_handler):
    file_id = "test_file_id"
    initialized_models[file_id] = {"type": "gemini", "model": mock_gemini_handler}

    # Patch the global gemini_handler
    with patch("rtl_rag_chatbot_api.app.gemini_handler", mock_gemini_handler):
        response = client.post(
            "/file/chat",
            json={
                "text": "Test query",
                "file_id": file_id,
                "model_choice": "gemini-pro",
            },
        )

    assert response.status_code == 200, f"Unexpected response: {response.json()}"
    assert response.json() == {"response": "Gemini response"}

    mock_gemini_handler.get_answer.assert_called_once_with("Test query", file_id)


def test_file_upload_existing(mock_gcs_handler):
    mock_gcs_handler.find_existing_file.return_value = "existing_file_id"
    mock_gcs_handler.download_files_from_folder_by_id.return_value = None

    with patch("rtl_rag_chatbot_api.app.initialize_chatbot") as mock_initialize_chatbot:
        mock_initialize_chatbot.return_value = None

        response = client.post(
            "/file/upload",
            files={"file": ("test.pdf", b"test content", "application/pdf")},
            data={"is_image": "false", "model_choice": "gpt-3.5-turbo"},
        )

    assert response.status_code == 200
    response_json = response.json()
    assert (
        "file_id" in response_json
    ), f"Response does not contain file_id: {response_json}"
    assert "File already exists" in response_json["message"]
    assert response_json["original_filename"] == "test.pdf"
    assert not response_json["is_image"]


def test_neighbors_gemini(mock_gemini_handler):
    file_id = "test_file_id"
    initialized_models[file_id] = {"type": "gemini", "model": mock_gemini_handler}

    # Patch the global gemini_handler
    with patch("rtl_rag_chatbot_api.app.gemini_handler", mock_gemini_handler):
        response = client.post(
            "/file/neighbors",
            json={"text": "Test query", "file_id": file_id, "n_neighbors": 2},
        )

    assert response.status_code == 200, f"Unexpected response: {response.json()}"
    assert response.json() == {"neighbors": ["Neighbor 1", "Neighbor 2"]}

    mock_gemini_handler.get_n_nearest_neighbours.assert_called_once_with(
        "Test query", file_id, 2
    )


def test_chat_endpoint_azure(mock_chatbot):
    file_id = "test_file_id"
    initialized_models[file_id] = {"type": "azure", "model": mock_chatbot}
    mock_chatbot.get_answer.return_value = "Test response"

    response = client.post(
        "/file/chat",
        json={
            "text": "Test query",
            "file_id": file_id,
            "model_choice": "gpt-3.5-turbo",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"response": "Test response"}


def test_chat_endpoint_not_initialized():
    response = client.post(
        "/file/chat",
        json={
            "text": "Test query",
            "file_id": "non_existent_file_id",
            "model_choice": "gpt-3.5-turbo",
        },
    )

    assert response.status_code == 404
    assert "Model not initialized" in response.json()["detail"]


def test_cleanup_files(mock_gcs_handler):
    mock_gcs_handler.cleanup_local_files.return_value = None

    response = client.post("/file/cleanup")

    assert response.status_code == 200
    assert response.json() == {"status": "Cleanup completed successfully"}
    mock_gcs_handler.cleanup_local_files.assert_called_once()


def test_cleanup_files_error(mock_gcs_handler):
    mock_gcs_handler.cleanup_local_files.side_effect = Exception("Test error")

    response = client.post("/file/cleanup")

    assert response.status_code == 500
    assert "An error occurred during cleanup" in response.json()["detail"]


def test_file_upload(mock_gcs_handler, mock_file, mock_chatbot):
    mock_gcs_handler.find_existing_file.return_value = None
    mock_gcs_handler.upload_to_gcs.return_value = None
    mock_gcs_handler.download_and_decrypt_file.return_value = "decrypted_file_path"

    with patch(
        "rtl_rag_chatbot_api.app.run_preprocessor"
    ) as mock_run_preprocessor, patch(
        "rtl_rag_chatbot_api.app.encrypt_file"
    ) as mock_encrypt_file:
        mock_run_preprocessor.return_value = None
        mock_encrypt_file.return_value = "encrypted_file_path"

        response = client.post(
            "/file/upload",
            files={"file": ("test.pdf", b"test content", "application/pdf")},
            data={"is_image": "false", "model_choice": "gpt-3.5-turbo"},
        )

    assert response.status_code == 200
    assert "file_id" in response.json()
    assert response.json()["original_filename"] == "test.pdf"
    assert response.json()["is_image"] is False


def test_neighbors_azure(mock_chatbot):
    file_id = "test_file_id"
    initialized_models[file_id] = {"type": "azure", "model": mock_chatbot}
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
    response = client.post(
        "/file/neighbors",
        json={
            "text": "Test query",
            "file_id": "non_existent_file_id",
            "n_neighbors": 2,
        },
    )

    assert response.status_code == 404
    assert "Model not initialized" in response.json()["detail"]


@patch("rtl_rag_chatbot_api.app.analyze_images")
def test_image_analyze(mock_analyze_images, mock_file):
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
    with patch(
        "rtl_rag_chatbot_api.app.analyze_images", side_effect=Exception("Test error")
    ):
        response = client.post(
            "/image/analyze",
            files={"file": ("test.jpg", b"test image content", "image/jpeg")},
        )

    assert response.status_code == 500
    assert "An error occurred during image analysis" in response.json()["detail"]


def test_model_initialize_azure():
    with patch("rtl_rag_chatbot_api.app.Chatbot"):
        response = client.post(
            "/model/initialize",
            json={"model_choice": "gpt-3.5-turbo", "file_id": "test_file_id"},
        )

    assert response.status_code == 200
    assert "Model gpt-3.5-turbo initialized successfully" in response.json()["message"]


def test_model_initialize_gemini():
    with patch("rtl_rag_chatbot_api.app.GeminiHandler"):
        response = client.post(
            "/model/initialize",
            json={"model_choice": "gemini-pro", "file_id": "test_file_id"},
        )

    assert response.status_code == 200
    assert "Model gemini-pro initialized successfully" in response.json()["message"]


def test_model_initialize_error():
    with patch("rtl_rag_chatbot_api.app.Chatbot", side_effect=Exception("Test error")):
        response = client.post(
            "/model/initialize",
            json={"model_choice": "gpt-3.5-turbo", "file_id": "test_file_id"},
        )

    assert response.status_code == 500
    assert "Error initializing model" in response.json()["detail"]
