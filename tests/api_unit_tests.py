from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from rtl_rag_chatbot_api.app import app
from rtl_rag_chatbot_api.chatbot.embedding_handler import EmbeddingHandler
from rtl_rag_chatbot_api.chatbot.gemini_handler import GeminiHandler

client = TestClient(app)


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


@patch("rtl_rag_chatbot_api.app.file_handler.process_file")
def test_file_upload(mock_process_file):
    mock_process_file.return_value = {
        "file_id": "test_file_id",
        "status": "new",
        "message": "File uploaded successfully",
    }
    response = client.post(
        "/file/upload",
        files={"file": ("test.pdf", b"test content", "application/pdf")},
        data={"is_image": "false"},
    )
    assert response.status_code == 200
    assert response.json()["file_id"] == "test_file_id"


@patch("rtl_rag_chatbot_api.app.model_handler.initialize_model")
@patch("rtl_rag_chatbot_api.app.embedding_handler.get_embeddings_info")
def test_initialize_model(mock_get_embeddings_info, mock_initialize_model):
    mock_get_embeddings_info.return_value = {"embeddings": {"azure": "completed"}}
    mock_initialize_model.return_value = None
    response = client.post(
        "/model/initialize",
        json={"model_choice": "gpt-3.5-turbo", "file_id": "test_file_id"},
    )
    assert response.status_code == 200
    assert "initialized successfully" in response.json()["message"]


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


@pytest.mark.asyncio
async def test_create_embeddings():
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


# -------


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
