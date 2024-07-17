import pytest
from starlette.testclient import TestClient

from rtl_rag_chatbot_api.app import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_info(client: TestClient) -> None:
    response = client.get("/info")
    assert response.status_code == 200
