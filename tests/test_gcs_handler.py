import json
from unittest.mock import MagicMock, patch

import pytest

from rtl_rag_chatbot_api.chatbot.gcs_handler import GCSHandler


@pytest.fixture
def mock_configs():
    class MockGCPResource:
        gcp_project = "mock_project"
        bucket_name = "mock_bucket"

    class MockConfigs:
        gcp_resource = MockGCPResource()

    return MockConfigs()


@pytest.fixture
def mock_gcs_handler():
    with patch("google.cloud.storage.Client") as mock_storage_client:
        mock_client_instance = MagicMock()
        mock_storage_client.return_value = mock_client_instance

        mock_bucket = MagicMock()
        mock_client_instance.get_bucket.return_value = mock_bucket

        mock_configs = MagicMock()
        mock_configs.gcp_resource.gcp_project = "test-project"
        mock_configs.gcp_resource.bucket_name = "test-bucket"

        gcs_handler = GCSHandler(mock_configs)

        return gcs_handler, mock_client_instance, mock_bucket


@patch("rtl_rag_chatbot_api.chatbot.gcs_handler.storage.Client")
def test_get_file_info(mock_storage_client, mock_gcs_handler):
    gcs_handler, _, mock_bucket = mock_gcs_handler  # Unpack fixture

    mock_blob = MagicMock()
    mock_blob.exists.return_value = True  # Simulate file exists
    mock_blob.download_as_bytes.return_value = json.dumps({"key": "value"}).encode(
        "utf-8"
    )

    mock_bucket.blob.return_value = (
        mock_blob  # Use mock_bucket instead of mock_storage_client
    )

    file_info = gcs_handler.get_file_info("mock_id")

    assert file_info == {"key": "value"}


@patch("rtl_rag_chatbot_api.chatbot.gcs_handler.storage.Client")
def test_update_file_info_username_append(mock_storage_client, mock_gcs_handler):
    gcs_handler, _, mock_bucket = mock_gcs_handler  # Unpack fixture

    mock_blob = MagicMock()
    mock_blob.exists.return_value = True  # Simulate that the file exists
    mock_blob.download_as_bytes.return_value = json.dumps({"username": "user1"}).encode(
        "utf-8"
    )
    mock_blob.upload_from_string = MagicMock()

    mock_bucket.blob.return_value = mock_blob  # Mocking the blob method

    gcs_handler.update_file_info("mock_id", {"username": "user1"})

    # Capture the uploaded data
    uploaded_json = json.loads(mock_blob.upload_from_string.call_args[0][0])

    assert uploaded_json["username"] == ["user1", "user1"]  # Expected transformation
    mock_blob.upload_from_string.assert_called_once_with(
        json.dumps(uploaded_json), content_type="application/json"
    )


@patch("rtl_rag_chatbot_api.chatbot.gcs_handler.storage.Client")
def test_update_file_info_username_existing_array(
    mock_storage_client, mock_gcs_handler
):
    gcs_handler, _, mock_bucket = mock_gcs_handler  # Unpack fixture

    mock_blob = MagicMock()
    mock_blob.exists.return_value = True  # Simulate that the file exists
    mock_blob.download_as_bytes.return_value = json.dumps(
        {"username": ["user1"]}
    ).encode("utf-8")
    mock_blob.upload_from_string = MagicMock()

    mock_bucket.blob.return_value = mock_blob  # Mocking the blob method

    gcs_handler.update_file_info("mock_id", {"username": "user2"})

    # Capture the uploaded data
    uploaded_json = json.loads(mock_blob.upload_from_string.call_args[0][0])

    assert uploaded_json["username"] == ["user1", "user2"]  # Expected transformation
    mock_blob.upload_from_string.assert_called_once_with(
        json.dumps(uploaded_json), content_type="application/json"
    )
