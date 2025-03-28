from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from workflows.db.helpers import filter_older_than_4_weeks
from workflows.gcs.helpers import delete_embeddings


@pytest.fixture
def mock_log():
    with patch("workflows.gcs.helpers.log") as mock_log:
        yield mock_log


@patch("workflows.db.helpers.datetime_from_iso8601_timestamp")
@patch("workflows.db.helpers.datetime_four_weeks_ago")
def test_filter_older_than_4_weeks(
    mock_datetime_four_weeks_ago, mock_datetime_from_iso8601_timestamp
):
    # setup mock for datetime four weeks ago and
    # mock the datetime conversion for the first user
    mock_datetime_four_weeks_ago.return_value = datetime(
        2024, 9, 12, 12, 0, 0
    )  # assume today is October 10th, 2024
    mock_datetime_from_iso8601_timestamp.side_effect = (
        lambda timestamp: datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    )

    # create two mock user objects
    class MockUser:
        def __init__(self, email, wf_deletion_timestamp):
            self.email = email
            self.wf_deletion_timestamp = wf_deletion_timestamp

    user1 = MockUser("user1@example.com", "2024-09-10T12:00:00Z")  # older than 4 weeks
    user2 = MockUser("user2@example.com", None)  # no deletion timestamp
    user3 = MockUser("user3@example.com", "2024-10-01T12:00:00Z")  # just one day older

    users = [user1, user2, user3]

    # call the test
    filtered_users = filter_older_than_4_weeks(users)

    assert len(filtered_users) == 1
    assert (
        filtered_users[0].email == "user1@example.com"
    )  # only user1 should be in the filtered list


@patch("workflows.gcs.helpers.bucket.blob")
@patch("workflows.gcs.helpers.bucket.delete_blobs")
@patch("workflows.gcs.helpers.bucket.list_blobs")
def test_delete_embeddings__no_user_left(
    mock_list_blobs, mock_delete_blobs, mock_blob, mock_log
):
    """
    Test the delete_embeddings function when one user is present in file_info.json.
    This should delete the user from file_info.json and eventually delete the blobs.

    File structure in GCS:
    file-embeddings/
        embedding1/         <- folder to be deleted
            azure/
            google/
            file_info.json  <- file to be updated (but eventually deleted)
        embedding2/
            ...
    """
    blob_prefix = "file-embeddings/embedding1"
    # rebuild the gcs folder structure
    mock_blob_azure_subfolder = MagicMock()
    mock_blob_azure_subfolder.name = f"{blob_prefix}/azure"
    mock_blob_google_subfolder = MagicMock()
    mock_blob_google_subfolder.name = f"{blob_prefix}/google"
    # mock file_info.json for embedding 1
    mock_file_info_blob = MagicMock()
    mock_file_info_blob.name = f"{blob_prefix}/file_info.json"
    mock_file_info_blob.exists.return_value = True
    mock_file_info_blob.download_as_text.return_value = (
        '{"username": ["user1@example.com"]}'
    )
    mock_blob.return_value = mock_file_info_blob

    # mock list_blobs
    blob_list = [
        mock_blob_azure_subfolder,
        mock_blob_google_subfolder,
        mock_file_info_blob,
    ]
    mock_list_blobs.return_value = blob_list

    # call function
    delete_embeddings("embedding1", "user1@example.com")

    # assertions
    mock_blob.assert_called_once()
    mock_file_info_blob.upload_from_string.assert_not_called()  # usernames empty, folder deleted
    mock_delete_blobs.assert_called_once_with(blob_list)

    # ensure no errors logged
    mock_log.error.assert_not_called()

    # Ensure logs are written
    mock_log.info.assert_any_call(
        "Removed user1@example.com from file_info.json for file_id: embedding1"
    )
    mock_log.info.assert_any_call(
        "Successfully deleted whole embeddings folder for file_id: embedding1"
    )


@patch("workflows.gcs.helpers.bucket.blob")
@patch("workflows.gcs.helpers.bucket.delete_blobs")
@patch("workflows.gcs.helpers.bucket.list_blobs")
def test_delete_embeddings__users_left(
    mock_list_blobs, mock_delete_blobs, mock_blob, mock_log
):
    """
    Test the delete_embeddings function when multiple users are present in file_info.json.
    This should delete the user from file_info.json and update it in the bucket.
    The embeddings folder should not be deleted.

    File structure in GCS:
    file-embeddings/
        embedding1/         <- folder to be deleted
            azure/
            google/
            file_info.json  <- file to be updated, not deleted
        embedding2/
            ...
    """
    blob_prefix = "file-embeddings/embedding1"
    # rebuild the gcs folder structure
    mock_blob_azure_subfolder = MagicMock()
    mock_blob_azure_subfolder.name = f"{blob_prefix}/azure"
    mock_blob_google_subfolder = MagicMock()
    mock_blob_google_subfolder.name = f"{blob_prefix}/google"
    # mock file_info.json for embedding 1
    mock_file_info_blob = MagicMock()
    mock_file_info_blob.name = f"{blob_prefix}/file_info.json"
    mock_file_info_blob.exists.return_value = True
    mock_file_info_blob.download_as_text.return_value = (
        '{"username": ["user1@example.com", "user2@example.com"]}'
    )
    mock_blob.return_value = mock_file_info_blob

    # mock list_blobs
    blob_list = [
        mock_blob_azure_subfolder,
        mock_blob_google_subfolder,
        mock_file_info_blob,
    ]
    mock_list_blobs.return_value = blob_list

    # call function
    delete_embeddings("embedding1", "user1@example.com")

    # assertions
    mock_blob.assert_called_once()
    mock_file_info_blob.upload_from_string.assert_called_once_with(
        '{"username": ["user2@example.com"]}', content_type="application/json"
    )  # one user left
    mock_delete_blobs.assert_not_called()  # folder not deleted

    # ensure no errors logged
    mock_log.error.assert_not_called()

    # Ensure logs are written
    mock_log.info.assert_any_call(
        "Removed user1@example.com from file_info.json for file_id: embedding1"
    )
    mock_log.info.assert_any_call(
        "Updated file_info.json and removed user1@example.com from list for file_id: embedding1"
    )
