from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from workflows.app_config import config
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


@patch("workflows.gcs.helpers.bucket.list_blobs")
def test_delete_embeddings(mock_list_blobs, mock_log):
    # Setup mock blobs
    mock_blob1 = MagicMock()
    mock_blob1.name = "embedding1"
    mock_blob2 = MagicMock()
    mock_blob2.name = "embedding2"
    mock_list_blobs.return_value = [mock_blob1, mock_blob2]

    # Call the function
    delete_embeddings("file1")

    # Assertions
    mock_list_blobs.assert_called_once_with(
        prefix=f"{config.gcp.embeddings_root_folder}/file1/"
    )
    assert mock_blob1.delete.call_count == 1
    assert mock_blob2.delete.call_count == 1

    # Ensure no error logs are written
    mock_log.error.assert_not_called()

    # Ensure logs are written
    mock_log.info.assert_any_call("Deleted blob: embedding1")
    mock_log.info.assert_any_call("Deleted blob: embedding2")
    mock_log.info.assert_any_call(
        "Successfully deleted all embeddings for file_id: file1"
    )


@patch("workflows.gcs.helpers.bucket.list_blobs")
def test_delete_embeddings_with_exception(mock_list_blobs, mock_log):
    # Setup mock blobs
    mock_blob1 = MagicMock()
    mock_blob1.name = "embeddings/file1/embedding1"
    mock_blob1.delete.side_effect = Exception("Deletion error")
    mock_list_blobs.return_value = [mock_blob1]

    delete_embeddings("test_file_id")

    # Assertions
    mock_list_blobs.assert_called_once_with(
        prefix=f"{config.gcp.embeddings_root_folder}/test_file_id/"
    )
    assert mock_blob1.delete.call_count == 1

    # Ensure log error was triggered
    mock_log.error.assert_any_call(
        f"Error deleting blob {mock_blob1.name}: Deletion error"
    )
