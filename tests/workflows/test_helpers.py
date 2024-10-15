from datetime import datetime
from unittest.mock import patch

from workflows.db.helpers import filter_older_than_4_weeks


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
