from datetime import datetime, timedelta
from typing import List
from zoneinfo import ZoneInfo

from dateutil import parser

LOCAL_TIMEZONE = ZoneInfo("Europe/Berlin")


def iso8601_timestamp_now() -> str:
    """"""

    iso_8601_timestamp = (
        # e.g. '2024-10-10T08:59:59.552+02:00'[:-6] -> '2024-10-10T08:59:59.552Z'
        datetime.now(LOCAL_TIMEZONE).isoformat(timespec="milliseconds")[:-6]
        + "Z"
    )

    # e.g. '2024-09-27T14:51:16.951Z'
    return iso_8601_timestamp


def datetime_from_iso8601_timestamp(iso8601_timestamp: str) -> datetime:
    """"""
    return parser.isoparse(iso8601_timestamp)


def datetime_four_weeks_ago():
    """"""
    return datetime.now(LOCAL_TIMEZONE) - timedelta(weeks=4)


def filter_older_than_4_weeks(users: List) -> List:
    """
    Keeps those users in the list for which the deletion timestamp of the user
    is smaller then the current_date - 4 weeks (so the timestamp is older than 4 weeks).
    Only compared if there is a deletion timestamp for the user; else the user is removed
    from the list.
    """
    # filter those with timestamp older than 4 weeks
    return list(
        filter(
            lambda user: datetime_from_iso8601_timestamp(user.wf_deletion_timestamp)
            <= datetime_four_weeks_ago()
            if user.wf_deletion_timestamp
            else False,
            users,
        )
    )
