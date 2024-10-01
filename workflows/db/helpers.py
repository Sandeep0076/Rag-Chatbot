from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from dateutil import parser

LOCAL_TIMEZONE = ZoneInfo("Europe/Berlin")


def iso8601_timestamp_now() -> str:
    """"""

    iso_8601_timestamp = (
        datetime.now(LOCAL_TIMEZONE).isoformat(timespec="milliseconds")[:-6] + "Z"
    )

    # e.g. '2024-09-27T14:51:16.951Z'
    return iso_8601_timestamp


def datetime_from_iso8601_timestamp(iso8601_timestamp: str) -> datetime:
    """"""
    return parser.isoparse(iso8601_timestamp)


def datetime_four_weeks_ago():
    """"""
    return datetime.now(LOCAL_TIMEZONE) - timedelta(weeks=4)
