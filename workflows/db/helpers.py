from datetime import datetime
from zoneinfo import ZoneInfo


def get_datetime_now() -> str:
    """"""
    local_timezone = ZoneInfo("Europe/Berlin")

    iso_8601_timestamp = (
        datetime.now(local_timezone).isoformat(timespec="milliseconds")[:-6] + "Z"
    )

    # e.g. '2024-09-27T14:51:16.951Z'
    return iso_8601_timestamp
