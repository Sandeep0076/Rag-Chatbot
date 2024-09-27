from datetime import datetime
from zoneinfo import ZoneInfo


def get_datetime_now():
    """"""

    local_timezone = ZoneInfo("Europe/Berlin")
    timestamp = (
        datetime.now(local_timezone).isoformat(timespec="milliseconds")[:-6] + "Z"
    )

    return timestamp
