import os
from datetime import datetime, timedelta

os.environ["TOKEN_URL"] = "some url"
os.environ["JWKS_URL"] = "https://some-JWKS_URL.url"
os.environ["CLIENT_ID"] = "some client id"


def replace_timestamps_in_test_sql_file():
    # Calculate the timestamp for the current date - 2 weeks
    timestamp_format = "%Y-%m-%dT%H:%M:%S.000Z"
    timestamp_less_than_4_weeks = (datetime.now() - timedelta(weeks=2)).strftime(
        timestamp_format
    )
    timestamp_more_than_4_weeks = (datetime.now() - timedelta(weeks=5)).strftime(
        timestamp_format
    )

    sql_file_path = "tests/workflows/test-data.sql"

    # Read the contents of the file
    with open(sql_file_path, "r") as file:
        file_data = file.read()

    # Replace the TIMESTAMP_LESS_THAN_4_WEEKS string with the new timestamp
    file_data = file_data.replace(
        "TIMESTAMP_LESS_THAN_4_WEEKS", f"'{timestamp_less_than_4_weeks}'"
    )
    file_data = file_data.replace(
        "TIMESTAMP_MORE_THAN_4_WEEKS", f"'{timestamp_more_than_4_weeks}'"
    )

    # Write the modified contents back to the file
    with open(sql_file_path, "w") as file:
        file.write(file_data)


replace_timestamps_in_test_sql_file()
