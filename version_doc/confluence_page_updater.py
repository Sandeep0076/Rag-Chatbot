import json
import os
import subprocess
from datetime import datetime

import requests
from prettytable import PrettyTable
from requests.auth import HTTPBasicAuth


def get_tags():
    tags = (
        subprocess.check_output(["git", "show-ref", "--tags", "-d"])
        .decode()
        .strip()
        .split("\n")
    )
    tag_dict = {}
    for tag in tags:
        if tag:
            commit_hash, ref = tag.split()
            tag_name = ref.split("/")[-1]
            if not tag_name.endswith("^{}"):
                tag_dict[commit_hash] = tag_name
    return tag_dict


def extract_git_commits():
    # Get all tags
    tags = get_tags()

    # Get all commits
    git_log = (
        subprocess.check_output(["git", "log", "--all", "--format=%H|%an|%ae|%ct|%s"])
        .decode()
        .strip()
        .split("\n")
    )

    table = PrettyTable()
    table.field_names = ["Tag/Version", "Author", "Date", "Message"]
    table.align = "l"
    table.max_width = 50

    for log_entry in git_log:
        commit_hash, author_name, author_email, commit_time, subject = log_entry.split(
            "|"
        )

        # Get tag if exists
        tag = tags.get(commit_hash, "")

        # Convert Unix timestamp to readable date
        commit_date = datetime.fromtimestamp(int(commit_time)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Get full commit message
        full_message = (
            subprocess.check_output(["git", "log", "-1", "--format=%B", commit_hash])
            .decode()
            .strip()
        )

        # Truncate message if it's too long
        if len(full_message) > 100:
            full_message = full_message[:97] + "..."

        table.add_row(
            [tag, f"{author_name} <{author_email}>", commit_date, full_message]
        )

    return table


def update_confluence_page(page_id, auth_user, auth_pass, base_url):
    # Extract git commits
    table = extract_git_commits()

    # Prepare the content
    content = f"h1. Git Commit History\n\n{table}"

    # Get the current page info
    response = requests.get(
        f"{base_url}/rest/api/content/{page_id}?expand=version",
        auth=HTTPBasicAuth(auth_user, auth_pass),
    )
    page_info = response.json()
    current_version = page_info["version"]["number"]
    page_title = page_info["title"]

    # Prepare the update payload
    update_data = {
        "version": {"number": current_version + 1},
        "type": "page",
        "title": page_title,
        "body": {"storage": {"value": content, "representation": "wiki"}},
    }

    # Send the update request
    response = requests.put(
        f"{base_url}/rest/api/content/{page_id}",
        auth=HTTPBasicAuth(auth_user, auth_pass),
        headers={"Content-Type": "application/json"},
        data=json.dumps(update_data),
    )

    if response.status_code == 200:
        print("Confluence page updated successfully")
    else:
        print(f"Failed to update Confluence page: {response.text}")


if __name__ == "__main__":
    PAGE_ID = 652017700
    AUTH_USER = "sandeep.pathania@rtl.de"
    AUTH_PASS = os.environ.get("CONFLUENCE_TOKEN")
    BASE_URL = "https://rtldata.atlassian.net/wiki"

    update_confluence_page(PAGE_ID, AUTH_USER, AUTH_PASS, BASE_URL)
