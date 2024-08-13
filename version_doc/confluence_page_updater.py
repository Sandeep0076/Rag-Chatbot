import json
import os
import subprocess
from datetime import datetime

import requests
from requests.auth import HTTPBasicAuth

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAST_COMMIT_FILE = os.path.join(SCRIPT_DIR, "last_updated_commit.txt")


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


def get_latest_commit_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def get_new_commits(last_commit_hash):
    git_log = (
        subprocess.check_output(
            ["git", "log", f"{last_commit_hash}..HEAD", "--format=%H|%an|%ae|%ct|%s"]
        )
        .decode()
        .strip()
        .split("\n")
    )
    return [log for log in git_log if log]


def extract_git_commits(new_commits, tags):
    table_rows = []
    for log_entry in new_commits:
        commit_hash, author_name, author_email, commit_time, subject = log_entry.split(
            "|"
        )
        tag = tags.get(commit_hash, "")
        commit_date = datetime.fromtimestamp(int(commit_time)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        full_message = (
            subprocess.check_output(["git", "log", "-1", "--format=%B", commit_hash])
            .decode()
            .strip()
        )
        if len(full_message) > 100:
            full_message = full_message[:97] + "..."
        table_rows.append(
            f"| {tag} | {author_name} <{author_email}> | {commit_date} | {full_message} |"
        )
    return "\n".join(table_rows)


def get_last_updated_commit():
    if not os.path.exists(LAST_COMMIT_FILE):
        return ""
    with open(LAST_COMMIT_FILE, "r") as f:
        return f.read().strip()


def save_last_updated_commit(commit_hash):
    with open(LAST_COMMIT_FILE, "w") as f:
        f.write(commit_hash)


def update_confluence_page(page_id, auth_user, auth_pass, base_url):
    last_updated_commit = get_last_updated_commit()
    latest_commit = get_latest_commit_hash()

    print(f"Last updated commit: {last_updated_commit}")
    print(f"Latest commit: {latest_commit}")

    if last_updated_commit == latest_commit:
        print("No new commits. Skipping update.")
        return

    new_commits = get_new_commits(last_updated_commit or "HEAD~1")
    print(f"New commits found: {len(new_commits)}")
    for commit in new_commits:
        print(commit)

    if not new_commits:
        print("No new commits. Skipping update.")
        return

    tags = get_tags()
    table_content = extract_git_commits(new_commits, tags)

    new_content = f"""h2. New Commits

||Tag/Version||Author||Date||Message||
{table_content}

----
"""

    response = requests.get(
        f"{base_url}/rest/api/content/{page_id}?expand=version,body.storage",
        auth=HTTPBasicAuth(auth_user, auth_pass),
    )
    page_info = response.json()
    current_version = page_info["version"]["number"]
    page_title = page_info["title"]
    existing_content = page_info["body"]["storage"]["value"]

    updated_content = new_content + existing_content

    update_data = {
        "version": {"number": current_version + 1},
        "type": "page",
        "title": page_title,
        "body": {"storage": {"value": updated_content, "representation": "wiki"}},
    }

    response = requests.put(
        f"{base_url}/rest/api/content/{page_id}",
        auth=HTTPBasicAuth(auth_user, auth_pass),
        headers={"Content-Type": "application/json"},
        data=json.dumps(update_data),
    )

    if response.status_code == 200:
        print("Confluence page updated successfully")
        save_last_updated_commit(latest_commit)
    else:
        print(f"Failed to update Confluence page: {response.text}")


if __name__ == "__main__":
    PAGE_ID = 652017700
    AUTH_USER = "sandeep.pathania@rtl.de"
    AUTH_PASS = os.environ.get("CONFLUENCE_TOKEN")
    BASE_URL = "https://rtldata.atlassian.net/wiki"

    update_confluence_page(PAGE_ID, AUTH_USER, AUTH_PASS, BASE_URL)
