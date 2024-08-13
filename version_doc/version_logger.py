import csv
import os
import subprocess
from datetime import datetime


def extract_git_commits(output_dir):
    # Get the current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    output_file = os.path.join(output_dir, f"git_commits_{current_date}.csv")

    # Get the latest commit hash from the previous run
    last_processed_commit = get_last_processed_commit(output_dir)

    # Get all tags
    tags = (
        subprocess.check_output(["git", "tag", "-l", "--sort=-v:refname"])
        .decode("utf-8")
        .strip()
        .split("\n")
    )

    # Get all branches
    branches = (
        subprocess.check_output(["git", "branch", "-r", "--format=%(refname:short)"])
        .decode("utf-8")
        .strip()
        .split("\n")
    )
    branches = [branch.strip() for branch in branches if branch.strip()]

    # Prepare the CSV file
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Tag/Branch", "Commit Hash", "Author", "Date", "Message"])

        # Process tags
        for tag in tags:
            process_ref(tag, writer, last_processed_commit)

        # Process branches
        for branch in branches:
            process_ref(branch, writer, last_processed_commit)

    print(f"Commit information has been saved to {output_file}")

    # Update the last processed commit
    update_last_processed_commit(output_dir)


def process_ref(ref, writer, last_processed_commit):
    # Get commit information for the ref
    commit_log = (
        subprocess.check_output(["git", "log", ref, "--format=%H|%an|%ad|%s"])
        .decode("utf-8")
        .strip()
        .split("\n")
    )

    for commit in commit_log:
        commit_info = commit.split("|")

        # Check if this commit has already been processed
        if commit_info[0] == last_processed_commit:
            return

        writer.writerow([ref] + commit_info)


def get_last_processed_commit(output_dir):
    last_commit_file = os.path.join(output_dir, "last_processed_commit.txt")
    if os.path.exists(last_commit_file):
        with open(last_commit_file, "r") as f:
            return f.read().strip()
    return None


def update_last_processed_commit(output_dir):
    last_commit_file = os.path.join(output_dir, "last_processed_commit.txt")
    latest_commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    )
    with open(last_commit_file, "w") as f:
        f.write(latest_commit)


if __name__ == "__main__":
    output_directory = "version_doc"
    extract_git_commits(output_directory)
