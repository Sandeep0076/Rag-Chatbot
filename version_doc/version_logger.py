import csv
import os
import subprocess
from datetime import datetime


def get_last_processed_commit(output_dir):
    last_commit_file = os.path.join(output_dir, "last_processed_commit.txt")
    if os.path.exists(last_commit_file):
        with open(last_commit_file, "r") as f:
            return f.read().strip()
    return None


def extract_git_commits(output_dir):
    current_date = datetime.now().strftime("%Y-%m-%d")
    output_file = os.path.join(output_dir, f"git_commits_{current_date}.csv")
    last_processed_commit = get_last_processed_commit(output_dir)

    branches = (
        subprocess.check_output(["git", "branch", "--format=%(refname:short)"])
        .decode("utf-8")
        .strip()
        .split("\n")
    )
    branches = [branch.strip() for branch in branches if branch.strip()]

    all_commits = []

    for branch in branches:
        all_commits.extend(process_branch(branch, last_processed_commit))

    # Sort commits based on date
    all_commits.sort(
        key=lambda x: datetime.strptime(x[3], "%a %b %d %H:%M:%S %Y %z"), reverse=True
    )

    if all_commits:
        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Branch", "Commit Hash", "Author", "Date", "Message"])
            writer.writerows(all_commits)

        print(f"Sorted commit information has been saved to {output_file}")
        update_last_processed_commit(
            output_dir, all_commits[0][1]
        )  # Update with the most recent commit hash
    else:
        print("No new commits found since the last processed commit.")


def process_branch(branch, last_processed_commit):
    if last_processed_commit:
        commit_log = (
            subprocess.check_output(
                [
                    "git",
                    "log",
                    f"{last_processed_commit}..{branch}",
                    "--format=%H|%an|%ad|%s",
                ]
            )
            .decode("utf-8")
            .strip()
            .split("\n")
        )
    else:
        commit_log = (
            subprocess.check_output(["git", "log", branch, "--format=%H|%an|%ad|%s"])
            .decode("utf-8")
            .strip()
            .split("\n")
        )

    branch_commits = []

    for commit in commit_log:
        if commit:  # Check if the commit is not an empty string
            commit_info = commit.split("|")
            tags = (
                subprocess.check_output(["git", "tag", "--points-at", commit_info[0]])
                .decode("utf-8")
                .strip()
            )
            branch_or_tag = f"{branch} ({tags})" if tags else branch
            branch_commits.append([branch_or_tag] + commit_info)

    return branch_commits


def update_last_processed_commit(output_dir, latest_commit):
    last_commit_file = os.path.join(output_dir, "last_processed_commit.txt")
    with open(last_commit_file, "w") as f:
        f.write(latest_commit)


if __name__ == "__main__":
    output_directory = "version_doc"
    extract_git_commits(output_directory)
