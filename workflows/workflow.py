import argparse
import inspect
import logging
import os
from contextlib import contextmanager
from typing import Dict

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import workflows.db.helpers as db_helpers
import workflows.msgraph as msgraph
from workflows.db.tables import Conversation, Folder, Message, User

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s/%(funcName)s - %(message)s"
)
log = logging.getLogger(__name__)


@contextmanager
def get_db_session():
    """
    Dependency to get a SQLAlchemy session.

    Returns:
        Session: SQLAlchemy session.
    """

    engine = create_engine(os.getenv("DATABASE_URL", ""))
    Session = sessionmaker(bind=engine)
    db_session = Session()
    try:
        yield db_session
    finally:
        db_session.close()


def get_users():
    """"""
    with get_db_session() as session:
        users = session.query(User).all()

        return users


def get_users_deletion_candicates():
    """"""
    with get_db_session() as session:
        # get all users
        users = (
            session.query(User)
            .filter(User.wf_deletion_candidate, User.wf_deletion_timestamp is not None)
            .all()
        )

        # filter those with timestamp older than 4 weeks
        filtered_users = list(
            filter(
                lambda user: db_helpers.datetime_from_iso8601_timestamp(
                    user.wf_deletion_timestamp
                )
                <= db_helpers.datetime_four_weeks_ago(),
                users,
            )
        )

        log.info(
            f"Found {len(users)} deletion candidates, "
            f"{len(filtered_users)} of them marked older than 4 weeks: "
            f"{', '.join(list(map(lambda u: u.email, filtered_users)))}"
        )

        return filtered_users


def is_new_deletion_candidate(user: User, account_statuses: Dict = {}) -> bool:
    """
    Implements the logic that determines whether the user should be marked as deletion candidate.
    Requirements:
        a. the user must exist;
        b. there should be information about the status (not None);
        c. the user is marked inactive in msgraph; and
        d. the user is not already marked as inactive (would result in setting a new timestamp)
    """
    return (
        user.email in account_statuses  # a. we got the user
        and account_statuses.get(user.email)
        is not None  # b. we got the information (None if user does not exist)
        and account_statuses.get(user.email) is False  # c. user is inactive
        and user.wf_deletion_candidate
        is False  # d. user is not already set to be inactive
    )


def mark_deletion_candidates():
    """"""
    # 1. get the list of users from the chatbot database
    users = get_users()

    # 2. get account info from azure graph for all users
    user_emails = [user.email for user in users]
    account_statuses = {
        user_email: msgraph.is_user_account_enabled(user_email)
        for user_email in user_emails
    }

    # 3. mark those whose account is disabled
    users_marked = []
    with get_db_session() as session:
        for user in users:
            # mark those users for which , which are not marked, and which are not yet already marked
            if is_new_deletion_candidate(user, account_statuses):
                user.wf_deletion_candidate = True
                user.wf_deletion_timestamp = db_helpers.iso8601_timestamp_now()
                users_marked.append(user.email)

            # add user to the session for committing changes
            session.add(user)

        log.info(
            f"Found {len(users_marked)} deletion candidates: {', '.join(users_marked)}"
        )
        # commit changes to the database
        session.commit()


def delete_candidate_user_data():
    """
    Workflow to delete user data for those marked as deletion candidates.
    """
    # 1. Get the list of users marked for deletion from the chatbot database
    users = get_users_deletion_candicates()

    # 2. Get related data for each user and perform the necessary actions
    with get_db_session() as session:
        for user in users:
            try:
                print(f"loading user data for {user.email}")
                log.info(f"Loading user data for {user.email}")

                # 2. get the list of Messages related to the user's conversations
                messages = (
                    session.query(Message)
                    .join(Conversation)
                    .filter(Conversation.userEmail == user.email)
                    .all()
                )
                # 3. get the list of Conversations related to the user
                conversations = (
                    session.query(Conversation)
                    .filter(Conversation.userEmail == user.email)
                    .all()
                )
                # 4. get the list of Folders related to the user
                folders = session.query(Folder).filter(Folder.userId == user.id).all()

                log.info(
                    "About to delete: "
                    f"{len(messages)} messages, "
                    f"{len(conversations)} conversations, "
                    f"{len(folders)} folders."
                )

                # delete the messages, conversations, and folders related to the user
                for message in messages:
                    session.delete(message)

                for conversation in conversations:
                    session.delete(conversation)

                for folder in folders:
                    session.delete(folder)

                # Finally, delete the user itself
                session.delete(user)

                # Commit changes to delete the user and related data
                session.commit()

                # Log success
                log.info(f"Successfully deleted data for user {user.email}.")

            except Exception as e:
                # Rollback in case of any errors and log the failure
                session.rollback()
                log.error(f"Failed to delete data for user {user.email}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a specific task method from the workflow"
    )
    parser.add_argument(
        "--task", type=str, required=True, help="The name of the task method to run"
    )
    parser.add_argument(
        "--args", nargs="*", help="Arguments to pass to the task method"
    )

    args = parser.parse_args()

    # dynamically map the task name to a function
    try:
        method = globals()[args.task]
        if callable(method):
            # check how many arguments the method expects
            method_signature = inspect.signature(method)
            num_params = len(method_signature.parameters)

            if num_params > 0 and not args.args:
                print(
                    f"Task '{args.task}' requires {num_params} arguments, but none were provided."
                )
            # call the method with the right number of arguments
            elif args.args and num_params == len(args.args):
                val = method(*args.args)
                print(val)
            elif args.args and num_params != len(args.args):
                print(
                    f"Task '{args.task}' requires {num_params} arguments, but {len(args.args)} were provided."
                )
            else:
                # call method with no arguments if none are required
                val = method()
                print(val)
        else:
            print(f"'{args.task}' is not a valid callable method.")
    except KeyError:
        print(f"Task '{args.task}' not found.")
