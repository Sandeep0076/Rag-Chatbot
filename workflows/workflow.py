import argparse
import inspect
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Dict, List

from sqlalchemy import and_, create_engine, distinct, or_, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

import workflows.db.helpers as db_helpers
import workflows.msgraph as msgraph
from workflows.app_config import config
from workflows.db.tables import (
    Citation,
    Conversation,
    Folder,
    Message,
    Prompt,
    PromptFolder,
    User,
)
from workflows.gcs.helpers import delete_embeddings, file_present_in_gcp

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s/%(funcName)s - %(message)s",
)
log = logging.getLogger(__name__)


@contextmanager
def get_db_session():
    """
    Dependency to get a SQLAlchemy session.

    Returns:
        Session: SQLAlchemy session.
    """
    engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///:memory:"))
    Session = sessionmaker(bind=engine)

    attempts = 0
    max_attempts = 10

    while attempts < max_attempts:
        # Try to establish a connection to the database
        try:
            attempts += 1
            db_session = Session()
            # Try to execute a simple query to check the connection
            db_session.execute(text("SELECT 1"))
            break
        except OperationalError:
            log.warning("Database connection failed. Retrying in 5 seconds...")
            time.sleep(5)
        finally:
            db_session.close()

    try:
        db_session = Session()
        yield db_session
    except Exception as e:
        log.error(f"Error in session: {e}")
    finally:
        db_session.close()


def get_users():
    """"""
    with get_db_session() as session:
        users = (
            # id = "dXNlci5kZWxldGVkQHJ0bC5kZQo=" is the operational user: user.deleted@rtl.de
            session.query(User)
            .filter(User.id != config.workflow.db_deleted_user_id)
            .all()
        )

        return users


def get_users_deletion_candidates():
    """
    Returns a list of users marked as deletion candidates in database which are marked for more than 4 weeks.
    """
    try:
        with get_db_session() as session:
            # get all users
            users = (
                session.query(User)
                .filter(
                    and_(
                        User.wf_deletion_candidate,
                        User.wf_deletion_timestamp.isnot(None),
                        # id = "dXNlci5kZWxldGVkQHJ0bC5kZQo=" is the operational user: user.deleted@rtl.de
                        User.id != config.workflow.db_deleted_user_id,
                    )
                )
                .all()
            )

            filtered_users = db_helpers.filter_older_than_4_weeks(users)

            log.info(
                f"Found {len(users)} deletion candidates, "
                f"{len(filtered_users)} of them marked older than 4 weeks: "
                f"{', '.join(list(map(lambda u: u.email, filtered_users)))}"
            )

            return filtered_users
    except Exception as e:
        log.error(f"Failed to retrieve deletion candidates. Original error: {e}")
        return []


def get_user_fileids(candidates: List[User]):
    """
    This function retrieves the file ids associated with the users passed into this function.
    """

    with get_db_session() as session:
        try:
            user_file_ids = (
                session.query(distinct(Conversation.fileId), Conversation.userEmail)
                .filter(
                    Conversation.userEmail.in_(
                        list(map(lambda u: u.email, candidates))
                    ),
                    Conversation.fileId.isnot(None),
                )
                .all()
            )

            log.info(f"Found {len(user_file_ids)} file ids to delete embeddings for.")

            # returns a mapping of file ids to user emails
            # e.g. [(file_id1, user_email1), (file_id2, user_email1), ...]
            return user_file_ids
        except Exception as e:
            log.error(
                f"Failed to retrieve file ids. Returning empty list. Original error: {e}"
            )
            return []


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
        and user.wf_deletion_candidate is False  # d. user is not already marked
    )


def is_user_got_reactivated(user: User, account_statuses: Dict = {}) -> bool:
    """"""
    return (
        user.email in account_statuses  # a. we got the user
        and account_statuses.get(user.email)
        is not None  # b. we got the information (None if user does not exist)
        and account_statuses.get(user.email) is True  # c. user is active
        and user.wf_deletion_candidate is True  # d. user is already marked
    )


def mark_deletion_candidates():
    """
    Workflow to mark users as deletion candidates.
    """
    log.info("Starting marking of deletion candidates.")

    # 1. get the list of users from the chatbot database
    users = get_users()

    # 2. get account info from azure graph for all users
    log.info("Getting account statuses from Azure Graph for all users.")
    user_emails = [user.email for user in users]
    account_statuses = {
        user_email: msgraph.is_user_account_enabled(user_email)
        for user_email in user_emails
    }

    # 3. mark those whose account is disabled
    users_marked = []
    with get_db_session() as session:
        for user in users:
            # Check if the user is inactive
            if is_new_deletion_candidate(user, account_statuses):
                user.wf_deletion_candidate = True
                user.wf_deletion_timestamp = db_helpers.iso8601_timestamp_now()
                users_marked.append(user.email)

            # Check if the user is reactivated
            elif is_user_got_reactivated(user, account_statuses):
                # Ensure re-actived users are not marked as deletion candidates
                user.wf_deletion_candidate = False
                user.wf_deletion_timestamp = None

            # Add user to the session for committing changes
            session.add(user)

        log.info(
            f"Found {len(users_marked)} deletion candidates: {', '.join(users_marked)}"
        )
        # commit changes to the database
        session.commit()

    log.info("Workflow step marking of deletion candidates completed.")


def delete_candidate_user_embeddings():
    """
    Workflow to delete user embeddings for those marked as deletion candidates.
    """
    log.info("Starting deletion of user embeddings.")

    # 1. Get the list of users marked for deletion from the chatbot database
    # 2. and get the list of file ids for each user
    users = get_users_deletion_candidates()
    file_ids_by_user_emails = get_user_fileids(users)

    if not file_ids_by_user_emails:
        log.warning("No file ids found for deletion candidates: nothing to delete.")
        log.info("Workflow step deletion of user embeddings completed with warnings.")
        return

    for file_id, user_email in file_ids_by_user_emails:
        # e.g. chatbot-storage-dev-gcs-eu/file-embeddings/07806aff-478b-4a45-8725-9bac7935975e

        # 3. check if the file exists in GCP
        # and delete the embeddings if they exist
        if not file_present_in_gcp(
            bucket_prefix=f"{config.gcp.embeddings_root_folder}/{file_id}",
        ):
            log.warning(f"Embeddings not found for file id '{file_id}'")
            continue

        delete_embeddings(file_id=file_id, user_email=user_email)

    log.info("Workflow step deletion of user embeddings completed.")


def anonymise_shared_prompts():
    """"""
    # 1. Get the list of users marked for deletion from the chatbot database
    # 2. and replace the user id with the operational user id
    users = get_users_deletion_candidates()

    if not users:
        log.info("No users found for anonymisation: nothing to anonymise.")
        log.info("Workflow step anonymising shared prompts completed.")
        return

    with get_db_session() as session:
        try:
            # replace all prompts created by the users in the list with the operational user id
            session.query(Prompt).filter(
                and_(
                    Prompt.published,
                    Prompt.userId.in_([user.id for user in users]),
                )
            ).update(
                {Prompt.userId: config.workflow.db_deleted_user_id},
                synchronize_session=False,
            )

            session.commit()

        except Exception as e:
            # Rollback in case of any errors and log the failure
            session.rollback()
            log.error(f"Failed to anonymise shared prompts for users: {e}")

    log.info("Workflow step anonymising shared prompts completed.")


def delete_private_prompts():
    """"""

    # 1. Get the list of users marked for deletion from the chatbot database
    # 2. and delete the private prompts
    users = get_users_deletion_candidates()

    if not users:
        log.info("No users found: no prompts and related objects to delete.")
        log.info("Workflow step 'deleting private prompts' completed.")
        return

    with get_db_session() as session:
        try:
            # NOTE: PromptLikes do not have to be deleted, since private Prompts did not receive any likes.
            # NOTE: PromptTags do not have to be deleted, since those are unique and should remain in the database.

            # Query for prompts that are not published (where Prompt.published is None) → delete
            prompts = (
                session.query(Prompt)
                .filter(
                    and_(
                        Prompt.userId.in_([user.id for user in users]),
                        # private prompts are not published
                        or_(Prompt.published.is_(None), Prompt.published.isnot(True)),
                    )
                )
                .all()
            )

            log.info("About to delete: " f"{len(prompts)} private Prompts.")

            # delete the prompts ..
            for prompt in prompts:
                session.delete(prompt)

            session.commit()

            # Query for folders that have no published prompts (where Prompt.published is None) → delete
            promptFolders = (
                session.query(PromptFolder)
                .join(Prompt)
                .filter(
                    and_(
                        Prompt.userId.in_([user.id for user in users]),
                        # private prompts are not published
                        or_(Prompt.published.is_(None), Prompt.published.isnot(True)),
                    )
                )
                .distinct()
                .all()
            )

            log.info(
                "About to delete: "
                f"{len(promptFolders)} PromptFolders related to private Prompts."
            )

            # .. and promptFolders related to the user
            for promptFolder in promptFolders:
                session.delete(promptFolder)

            # Commit changes
            session.commit()

            # Log success
            log.info("Successfully deleted Prompts and PromptFolders for users.")

        except Exception as e:
            session.rollback()
            log.error(f"Failed to delete prompts and folders for users: {e}")
            return

    log.info("Workflow step 'deleting private prompts' completed.")


def delete_candidate_user_data():
    """
    Workflow to delete user data for those marked as deletion candidates.
    """
    log.info("Starting deletion of user data.")

    # 1. Get the list of users marked for deletion from the chatbot database
    users = get_users_deletion_candidates()

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
                # 5. get the list of Citations related to user's messages
                citations = (
                    session.query(Citation)
                    .filter(
                        Citation.messageId.in_([message.id for message in messages])
                    )
                    .all()
                )

                log.info(
                    "About to delete: "
                    f"{len(citations)} citations, "
                    f"{len(messages)} messages, "
                    f"{len(conversations)} conversations, "
                    f"{len(folders)} folders."
                )

                # delete the citations related to the user's messages
                for citation in citations:
                    session.delete(citation)

                # delete the messages, conversations, and folders related to the user
                for message in messages:
                    session.delete(message)

                for conversation in conversations:
                    session.delete(conversation)

                for folder in folders:
                    session.delete(folder)

                # NOTE private prompts and related objects are deleted in separate step: delete_private_prompts()
                # NOTE anonmising shared prompts is handled in separate workflow step: anonymise_shared_prompts()

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

    log.info("Workflow step deletion of user data completed.")


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
        log.info(f"Program invoked with workflow step argument: '{args.task}'")
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
            elif args.args and num_params != len(args.args):
                print(
                    f"Task '{args.task}' requires {num_params} arguments, but {len(args.args)} were provided."
                )
            else:
                # call method with no arguments if none are required
                val = method()
        else:
            print(f"'{args.task}' is not a valid callable method.")
    except KeyError:
        print(f"Task '{args.task}' not found.")
