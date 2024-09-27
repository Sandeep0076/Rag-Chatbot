import argparse
import inspect
import logging
import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import workflows.db.helpers as db_helpers
import workflows.msgraph as msgraph
from workflows.db.tables import User

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
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


def get_user_list():
    """"""
    with get_db_session() as session:
        users = session.query(User).all()

        return users


def mark_deletion_candidates():
    """"""
    # 1. get the list of users from the chatbot database
    users = get_user_list()

    # 2. get account info from azure graph for all users
    user_emails = [user.email for user in users]
    account_statuses = {
        user_email: msgraph.is_user_account_enabled(user_email)
        for user_email in user_emails
    }

    # 3. mark those whose account is disabled
    with get_db_session() as session:
        for user in users:
            if user.email in account_statuses and not account_statuses.get(user.email):
                user.wf_deletion_candidate = True
                user.wf_deletion_timestamp = db_helpers.get_datetime_now()

            # add user to the session for committing changes
            session.add(user)

        # commit changes to the database
        session.commit()


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
                method(*args.args)
            elif args.args and num_params != len(args.args):
                print(
                    f"Task '{args.task}' requires {num_params} arguments, but {len(args.args)} were provided."
                )
            else:
                # call method with no arguments if none are required
                method()
        else:
            print(f"'{args.task}' is not a valid callable method.")
    except KeyError:
        print(f"Task '{args.task}' not found.")
