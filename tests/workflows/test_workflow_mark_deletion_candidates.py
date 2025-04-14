from unittest.mock import patch

import pytest
from sqlalchemy import and_, create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from workflows.db.helpers import datetime_from_iso8601_timestamp, iso8601_timestamp_now
from workflows.db.tables import Base, User
from workflows.workflow import (
    get_users,
    get_users_deletion_candidates,
    mark_deletion_candidates,
)

# Create a session for a SQLite in-memory database
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def test_db_session():
    """"""
    try:
        db = TestingSessionLocal()

        # Read SQL statements from a file
        with open("tests/workflows/test-data.sql.runtime", "r") as f:
            sql_commands = f.read().split(";")

        Base.metadata.drop_all(bind=db.get_bind())
        Base.metadata.create_all(bind=db.get_bind())

        # Execute each SQL command
        for command in sql_commands:
            if command.strip():
                db.execute(text(command))

        db.commit()
        yield db

    except Exception as e:
        db.rollback()  # Rollback in case of any error
        raise e

    finally:
        db.close()


@pytest.fixture
def mock_log():
    with patch("workflows.workflow.log") as mock_log:
        yield mock_log


@patch("workflows.workflow.get_db_session")
def test_workflow_get_users(mock_get_db_session, test_db_session):
    """
    Tests whether workflows.workflow.get_users() returns all users in db.
    """
    # db session should use the test session. Use of __enter__ is necessary because
    # workflows.workflow.get_db_session has a `with get_db_session as ..:` statement.
    mock_get_db_session.return_value.__enter__.return_value = test_db_session

    users = get_users()
    assert len(users) == 8


@patch("workflows.workflow.get_db_session")
def test_workflow_get_users_deletion_candicates(mock_get_db_session, test_db_session):
    """
    Tests whether workflows.workflow.get_users_deletion_candicates() returns all users
    marked as deletion candidates in db.
    """
    # db session should use the test session. Use of __enter__ is necessary because
    # workflows.workflow.get_db_session has a `with get_db_session as ..:` statement.
    mock_get_db_session.return_value.__enter__.return_value = test_db_session

    users = get_users_deletion_candidates()
    assert len(users) == 2
    assert users[0].email == "user5@example.com"
    assert users[1].email == "user6@example.com"


@patch("workflows.workflow.get_db_session")
@patch("workflows.msgraph.is_user_account_enabled")  # Mock the Azure Graph API call
@patch("workflows.db.helpers.iso8601_timestamp_now")  # Mock the datetime helper
def test_mark_deletion_candidates(
    mock_iso8601_timestamp_now,
    mock_is_user_account_enabled,
    mock_get_db_session,
    test_db_session,
):
    """
    Mocks is_user_account_enabled() to return an inactive user: user2@example.com.
    This should lead mark_deletion_candidates to return more users that were marked as
    inactive in the function.
    """
    # db session should use the test session. Use of __enter__ is necessary because
    # workflows.workflow.get_db_session has a `with get_db_session as ..:` statement.
    mock_get_db_session.return_value.__enter__.return_value = test_db_session

    # "side_effect" simulates that the particular user is disabled in Azure,
    # because is_user_account_enabled takes email as input and returns True/False.
    mock_is_user_account_enabled.side_effect = (
        lambda email: email != "user2@example.com"
    )  # Only 'user2@example.com' is disabled

    mock_datetime_now = "2024-10-02T15:48:39.500Z"
    mock_iso8601_timestamp_now.return_value = mock_datetime_now

    # call the workflow function
    mark_deletion_candidates()

    # check the database for changes
    with test_db_session as session:
        for user in session.query(User).all():
            print(f"{user.email} -> {user.wf_deletion_candidate}")

        # query for the users who should be marked for deletion
        user = session.query(User).filter(User.email == "user2@example.com").first()
        assert (
            user.wf_deletion_candidate is True
        ), "User 2 should be marked for deletion."
        assert (
            user.wf_deletion_timestamp == mock_datetime_now
        ), "User 2 should have the correct deletion timestamp."

        # assert that other users are not marked for deletion
        for user in session.query(User).filter(
            and_(
                User.email != "user2@example.com",
                User.email != "user4@example.com",
                # already marked before test
                # already marked before test
                User.email != "user5@example.com",
                User.email != "user6@example.com",
                User.email != "user7@example.com",
                User.email != "user8@example.com",
            )
        ):
            assert (
                user.wf_deletion_candidate is False
            ), f"{user.email} should not be marked for deletion."


@patch("workflows.workflow.get_db_session")
@patch("workflows.msgraph.is_user_account_enabled")  # Mock the Azure Graph API call
@patch("workflows.db.helpers.iso8601_timestamp_now")  # Mock the datetime helper
def test_already_marked_users(
    mock_iso8601_timestamp_now,
    mock_is_user_account_enabled,
    mock_get_db_session,
    test_db_session,
):
    """"""
    # db session should use the test session. Use of __enter__ is necessary because
    # workflows.workflow.get_db_session has a `with get_db_session as ..:` statement.
    mock_get_db_session.return_value.__enter__.return_value = test_db_session

    # iso8601_timestamp_now needs to be mocked because otherwise iso8601_timestamp_now()
    # will return a new timestamp each time
    mock_datetime_now = iso8601_timestamp_now()
    mock_iso8601_timestamp_now.return_value = mock_datetime_now

    # call the workflow function
    mark_deletion_candidates()

    # assert user5 does not have a new timestamp, since it got already marked
    with test_db_session as db:
        user5 = db.query(User).filter(User.email == "user5@example.com").first()

        assert user5.wf_deletion_candidate is True
        # user5 already marked and timestamp must keep its value
        assert user5.wf_deletion_timestamp != iso8601_timestamp_now
        assert (
            datetime_from_iso8601_timestamp(iso8601_timestamp_now())
            - datetime_from_iso8601_timestamp(user5.wf_deletion_timestamp)
        ).days > 28, "User 5's deletion timestamp should be older than 4 weeks."
