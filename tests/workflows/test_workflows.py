from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from workflows.db.helpers import get_datetime_now
from workflows.db.tables import Base, User
from workflows.workflow import mark_deletion_candidates

# Create a session for a SQLite in-memory database
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def get_db_session():
    """"""
    try:
        db = TestingSessionLocal()

        # Read SQL statements from a file
        with open("tests/workflows/test-data.sql", "r") as f:
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


def test_user_count(get_db_session):
    """Test to ensure that there are 5 users in the User table."""

    with get_db_session as db:
        user_count = db.query(
            User
        ).count()  # Query the number of users in the User table
        assert user_count == 5, f"Expected 5 users, but found {user_count}"


@pytest.fixture
def mock_user_data(get_db_session):
    """Fixture to set up user data in the database for testing."""
    with get_db_session as session:
        users = session.query(User).all()

    return users  # Return the user objects for testing


@patch("workflows.workflow.get_db_session")
@patch("workflows.workflow.get_user_list")
@patch("workflows.msgraph.is_user_account_enabled")  # Mock the Azure Graph API call
@patch("workflows.db.helpers.get_datetime_now")  # Mock the datetime helper
def test_mark_deletion_candidates(
    mock_get_datetime_now,
    mock_is_user_account_enabled,
    mock_get_user_list,
    mock_get_db_session,
    get_db_session,
    mock_user_data,
):
    """Test the mark_deletion_candidates function."""
    # db session should use the test session. Use of __enter__ is necessary because
    # workflows.workflow.get_db_session has a `with get_db_session as ..:` statement.
    mock_get_db_session.return_value.__enter__.return_value = get_db_session

    # mock sone return values
    mock_get_user_list.return_value = mock_user_data

    # "side_effect" simulates that the particular user is disabled in Azure,
    # because is_user_account_enabled takes email as input and returns True/False.
    mock_is_user_account_enabled.side_effect = (
        lambda email: email != "user2@example.com"
    )  # Only 'user2@example.com' is disabled

    mock_datetime_now = get_datetime_now()
    mock_get_datetime_now.return_value = mock_datetime_now

    # call the workflow function
    mark_deletion_candidates()

    # check the database for changes
    with get_db_session as session:
        # query for the users who should be marked for deletion
        user = session.query(User).filter(User.email == "user2@example.com").first()
        assert (
            user.wf_deletion_candidate is True
        ), "User 2 should be marked for deletion."
        assert (
            user.wf_deletion_timestamp == mock_datetime_now
        ), "User 2 should have the correct deletion timestamp."

        # assert that other users are not marked for deletion
        for user in session.query(User).filter(User.email != "user2@example.com"):
            assert (
                user.wf_deletion_candidate is False
            ), f"{user.email} should not be marked for deletion."
