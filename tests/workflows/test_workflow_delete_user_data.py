from unittest.mock import Mock, call, patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from workflows.db.tables import Base, Conversation, Folder, Message, User
from workflows.workflow import delete_candidate_user_data, get_users_deletion_candicates

# Create a session for a SQLite in-memory database
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def test_db_session():
    """
    Loads the example sqlite database which contains 6 users. 3 were marked
    as candidates for deletion, i.e. user4, user5, user 6. 2 with timestamp
    older than 4 weeks.
    """
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


def test_user_count_deletion_candidates(test_db_session):
    """Test to ensure that there are 8 users in the User table."""

    with test_db_session as db:
        user_count = db.query(
            User
        ).count()  # Query the number of users in the User table
        assert user_count == 8, f"Expected 8 users, but found {user_count}"

        deletion_candidates = db.query(User).filter(User.wf_deletion_candidate).count()
        assert deletion_candidates == 5


@patch("workflows.workflow.get_db_session")
def test_all_deletion_candidates_gone(mock_get_db_session, mock_log, test_db_session):
    """
    This method checks whether all user data has been sucessfully deleted.
    """
    # Mock the get_db_session to return the in-memory test session
    mock_get_db_session.return_value = test_db_session

    # call the function under test
    delete_candidate_user_data()

    # check the database for changes
    with test_db_session as db:
        # 2 deleted, 3 left
        delete_candidate_users = db.query(User).filter(User.wf_deletion_candidate).all()
        assert len(delete_candidate_users) == 3


@patch("workflows.workflow.get_db_session")
def test_deletion_successful_logging(mock_get_db_session, mock_log, test_db_session):
    """"""
    # Mock the get_db_session to return the in-memory test session
    mock_get_db_session.return_value = test_db_session

    # call the function under test
    delete_candidate_user_data()

    expected_calls = [
        # log reports about how many users to delete
        call(
            "Found 5 deletion candidates, 2 of them marked older than 4 weeks: user5@example.com, user6@example.com"
        ),
        # log reports about user5
        call("Loading user data for user5@example.com"),
        call("About to delete: 3 messages, 2 conversations, 2 folders."),
        call("Successfully deleted data for user user5@example.com."),
        # log reports about user6
        call("Loading user data for user6@example.com"),
        call("About to delete: 0 messages, 0 conversations, 0 folders."),
        call("Successfully deleted data for user user6@example.com."),
    ]

    assert mock_log.info.call_args_list == expected_calls


@patch("workflows.workflow.get_db_session")
def test_all_deletion_candidates_data_gone(
    mock_get_db_session, mock_log, test_db_session
):
    """"""
    # Mock the get_db_session to return the in-memory test session
    mock_get_db_session.return_value = test_db_session

    # call the function under test
    delete_candidate_user_data()

    test_data = [{"email": f"user{i}@example.com", "id": f"user{i}_id"} for i in [5, 6]]

    # check the database for changes
    with test_db_session as db:
        for user in test_data:
            messages = (
                db.query(Message)
                .join(Conversation)
                .filter(Conversation.userEmail == user["email"])
                .all()
            )
            # 3. get the list of Conversations related to the user
            conversations = (
                db.query(Conversation)
                .filter(Conversation.userEmail == user["email"])
                .all()
            )
            # 4. get the list of Folders related to the user
            folders = db.query(Folder).filter(Folder.userId == user["id"]).all()

            assert len(messages) == 0
            assert len(conversations) == 0
            assert len(folders) == 0


@patch("workflows.workflow.get_db_session")
def test_no_else_data_deleted(mock_get_db_session, mock_log, test_db_session):
    """
    Example sqlite database contains 5 users, where 2 users are marked
    as candidates for deletion, i.e. user4@example.com and user5@example.com.
    """
    # Mock the get_db_session to return the in-memory test session
    mock_get_db_session.return_value = test_db_session

    # call the function under test
    delete_candidate_user_data()

    # check the database for changes
    with test_db_session as db:
        # conversations left
        conversations = db.query(Conversation).all()
        assert len(conversations) == 10

        # messages left
        messages = db.query(Message).all()
        assert len(messages) == 12

        # folders left
        folders = db.query(Folder).all()
        assert len(folders) == 6

        # users left
        users = db.query(User).all()
        assert len(users) == 6


# Helper function to raise the exception
def raise_exception():
    raise Exception("Delete error")


@patch("workflows.workflow.get_db_session")
def test_delete_candidate_user_data_failure(
    mock_get_db_session, mock_log, test_db_session
):
    """"""
    # mock the get_db_session to return the in-memory test session
    mock_get_db_session.return_value.__enter__.return_value = test_db_session

    # simulate an exception when trying to delete data for the second user
    test_db_session.rollback = Mock()
    test_db_session.delete = Mock()
    test_db_session.delete.side_effect = (
        lambda user: None if user.email != "user6@example.com" else raise_exception()
    )

    # call the function under test
    delete_candidate_user_data()

    # ensure function calls on exception
    test_db_session.delete.call_count = 2
    test_db_session.rollback.call_count = 1

    # ensure the first user was deleted successfully and the second caused a failure
    mock_log.info.assert_any_call("Loading user data for user5@example.com")
    mock_log.info.assert_any_call("Loading user data for user6@example.com")
    mock_log.error.assert_any_call(
        "Failed to delete data for user user6@example.com: Delete error"
    )

    # check if users are still there
    remaining_users = get_users_deletion_candicates()
    # 1 successfully deleted, 2 left (see `test_all_deletion_candidates_gone`)
    assert len(remaining_users) == 2
