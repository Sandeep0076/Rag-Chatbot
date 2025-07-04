# write a test for the delete_private_prompts function
from unittest.mock import patch

import pytest
from sqlalchemy import and_, create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from workflows.db.tables import Base, Prompt, PromptFolder, PromptLike
from workflows.workflow import delete_private_prompts

# Create a session for a SQLite in-memory database
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def mock_get_db_session():
    with patch("workflows.workflow.get_db_session") as mock:
        yield mock


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


@patch("workflows.workflow.get_db_session")
def test_delete_private_prompts(mock_get_db_session, test_db_session):
    """
    Test the delete_private_prompts function to ensure it deletes private prompts,
    their likes, and associated folders.
    """
    # Mock the get_db_session to return the in-memory test session
    mock_get_db_session.return_value = test_db_session

    delete_private_prompts()

    expected_prompts = {5: 2, 6: 1}
    expected_prompt_likes = {5: 2, 6: 0}
    test_data = [
        {
            "email": f"user{i}@example.com",
            "id": f"user{i}_id",
            "expected_prompts": expected_prompts[i],
            "expected_prompt_likes": expected_prompt_likes[i],
        }
        for i in [5, 6]
    ]

    # check the database for changes
    with test_db_session as db:
        for user in test_data:
            # Check if the user has no private Prompts left
            prompts = (
                db.query(Prompt)
                # only published prompts are left
                .filter(and_(Prompt.userId == user["id"], Prompt.published.is_(True)))
                .distinct()
                .all()
            )
            assert (
                len(prompts) == user["expected_prompts"]
            ), f"User {user['email']} should have {user['expected_prompts']} shared prompts left."

            # Check if the user has PromptLikes left.
            # At this point and in this test, there are no anonymised prompts yet.
            prompt_likes = (
                db.query(PromptLike)
                .join(Prompt)
                .filter(and_(Prompt.userId == user["id"]))
                .all()
            )
            assert (
                len(prompt_likes) == user["expected_prompt_likes"]
            ), f"User {user['email']} should have {user['expected_prompt_likes']} prompt likes left."


def test_delete_private_prompts_folders_left(mock_get_db_session, test_db_session):
    """
    Test that the delete_private_prompts function does not delete folders
    that contain shared prompts.
    """
    # Mock the get_db_session to return the in-memory test session
    mock_get_db_session.return_value = test_db_session

    delete_private_prompts()

    expected_prompt_folders = {5: 1, 6: 1}
    test_data = [
        {
            "email": f"user{i}@example.com",
            "id": f"user{i}_id",
            "expected_prompt_folders": expected_prompt_folders[i],
        }
        for i in [5, 6]
    ]

    with test_db_session as db:
        for user in test_data:
            # Check PromptFolders remain with published prompts
            prompt_folders = (
                db.query(PromptFolder)
                .filter(and_(PromptFolder.userId == user["id"]))
                .distinct()
                .all()
            )
            assert (
                len(prompt_folders) == user["expected_prompt_folders"]
            ), f"User {user['email']} should have {user['expected_prompt_folders']} prompt folders left."

            # Check other folders remain
            prompt_folders_count = db.query(PromptFolder).count()
            assert (
                prompt_folders_count == 4
            ), f"Expected 4 prompt folders, but found {prompt_folders_count}."
