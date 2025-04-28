from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from workflows.db.tables import Base
from workflows.workflow import (
    delete_candidate_user_embeddings,
    get_user_fileids,
    get_users_deletion_candidates,
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


@patch("workflows.workflow.delete_embeddings")
@patch("workflows.workflow.file_present_in_gcp")
@patch("workflows.workflow.get_db_session")
def test_delete_candidate_user_embeddings(
    mock_get_db_session,
    mock_file_present_in_gcp,
    mock_delete_embeddings,
    test_db_session,
):
    """
    Tests whether the delete_candidate_user_embeddings calls the right functions.
    """
    # Mock the get_db_session to return the in-memory test session
    mock_get_db_session.return_value = test_db_session

    # Mock the return values
    mock_delete_embeddings.return_value = None
    mock_file_present_in_gcp.return_value = True

    # Call the function
    delete_candidate_user_embeddings()

    # Assertions
    assert mock_file_present_in_gcp.call_count == 2
    assert mock_delete_embeddings.call_count == 2


@patch("workflows.workflow.file_present_in_gcp")
@patch("workflows.workflow.get_db_session")
def test_deletion_candidates_fileids(
    mock_get_db_session, mock_file_present_in_gcp, test_db_session
):
    """
    Tests whether the file ids are correctly returned for the deletion candidates.
    The test_db_session fixture is used to mock the database session.
    """
    # Mock the get_db_session to return the in-memory test session
    mock_get_db_session.return_value = test_db_session

    # Mock the return values
    mock_file_present_in_gcp.return_value = True

    # Call the function
    candidates = get_users_deletion_candidates()
    file_ids = get_user_fileids(candidates)
    assert len(file_ids) == 2


@patch("workflows.workflow.delete_embeddings")
@patch("workflows.workflow.get_user_fileids")
@patch("workflows.workflow.file_present_in_gcp")
@patch("workflows.workflow.get_db_session")
def test_deletion_condidates_fileids__None(
    mock_get_db_session,
    mock_file_present_in_gcp,
    mock_get_user_fileids,
    mock_delete_embeddings,
    test_db_session,
):
    """"""
    # Mock the get_db_session to return the in-memory test session
    mock_get_db_session.return_value = test_db_session

    # Mock the return values
    mock_get_user_fileids.return_value = None

    # Call the function
    delete_candidate_user_embeddings()

    assert mock_delete_embeddings.call_count == 0


@patch("workflows.workflow.file_present_in_gcp")
@patch("workflows.workflow.delete_embeddings")
@patch("workflows.workflow.get_db_session")
def test_deletion_condidates_fileids_FileNotFound(
    mock_get_db_session,
    mock_delete_embeddings,
    mock_file_present_in_gcp,
    mock_log,
    test_db_session,
):
    """
    This test simulates the case where the file is not found in GCP
    by adding a side effect to the mock_file_present_in_gcp function.
    """
    # Mock the get_db_session to return the in-memory test session
    mock_get_db_session.return_value = test_db_session

    # Mock the return values
    # Side effect function: only return False for file id "123123123"
    def file_present_in_gcp_side_effect(bucket_prefix):
        if "767703e0-8195-4345-914b-81bbaa0588b7" in bucket_prefix:
            return False  # Simulate missing file for user with ID = 3
        return True  # Other files exist

    mock_file_present_in_gcp.side_effect = file_present_in_gcp_side_effect

    # Call the function
    delete_candidate_user_embeddings()

    # Assertions
    assert mock_file_present_in_gcp.call_count == 2  # Called for each file ID
    assert mock_delete_embeddings.call_count == 1  # Only deletes existing files

    # Ensure the warning was logged
    mock_log.warning.assert_called_with(
        "Embeddings not found for file id '767703e0-8195-4345-914b-81bbaa0588b7'"
    )
