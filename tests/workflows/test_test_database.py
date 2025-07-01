from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from workflows.db.tables import Base, Citation, Prompt, User

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
        assert user_count == 9, f"Expected 9 users, but found {user_count}"

        deletion_candidates = db.query(User).filter(User.wf_deletion_candidate).count()
        assert deletion_candidates == 5


def test_published_prompts(test_db_session):
    """Test to ensure that there are 4 published prompts in the Prompt table."""

    with test_db_session as db:
        prompt_count = (
            db.query(Prompt).filter(Prompt.published).count()
        )  # Query the number of shared prompts in the Prompt table
        assert prompt_count == 4, f"Expected 4 prompts, but found {prompt_count}"


def test_citations(test_db_session):
    """Test to ensure that there are 4 citations in the Citation table."""

    with test_db_session as db:
        citation_count = db.query(
            Citation
        ).count()  # Query the number of citations in the Citation table
        assert citation_count == 4, f"Expected 4 citations, but found {citation_count}"
