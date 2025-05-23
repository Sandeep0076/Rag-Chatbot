import os
from unittest.mock import patch

import pytest
from sqlalchemy import and_, create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from workflows.db.tables import Base, Prompt
from workflows.workflow import anonymise_shared_prompts

DATABASE_URL = "sqlite:///:memory:"
os.getenv("DATABASE_URL", DATABASE_URL)

# Create a session for a SQLite in-memory database
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(autouse=True)
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


@patch("workflows.workflow.get_db_session")
def test_anonymise_prompts(mock_get_db_session, mock_log, test_db_session):
    """
    Test the anonymisation of prompts in the database.
    """
    # db session should use the test session. Use of __enter__ is necessary because
    # workflows.workflow.get_db_session has a `with get_db_session as ..:` statement.
    mock_get_db_session.return_value.__enter__.return_value = test_db_session

    # check that the prompts are not anonymised before method call
    with test_db_session as db:
        prompts = (
            db.query(Prompt)
            .filter(
                and_(
                    Prompt.published,
                    Prompt.userId == os.getenv("WF_DB_DELETED_USER_ID"),
                )
            )
            .all()
        )

        assert len(prompts) == 0, f"Expected 0 prompts, but found {len(prompts)}"

    # call the workflow function
    anonymise_shared_prompts()

    # check that the prompts have been anonymised
    with test_db_session as db:
        prompts = (
            db.query(Prompt)
            .filter(
                and_(
                    Prompt.published,
                    Prompt.userId == os.getenv("WF_DB_DELETED_USER_ID"),
                )
            )
            .all()
        )

        assert len(prompts) == 2, f"Expected 2 prompts, but found {len(prompts)}"
