"""Pytest configuration and fixtures."""
import logging

import pytest
from fastapi.testclient import TestClient

from rtl_rag_chatbot_api.app import app

from .test_utils import ResourceManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Create test client
client = TestClient(app)

# Create a single instance for the whole test session
_resource_manager = ResourceManager()


@pytest.fixture(scope="session")
def resource_manager():
    """Provide the ResourceManager instance."""
    return _resource_manager


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_resources(resource_manager):
    """Cleanup test resources after all tests have completed."""
    logger.info("=== Starting cleanup_test_resources fixture ===")

    # Clear at start
    logger.info("Clearing existing file IDs at start")
    resource_manager.clear_file_ids()

    logger.info("Yielding control to tests...")
    yield

    # After all tests, get IDs and call delete endpoint
    logger.info("=== All tests completed, starting cleanup ===")
    file_ids = resource_manager.get_file_ids()
    logger.info(f"Found {len(file_ids)} file IDs to clean up: {file_ids}")

    if file_ids:
        logger.info("Sending DELETE request to cleanup resources...")
        try:
            response = client.request(
                "DELETE", "/delete", json={"file_ids": file_ids, "include_gcs": True}
            )
            logger.info(f"Delete response status: {response.status_code}")
            logger.info(f"Delete response: {response.text}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    else:
        logger.info("No file IDs to clean up")

    logger.info("=== Cleanup completed ===")
