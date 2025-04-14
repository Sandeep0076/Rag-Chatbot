import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_mock_file_names() -> List[str]:
    """
    Get all file names from the tests/mock_files directory.

    Returns:
        List[str]: List of file names in the mock_files directory
    """
    mock_files_dir = Path(__file__).parent / "mock_files"
    if not mock_files_dir.exists():
        logger.error(f"Mock files directory not found: {mock_files_dir}")
        return []

    # Get all files in the directory (excluding directories and hidden files)
    return [
        f.name
        for f in mock_files_dir.iterdir()
        if f.is_file() and not f.name.startswith(".")
    ]


def find_file_ids() -> List[str]:
    """
    Find file IDs for all files in the tests/mock_files directory.
    Uses the /find-file-by-name endpoint to get the file IDs.

    Returns:
        List[str]: List of file IDs that were found
    """
    base_url = "http://localhost:8080"  # API is running on port 8080
    endpoint = f"{base_url}/find-file-by-name"

    mock_files = get_mock_file_names()
    logger.info(f"Found {len(mock_files)} mock files: {mock_files}")

    file_ids = []

    for filename in mock_files:
        try:
            # Call the endpoint to find the file ID
            response = requests.get(endpoint, params={"filename": filename})
            response.raise_for_status()

            data = response.json()
            if data.get("found") and data.get("file_id"):
                file_id = data["file_id"]
                logger.info(f"Found file ID for {filename}: {file_id}")
                file_ids.append(file_id)
            else:
                logger.info(f"No file ID found for {filename}")
        except Exception as e:
            logger.error(f"Error finding file ID for {filename}: {str(e)}")

    return file_ids


def delete_embeddings(file_ids: List[str]) -> Dict[str, Any]:
    """
    Delete embeddings for the given file IDs.
    Uses the /delete_all_embeddings endpoint with include_gcs=true.

    Args:
        file_ids (List[str]): List of file IDs to delete embeddings for

    Returns:
        Dict[str, Any]: Response from the API
    """
    if not file_ids:
        logger.warning("No file IDs provided for deletion")
        return {"message": "No file IDs provided for deletion"}

    base_url = "http://localhost:8080"  # API is running on port 8080
    endpoint = f"{base_url}/delete_all_embeddings"

    # Prepare the request payload
    payload = {"file_ids": file_ids, "include_gcs": True}

    try:
        # Call the endpoint to delete embeddings
        logger.info(
            f"Deleting embeddings for {len(file_ids)} file IDs with include_gcs=true"
        )
        response = requests.delete(endpoint, json=payload)
        response.raise_for_status()

        result = response.json()
        logger.info(f"Deletion result: {json.dumps(result, indent=2)}")
        return result
    except Exception as e:
        logger.error(f"Error deleting embeddings: {str(e)}")
        return {"error": str(e)}


def main():
    """
    Main function to run the script.
    """
    logger.info("Starting to find file IDs for mock files...")
    file_ids = find_file_ids()

    if file_ids:
        logger.info(f"Found {len(file_ids)} file IDs: {file_ids}")
        _ = delete_embeddings(file_ids)  # Use _ to indicate intentionally unused result
        logger.info("Deletion completed.")
    else:
        logger.info("No file IDs found for mock files")

    return file_ids


if __name__ == "__main__":
    main()
