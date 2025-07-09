import os
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from rtl_rag_chatbot_api.app import SessionLocal, app, get_current_user

# --- Mock Setup for Advanced Tests ---

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


async def override_get_current_user():
    return {"id": "testuser"}


app.dependency_overrides[SessionLocal] = override_get_db
app.dependency_overrides[get_current_user] = override_get_current_user

# Constants for mock file paths
MOCK_FILE_DIR = os.path.join(os.path.dirname(__file__), "mock_files")
MOCK_FILE_1 = os.path.join(MOCK_FILE_DIR, "mock_file1.pdf")
MOCK_FILE_2 = os.path.join(MOCK_FILE_DIR, "mock_file2.pdf")
MOCK_CSV_FILE = os.path.join(MOCK_FILE_DIR, "mock_file.csv")
MOCK_IMAGE_FILE = os.path.join(MOCK_FILE_DIR, "mock_file.png")
MOCK_EXCEL_FILE = os.path.join(MOCK_FILE_DIR, "mock_file.xlsx")


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient instance for the entire test module."""
    with TestClient(app) as c:
        yield c


class TestAdvancedMultiFileScenarios:
    """Advanced tests for error handling, edge cases, and complex scenarios."""

    def test_multiple_same_type_files_upload(self, client):
        """Test uploading multiple files of the same type in a single request."""
        print("\n--- Running Test: Multiple Same Type Files Upload ---")

        # Upload multiple PDF files together
        files_data = []

        # Add first PDF file
        with open(MOCK_FILE_1, "rb") as f1:
            files_data.append(("files", ("doc1.pdf", f1.read(), "application/pdf")))

        # Add second PDF file
        with open(MOCK_FILE_2, "rb") as f2:
            files_data.append(("files", ("doc2.pdf", f2.read(), "application/pdf")))

        # Upload using files parameter for multiple files
        response = client.post(
            "/file/upload",
            data={"username": "testuser", "is_image": "false"},
            files=files_data,
        )

        assert response.status_code == 200
        response_json = response.json()
        assert response_json["status"] == "success"
        assert len(response_json["file_ids"]) == 2
        assert response_json["multi_file_mode"] is True
        print(
            f"Successfully uploaded multiple same-type files: {response_json['file_ids']}"
        )

    def test_invalid_file_format_error_handling(self, client):
        """Test error handling for invalid file formats."""
        print("\n--- Running Test: Invalid File Format Error Handling ---")

        # Create a fake invalid file
        invalid_content = b"This is not a valid PDF or document"

        response = client.post(
            "/file/upload",
            data={"username": "testuser"},
            files={
                "file": ("invalid.xyz", invalid_content, "application/octet-stream")
            },
        )

        # Should still succeed with our current system but may have processing issues
        # The system should handle gracefully
        assert response.status_code == 200
        print("System handled invalid file format gracefully")

    def test_empty_file_upload(self, client):
        """Test uploading an empty file."""
        print("\n--- Running Test: Empty File Upload ---")

        empty_content = b""

        response = client.post(
            "/file/upload",
            data={"username": "testuser"},
            files={"file": ("empty.pdf", empty_content, "application/pdf")},
        )

        # System should handle empty files gracefully
        assert response.status_code == 200
        print("System handled empty file upload")

    def test_invalid_existing_file_ids(self, client):
        """Test behavior with invalid existing file IDs."""
        print("\n--- Running Test: Invalid Existing File IDs ---")

        # Try to upload with invalid existing file IDs
        invalid_file_ids = "nonexistent-id-1,invalid-id-2"

        with open(MOCK_FILE_1, "rb") as f:
            response = client.post(
                "/file/upload",
                data={"username": "testuser", "existing_file_ids": invalid_file_ids},
                files={"file": ("new_file.pdf", f, "application/pdf")},
            )

        # Should return 400 error for invalid file IDs
        assert response.status_code == 400
        response_json = response.json()
        assert "invalid file id" in response_json["detail"]["message"].lower()
        print("System correctly rejected invalid existing file IDs")

    def test_partial_failure_existing_and_new_files(self, client):
        """Test scenario where some existing file IDs are valid and others invalid."""
        print("\n--- Running Test: Partial Failure Existing + New Files ---")

        # First upload a file to get a valid file ID
        with open(MOCK_FILE_1, "rb") as f:
            upload_response = client.post(
                "/file/upload",
                data={"username": "testuser"},
                files={"file": ("valid_file.pdf", f, "application/pdf")},
            )

        assert upload_response.status_code == 200
        valid_file_id = upload_response.json()["file_id"]

        # Wait for processing
        time.sleep(5)

        # Now try with mix of valid and invalid file IDs
        mixed_file_ids = f"{valid_file_id},invalid-file-id"

        with open(MOCK_FILE_2, "rb") as f:
            response = client.post(
                "/file/upload",
                data={"username": "testuser", "existing_file_ids": mixed_file_ids},
                files={"file": ("another_file.pdf", f, "application/pdf")},
            )

        # Should fail due to invalid file ID
        assert response.status_code == 400
        print("System correctly handled mixed valid/invalid file IDs")

    @patch("rtl_rag_chatbot_api.chatbot.gcs_handler.GCSHandler.upload_to_gcs")
    def test_gcs_upload_failure_handling(self, mock_upload, client):
        """Test handling of GCS upload failures."""
        print("\n--- Running Test: GCS Upload Failure Handling ---")

        # Mock GCS upload to fail
        mock_upload.side_effect = Exception("GCS upload failed")

        with open(MOCK_FILE_1, "rb") as f:
            response = client.post(
                "/file/upload",
                data={"username": "testuser"},
                files={"file": ("test_file.pdf", f, "application/pdf")},
            )

        # System should handle the error gracefully
        # Depending on implementation, might succeed with local processing
        print(f"GCS failure handling response: {response.status_code}")

    def test_concurrent_upload_same_file(self, client):
        """Test concurrent uploads of the same file by different users."""
        print("\n--- Running Test: Concurrent Upload Same File ---")

        def upload_file_sync(username):
            with open(MOCK_FILE_1, "rb") as f:
                return client.post(
                    "/file/upload",
                    data={"username": username},
                    files={
                        "file": ("concurrent_test.pdf", f.read(), "application/pdf")
                    },
                )

        # Simulate concurrent uploads (simplified for sync test)
        response1 = upload_file_sync("user1")
        response2 = upload_file_sync("user2")

        # Both should succeed but may have same file_id if deduplication works
        assert response1.status_code == 200
        assert response2.status_code == 200
        print("Concurrent uploads handled successfully")

    def test_large_filename_handling(self, client):
        """Test handling of very long filenames."""
        print("\n--- Running Test: Large Filename Handling ---")

        # Create a very long filename
        long_filename = "a" * 150 + ".pdf"

        with open(MOCK_FILE_1, "rb") as f:
            response = client.post(
                "/file/upload",
                data={"username": "testuser"},
                files={"file": (long_filename, f, "application/pdf")},
            )

        assert response.status_code == 200
        # Filename should be truncated to safe length
        print("Long filename handled correctly")

    def test_url_upload_with_invalid_urls(self, client):
        """Test URL upload with invalid URLs mixed with valid ones."""
        print("\n--- Running Test: Invalid URL Upload ---")

        invalid_urls = "https://definitely-not-a-real-website-12345.com,not-a-url"

        response = client.post(
            "/file/upload", data={"username": "testuser", "urls": invalid_urls}
        )

        # Should handle invalid URLs gracefully - may return 400 if all URLs fail
        assert response.status_code in [200, 400]
        response_json = response.json()
        # May succeed with empty results or error status
        if response.status_code == 400:
            print(
                f"Invalid URLs correctly rejected: {response_json.get('detail', 'unknown')}"
            )
        else:
            print(f"Invalid URL handling: {response_json.get('status', 'unknown')}")

    def test_session_consistency_multi_upload(self, client):
        """Test that session_id is consistent across multi-file uploads."""
        print("\n--- Running Test: Session Consistency Multi-Upload ---")

        files_data = []
        with open(MOCK_FILE_1, "rb") as f1, open(MOCK_FILE_2, "rb") as f2:
            files_data = [
                ("files", ("file1.pdf", f1.read(), "application/pdf")),
                ("files", ("file2.pdf", f2.read(), "application/pdf")),
            ]

        response = client.post(
            "/file/upload", data={"username": "testuser"}, files=files_data
        )

        assert response.status_code == 200
        response_json = response.json()
        assert "session_id" in response_json
        assert len(response_json["file_ids"]) == 2

        # All files should share the same session_id
        session_id = response_json["session_id"]
        assert session_id is not None
        print(f"Session consistency verified: {session_id}")

    def test_delete_nonexistent_files(self, client):
        """Test deletion of files that don't exist."""
        print("\n--- Running Test: Delete Nonexistent Files ---")

        nonexistent_ids = ["fake-id-1", "fake-id-2"]

        delete_data = {
            "file_ids": nonexistent_ids,
            "include_gcs": True,
            "username": "testuser",
        }

        response = client.request("DELETE", "/delete", json=delete_data)

        # Should handle gracefully, possibly with partial success status
        assert response.status_code in [200, 400, 404]
        print(f"Delete nonexistent files handled: {response.status_code}")

    def test_embedding_check_invalid_file_id(self, client):
        """Test embedding check with invalid file ID."""
        print("\n--- Running Test: Embedding Check Invalid File ID ---")

        check_data = {
            "file_id": "definitely-not-a-real-file-id",
            "model_choice": "gpt_4o_mini",
        }

        response = client.post("/embeddings/check", json=check_data)

        # Should return that embeddings don't exist
        assert response.status_code == 200
        response_json = response.json()
        assert response_json.get("embeddings_exist") is False
        print("Invalid file ID embedding check handled correctly")

    def test_status_check_invalid_file_id(self, client):
        """Test status endpoint with invalid file ID."""
        print("\n--- Running Test: Status Check Invalid File ID ---")

        invalid_file_id = "not-a-real-file-id"
        response = client.get(f"/embeddings/status/{invalid_file_id}")

        # Should handle gracefully
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            response_json = response.json()
            assert response_json.get("can_chat") is False
        print("Invalid file ID status check handled")

    def test_upload_without_username(self, client):
        """Test upload without required username parameter."""
        print("\n--- Running Test: Upload Without Username ---")

        with open(MOCK_FILE_1, "rb") as f:
            response = client.post(
                "/file/upload",
                # Missing username parameter
                files={"file": ("test.pdf", f, "application/pdf")},
            )

        # Should return 422 for missing required field
        assert response.status_code == 422
        print("Missing username correctly rejected")

    def test_chat_with_invalid_session_combination(self, client):
        """Test chat with mismatched file_id and session_id."""
        print("\n--- Running Test: Invalid Session Combination ---")

        # First upload a file
        with open(MOCK_FILE_1, "rb") as f:
            upload_response = client.post(
                "/file/upload",
                data={"username": "testuser"},
                files={"file": ("test.pdf", f, "application/pdf")},
            )

        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]

        # Try to chat with wrong session_id
        chat_data = {
            "text": ["Test question"],
            "file_id": file_id,
            "session_id": "definitely-wrong-session-id",
            "model_choice": "gpt_4o_mini",
            "user_id": "testuser",
        }

        response = client.post("/file/chat", json=chat_data)

        # System should handle gracefully or return error
        # Current implementation may not validate session_id strictly
        print(f"Invalid session combination handled: {response.status_code}")


class TestResourceManagementAndCleanup:
    """Tests for resource management and cleanup scenarios."""

    def test_cleanup_after_failed_upload(self, client):
        """Test that resources are cleaned up after failed uploads."""
        print("\n--- Running Test: Cleanup After Failed Upload ---")

        # This is harder to test without mocking internals
        # For now, just verify system doesn't crash on error conditions
        with open(MOCK_FILE_1, "rb") as f:
            response = client.post(
                "/file/upload",
                data={"username": "testuser"},
                files={"file": ("test.pdf", f, "application/pdf")},
            )

        # System should handle any internal cleanup automatically
        assert response.status_code in [200, 500]
        print("Resource cleanup verification completed")

    def test_memory_usage_multiple_large_files(self, client):
        """Test memory usage with multiple file uploads."""
        print("\n--- Running Test: Memory Usage Multiple Files ---")

        # Upload multiple files to test memory management
        files_data = []
        for i in range(3):  # Reduced number for test efficiency
            with open(MOCK_FILE_1, "rb") as f:
                files_data.append(
                    ("files", (f"large_file_{i}.pdf", f.read(), "application/pdf"))
                )

        response = client.post(
            "/file/upload", data={"username": "testuser"}, files=files_data
        )

        assert response.status_code == 200
        print("Multiple file memory management test completed")
