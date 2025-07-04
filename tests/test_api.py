import os
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from rtl_rag_chatbot_api.app import SessionLocal, app, get_current_user

# --- Pytest Configuration ---

# Mock the database session for all tests
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


# Mock authentication for all tests
async def override_get_current_user():
    return {"id": "testuser"}


# Apply the overrides for the app instance
app.dependency_overrides[SessionLocal] = override_get_db
app.dependency_overrides[get_current_user] = override_get_current_user

# Constants for mock file paths
MOCK_FILE_DIR = os.path.join(os.path.dirname(__file__), "mock_files")
MOCK_FILE_1 = os.path.join(MOCK_FILE_DIR, "mock_file1.pdf")
MOCK_FILE_2 = os.path.join(MOCK_FILE_DIR, "mock_file2.pdf")
MOCK_CSV_FILE = os.path.join(MOCK_FILE_DIR, "mock_file.csv")
MOCK_TXT_FILE = os.path.join(MOCK_FILE_DIR, "mock_txt.txt")


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient instance for the entire test module."""
    with TestClient(app) as c:
        yield c


# --- Stateful End-to-End Test Pipeline ---


class TestEndToEndPipeline:
    """
    Runs a series of stateful integration tests in a specific order.
    State (like file_id and session_id) is passed from one test to the next.
    """

    shared_data = {}  # Class-level dictionary to share state

    def wait_for_file_processing(self, client, file_id):
        """Helper function to poll the status endpoint."""
        print(f"\nPolling for embedding status of file_id: {file_id}...")
        start_time = time.time()
        while time.time() - start_time < 120:  # 2-minute timeout
            status_response = client.get(f"/embeddings/status/{file_id}")
            assert status_response.status_code == 200
            if status_response.json().get("can_chat"):
                print(f"Success! file_id: {file_id} is ready for chat.")
                return
            time.sleep(5)
        pytest.fail(f"Timeout: Waited 120s for file_id: {file_id}")

    def test_1_single_file_upload(self, client):
        """Uploads a single file and saves its ID and session ID."""
        print("\n--- Running Test: 1. Single File Upload ---")
        with open(MOCK_FILE_1, "rb") as f:
            files = {"file": ("mock_file1.pdf", f, "application/pdf")}
            response = client.post(
                "/file/upload", data={"username": "testuser"}, files=files
            )

        assert response.status_code == 200
        upload_json = response.json()
        assert upload_json["status"] == "success"

        file_id = upload_json.get("file_id")
        session_id = upload_json.get("session_id")
        assert file_id and session_id

        self.wait_for_file_processing(client, file_id)

        TestEndToEndPipeline.shared_data["single_file_id"] = file_id
        TestEndToEndPipeline.shared_data["single_session_id"] = session_id
        print(f"Saved single_file_id: {file_id}, single_session_id: {session_id}")

    def test_2_check_embeddings(self, client):
        """Checks the embedding status of the file from the previous test."""
        print("\n--- Running Test: 2. Check Embeddings ---")
        file_id = self.shared_data.get("single_file_id")
        assert file_id is not None

        check_data = {"file_id": file_id, "model_choice": "gpt_4o_mini"}
        response = client.post("/embeddings/check", json=check_data)

        assert response.status_code == 200
        response_json = response.json()
        assert response_json.get("embeddings_exist") is True
        assert response_json.get("file_id") == file_id
        print(f"Successfully confirmed embeddings exist for file_id: {file_id}")

    def test_3_single_file_chat(self, client):
        """Chats with the single file uploaded in the previous test."""
        print("\n--- Running Test: 3. Single File Chat ---")
        file_id = self.shared_data.get("single_file_id")
        session_id = self.shared_data.get("single_session_id")
        assert file_id and session_id

        chat_data = {
            "text": ["From which country this paper is from ?"],
            "file_id": file_id,
            "session_id": session_id,
            "model_choice": "gpt_4o_mini",
            "user_id": "testuser",
        }
        response = client.post("/file/chat", json=chat_data)

        assert response.status_code == 200
        response_json = response.json()
        assert "response" in response_json
        assert "Uzbekistan" in response_json["response"]
        print("Received single-file chat response and found 'Uzbekistan'")

    def test_4_multi_file_upload(self, client):
        """Uploads a new file and passes the ID of the previously uploaded file."""
        print("\n--- Running Test: 4. Multi File Upload (New + Existing) ---")
        existing_file_id = self.shared_data.get("single_file_id")
        assert existing_file_id

        with open(MOCK_FILE_2, "rb") as f:
            files = {"files": ("mock_file2.pdf", f, "application/pdf")}
            data = {"username": "testuser", "existing_file_ids": existing_file_id}
            response = client.post("/file/upload", data=data, files=files)

        assert response.status_code == 200
        upload_json = response.json()
        assert upload_json["status"] == "success"

        file_ids = upload_json.get("file_ids")
        session_id = upload_json.get("session_id")
        assert file_ids and len(file_ids) == 2 and session_id

        # Wait for the *new* file to be processed
        new_file_id = next(fid for fid in file_ids if fid != existing_file_id)
        self.wait_for_file_processing(client, new_file_id)

        TestEndToEndPipeline.shared_data["multi_file_ids"] = file_ids
        TestEndToEndPipeline.shared_data["multi_session_id"] = session_id
        print(f"Saved multi_file_ids: {file_ids}, multi_session_id: {session_id}")

    def test_5_multi_file_chat(self, client):
        """Chats with the two files from the previous test."""
        print("\n--- Running Test: 5. Multi File Chat ---")
        file_ids = self.shared_data.get("multi_file_ids")
        session_id = self.shared_data.get("multi_session_id")
        assert file_ids and session_id

        chat_data = {
            "text": [
                "What is the country in the first document and who is the author of the second?"
            ],
            "file_ids": file_ids,
            "session_id": session_id,
            "model_choice": "gpt_4o_mini",
            "user_id": "testuser",
        }
        response = client.post("/file/chat", json=chat_data)

        assert response.status_code == 200
        response_json = response.json()
        response_text = response_json.get("response", "")
        assert "Uzbekistan" in response_text and "Andrew Lang" in response_text
        print(
            "Received multi-file chat response and found 'Uzbekistan' and 'Andrew Lang'"
        )


def test_chat_with_csv(client):
    """
    Tests the full pipeline for uploading and chatting with a single CSV file.
    """
    print("\n--- Running Test: Chat with CSV ---")
    with open(MOCK_CSV_FILE, "rb") as f:
        files = {"file": ("mock_file.csv", f, "text/csv")}
        upload_response = client.post(
            "/file/upload", data={"username": "testuser"}, files=files
        )

    assert upload_response.status_code == 200
    upload_json = upload_response.json()
    file_id = upload_json.get("file_id")
    session_id = upload_json.get("session_id")
    assert file_id and session_id
    print(f"Uploaded CSV file. Received file_id: {file_id}")

    # Poll the status endpoint to ensure the tabular data is processed
    print(f"Polling for status of file_id: {file_id}...")
    start_time = time.time()
    while time.time() - start_time < 60:  # 1-minute timeout for CSV processing
        status_response = client.get(f"/embeddings/status/{file_id}")
        assert status_response.status_code == 200
        if status_response.json().get("can_chat"):
            print(f"Success! CSV file {file_id} is ready for chat.")
            break
        time.sleep(3)
    else:
        pytest.fail(f"Timeout waiting for CSV file {file_id} to process.")

    # Chat with the processed CSV file
    chat_data = {
        "text": ["Show me the average of pregnancies"],
        "file_id": file_id,
        "session_id": session_id,
        "model_choice": "gpt_4o_mini",
        "user_id": "testuser",
    }
    chat_response = client.post("/file/chat", json=chat_data)

    assert chat_response.status_code == 200
    response_json = chat_response.json()
    response_text = response_json.get("response", "")
    print(f"Received chat response for CSV: '{response_text}'")

    # Assert that the correct numerical answer is in the response
    assert "3.8" in response_text


def test_chat_with_doc_gemini(client):
    """
    Tests the full pipeline for uploading and chatting with a single .txt
    file using a Gemini model.
    """
    print("\n--- Running Test: Chat with Doc (Gemini) ---")
    with open(MOCK_TXT_FILE, "rb") as f:
        files = {"file": ("mock_txt.txt", f, "text/plain")}
        upload_response = client.post(
            "/file/upload", data={"username": "testuser"}, files=files
        )

    assert upload_response.status_code == 200
    upload_json = upload_response.json()
    file_id = upload_json.get("file_id")
    session_id = upload_json.get("session_id")
    assert file_id and session_id
    print(f"Uploaded document. Received file_id: {file_id}")

    # Helper function from the class to wait for processing
    # Note: We need a way to call wait_for_file_processing.
    # For simplicity here, we re-implement the polling loop.
    print(f"Polling for status of file_id: {file_id}...")
    start_time = time.time()
    while time.time() - start_time < 60:
        status_response = client.get(f"/embeddings/status/{file_id}")
        assert status_response.status_code == 200
        if status_response.json().get("can_chat"):
            print(f"Success! Document {file_id} is ready for chat.")
            break
        time.sleep(3)
    else:
        pytest.fail(f"Timeout waiting for document {file_id} to process.")

    # Chat with the processed document using Gemini
    chat_data = {
        "text": ["How many mangoes are there in Garden"],
        "file_id": file_id,
        "session_id": session_id,
        "model_choice": "gemini-2.5-pro",
        "user_id": "testuser",
    }
    chat_response = client.post("/file/chat", json=chat_data)

    assert chat_response.status_code == 200
    response_json = chat_response.json()
    response_text = response_json.get("response", "").lower()
    print(f"Received chat response for document: '{response_text}'")

    # Assert that the correct answer is in the response (case-insensitive)
    assert any(val in response_text for val in ["50", "fifty"])
