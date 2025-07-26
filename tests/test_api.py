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
MOCK_IMAGE_FILE_1 = os.path.join(MOCK_FILE_DIR, "mock_file1.png")
MOCK_IMAGE_FILE_2 = os.path.join(MOCK_FILE_DIR, "mock_file2.jpg")
MOCK_DB_FILE = os.path.join(MOCK_FILE_DIR, "mock_file.sqlite")
MOCK_EXCEL_FILE = os.path.join(MOCK_FILE_DIR, "mock_file.xlsx")


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
        while time.time() - start_time < 180:  # 3-minute timeout
            status_response = client.get(f"/embeddings/status/{file_id}")
            assert status_response.status_code == 200
            if status_response.json().get("can_chat"):
                print(f"Success! file_id: {file_id} is ready for chat.")
                return
            time.sleep(5)
        pytest.fail(f"Timeout: Waited 180s for file_id: {file_id}")

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

        check_data = {"file_ids": [file_id], "model_choice": "gpt_4o_mini"}
        response = client.post("/embeddings/check", json=check_data)

        assert response.status_code == 200
        response_json = response.json()
        assert "results" in response_json
        assert len(response_json["results"]) == 1
        result = response_json["results"][0]
        assert result.get("embeddings_exist") is True
        assert result.get("file_id") == file_id
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
                "In mock_file1.pdf, From which country this paper is from, and in mock_file2.pdf, who is the author?"
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

    def test_6_delete_single_file(self, client):
        """Deletes the single file uploaded and chatted with in previous tests."""
        print("\n--- Running Test: 6. Delete Single File ---")
        file_id = self.shared_data.get("single_file_id")
        assert file_id

        delete_data = {"file_ids": file_id, "include_gcs": True, "username": "testuser"}
        response = client.request("DELETE", "/delete", json=delete_data)

        assert response.status_code == 200
        response_json = response.json()
        assert "message" in response_json
        # The delete endpoint may either delete embeddings or just remove username
        assert any(
            keyword in response_json["message"].lower()
            for keyword in ["deleted", "removed", "success"]
        ), f"Unexpected delete response: {response_json['message']}"
        print(
            f"Successfully processed delete for single file: {file_id} - {response_json['message']}"
        )

    def test_7_delete_multi_files(self, client):
        """Deletes the multiple files uploaded and chatted with in previous tests."""
        print("\n--- Running Test: 7. Delete Multi Files ---")
        file_ids = self.shared_data.get("multi_file_ids")
        single_file_id = self.shared_data.get("single_file_id")
        assert file_ids and len(file_ids) == 2

        # Filter out the single_file_id that was already deleted in test_6
        # Only delete the new file that was added in the multi-file upload
        remaining_file_ids = [fid for fid in file_ids if fid != single_file_id]

        if not remaining_file_ids:
            print("No remaining files to delete (all were already deleted)")
            return

        delete_data = {
            "file_ids": remaining_file_ids,
            "include_gcs": True,
            "username": "testuser",
        }
        response = client.request("DELETE", "/delete", json=delete_data)

        assert response.status_code == 200
        response_json = response.json()

        # Handle both single file response format and multi-file response format
        if "message" in response_json:
            # Single file deletion response format
            assert any(
                keyword in response_json["message"].lower()
                for keyword in ["deleted", "removed", "success"]
            ), f"Unexpected delete response: {response_json['message']}"
            print(
                f"Successfully processed delete for remaining file: "
                f"{remaining_file_ids[0]} - {response_json['message']}"
            )
        elif "results" in response_json:
            # Multi-file deletion response format
            results = response_json["results"]
            assert len(results) == len(remaining_file_ids)

            # Check that all remaining files were successfully processed (deleted or username removed)
            for file_id in remaining_file_ids:
                assert file_id in results
                assert (
                    "Success" in results[file_id]
                ), f"Failed to process file {file_id}: {results[file_id]}"

            print(
                f"Successfully processed delete for remaining files: "
                f"{remaining_file_ids}"
            )
        else:
            pytest.fail(f"Unexpected response format: {response_json}")


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


def test_chat_with_csv_visualization(client):
    """
    Tests that a query asking for a chart from a CSV file
    returns a valid chart configuration.
    """
    print("\n--- Running Test: Chat with CSV (Visualization) ---")
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
    while time.time() - start_time < 60:
        status_response = client.get(f"/embeddings/status/{file_id}")
        assert status_response.status_code == 200
        if status_response.json().get("can_chat"):
            print(f"Success! CSV file {file_id} is ready for chat.")
            break
        time.sleep(3)
    else:
        pytest.fail(f"Timeout waiting for CSV file {file_id} to process.")

    # Chat with a query that should trigger a visualization
    chat_data = {
        "text": [
            "Generate a chart for Relationship between pregnancies and age for first 10 entries"
        ],
        "file_id": file_id,
        "session_id": session_id,
        "model_choice": "gpt_4o_mini",
        "user_id": "testuser",
    }
    chat_response = client.post("/file/chat", json=chat_data)

    assert chat_response.status_code == 200
    response_json = chat_response.json()
    print(f"Received visualization response: {response_json}")

    # Verify chart configuration
    assert "chart_config" in response_json
    chart_config = response_json["chart_config"]
    assert isinstance(chart_config, dict)
    assert "data" in chart_config
    assert "datasets" in chart_config["data"]


def test_chat_with_single_image(client):
    """
    Tests the full pipeline for uploading and chatting with a single image file.
    """
    print("\n--- Running Test: Chat with Single Image ---")
    with open(MOCK_IMAGE_FILE_1, "rb") as f:
        files = {"file": ("mock_file1.png", f, "image/png")}
        upload_response = client.post(
            "/file/upload",
            data={"username": "testuser", "is_image": "true"},
            files=files,
        )

    assert upload_response.status_code == 200
    upload_json = upload_response.json()
    file_id = upload_json.get("file_id")
    session_id = upload_json.get("session_id")
    assert file_id and session_id
    print(f"Uploaded image file. Received file_id: {file_id}")

    # Poll the status endpoint to ensure the image is processed
    print(f"Polling for status of file_id: {file_id}...")
    start_time = time.time()
    while time.time() - start_time < 60:  # 1-minute timeout for image processing
        status_response = client.get(f"/embeddings/status/{file_id}")
        assert status_response.status_code == 200
        if status_response.json().get("can_chat"):
            print(f"Success! Image file {file_id} is ready for chat.")
            break
        time.sleep(3)
    else:
        pytest.fail(f"Timeout waiting for image file {file_id} to process.")

    # Chat with the processed image file
    chat_data = {
        "text": ["Who has highest gdp per capita in the given image.?"],
        "file_id": file_id,
        "session_id": session_id,
        "model_choice": "gpt_4o_mini",
        "user_id": "testuser",
    }
    chat_response = client.post("/file/chat", json=chat_data)

    assert chat_response.status_code == 200
    response_json = chat_response.json()
    response_text = response_json.get("response", "")
    print(f"Received chat response for image: '{response_text}'")

    # Assert that the response contains reference to United States or similar
    assert any(
        country in response_text for country in ["United States", "US", "USA"]
    ), f"Expected US reference in response: {response_text}"


def test_chat_with_multiple_images(client):
    """
    Tests the full pipeline for uploading and chatting with multiple image files.
    """
    print("\n--- Running Test: Chat with Multiple Images ---")

    # First upload the first image
    with open(MOCK_IMAGE_FILE_1, "rb") as f:
        files = {"file": ("mock_file1.png", f, "image/png")}
        upload_response_1 = client.post(
            "/file/upload",
            data={"username": "testuser", "is_image": "true"},
            files=files,
        )

    assert upload_response_1.status_code == 200
    upload_json_1 = upload_response_1.json()
    file_id_1 = upload_json_1.get("file_id")
    assert file_id_1
    print(f"Uploaded first image file. Received file_id: {file_id_1}")

    # Wait for first image to be processed
    print(f"Polling for status of first image file_id: {file_id_1}...")
    start_time = time.time()
    while time.time() - start_time < 60:
        status_response = client.get(f"/embeddings/status/{file_id_1}")
        assert status_response.status_code == 200
        if status_response.json().get("can_chat"):
            print(f"Success! First image file {file_id_1} is ready for chat.")
            break
        time.sleep(3)
    else:
        pytest.fail(f"Timeout waiting for first image file {file_id_1} to process.")

    # Now upload the second image with the existing file ID
    with open(MOCK_IMAGE_FILE_2, "rb") as f:
        files = {"files": ("mock_file2.jpg", f, "image/jpeg")}
        data = {
            "username": "testuser",
            "is_image": "true",
            "existing_file_ids": file_id_1,
        }
        upload_response_2 = client.post("/file/upload", data=data, files=files)

    assert upload_response_2.status_code == 200
    upload_json_2 = upload_response_2.json()
    file_ids = upload_json_2.get("file_ids")
    session_id = upload_json_2.get("session_id")
    assert file_ids and len(file_ids) == 2 and session_id
    print(f"Uploaded second image file. Received file_ids: {file_ids}")

    # Wait for the new (second) image to be processed
    new_file_id = next(fid for fid in file_ids if fid != file_id_1)
    print(f"Polling for status of second image file_id: {new_file_id}...")
    start_time = time.time()
    while time.time() - start_time < 60:
        status_response = client.get(f"/embeddings/status/{new_file_id}")
        assert status_response.status_code == 200
        if status_response.json().get("can_chat"):
            print(f"Success! Second image file {new_file_id} is ready for chat.")
            break
        time.sleep(3)
    else:
        pytest.fail(f"Timeout waiting for second image file {new_file_id} to process.")

    # Chat with both images
    chat_data = {
        "text": [
            "In mock_file1.png, which country has the highest GDP per capita, "
            "and in mock_file2.jpg, which model has the highest Global Average value?"
        ],
        "file_ids": file_ids,
        "session_id": session_id,
        "model_choice": "gpt_4o_mini",
        "user_id": "testuser",
    }
    chat_response = client.post("/file/chat", json=chat_data)

    assert chat_response.status_code == 200
    response_json = chat_response.json()
    response_text = response_json.get("response", "")
    print(f"Received chat response for multiple images: '{response_text}'")

    # Assert that the response contains references to both expected answers
    # For first image: United States or similar for GDP
    # For second image: some model name for Global Average
    assert any(
        country in response_text for country in ["United States", "US", "USA"]
    ), f"Expected US reference for GDP question in response: {response_text}"

    # Check for any model-related terms that might indicate the answer to the second question
    assert any(
        term in response_text.lower() for term in ["o1 High", "o1"]
    ), f"Expected model/average related terms in response: {response_text}"


def test_chat_with_database(client):
    """
    Tests the full pipeline for uploading and chatting with a database file.
    """
    print("\n--- Running Test: Chat with Database ---")
    with open(MOCK_DB_FILE, "rb") as f:
        files = {"file": ("mock_file.sqlite", f, "application/x-sqlite3")}
        upload_response = client.post(
            "/file/upload",
            data={"username": "testuser", "is_image": "false"},
            files=files,
        )

    assert upload_response.status_code == 200
    upload_json = upload_response.json()
    file_id = upload_json.get("file_id")
    session_id = upload_json.get("session_id")
    assert file_id and session_id
    print(f"Uploaded database file. Received file_id: {file_id}")

    # Poll the status endpoint to ensure the database is processed
    print(f"Polling for status of file_id: {file_id}...")
    start_time = time.time()
    while time.time() - start_time < 60:  # 1-minute timeout for database processing
        status_response = client.get(f"/embeddings/status/{file_id}")
        assert status_response.status_code == 200
        if status_response.json().get("can_chat"):
            print(f"Success! Database file {file_id} is ready for chat.")
            break
        time.sleep(3)
    else:
        pytest.fail(f"Timeout waiting for database file {file_id} to process.")

    # Chat with the processed database file
    chat_data = {
        "text": ["What is the address of customer Maria Anders?"],
        "file_id": file_id,
        "session_id": session_id,
        "model_choice": "gpt_4o_mini",
        "user_id": "testuser",
    }
    chat_response = client.post("/file/chat", json=chat_data)

    assert chat_response.status_code == 200
    response_json = chat_response.json()
    response_text = response_json.get("response", "")
    print(f"Received chat response for database: '{response_text}'")

    # Assert that the response contains the expected address
    assert (
        "Obere Str. 57" in response_text
    ), f"Expected 'Obere Str. 57' in response: {response_text}"


def test_chat_with_excel(client):
    """
    Tests the full pipeline for uploading and chatting with an Excel file.
    """
    print("\n--- Running Test: Chat with Excel ---")
    with open(MOCK_EXCEL_FILE, "rb") as f:
        files = {
            "file": (
                "mock_file.xlsx",
                f,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        }
        upload_response = client.post(
            "/file/upload",
            data={"username": "testuser", "is_image": "false"},
            files=files,
        )

    assert upload_response.status_code == 200
    upload_json = upload_response.json()
    file_id = upload_json.get("file_id")
    session_id = upload_json.get("session_id")
    assert file_id and session_id
    print(f"Uploaded Excel file. Received file_id: {file_id}")

    # Poll the status endpoint to ensure the Excel file is processed
    print(f"Polling for status of file_id: {file_id}...")
    start_time = time.time()
    while time.time() - start_time < 60:  # 1-minute timeout for Excel processing
        status_response = client.get(f"/embeddings/status/{file_id}")
        assert status_response.status_code == 200
        if status_response.json().get("can_chat"):
            print(f"Success! Excel file {file_id} is ready for chat.")
            break
        time.sleep(3)
    else:
        pytest.fail(f"Timeout waiting for Excel file {file_id} to process.")

    # Chat with the processed Excel file
    chat_data = {
        "text": ["Employee Krista Orcutt is from which location?"],
        "file_id": file_id,
        "session_id": session_id,
        "model_choice": "gpt_4o_mini",
        "user_id": "testuser",
    }
    chat_response = client.post("/file/chat", json=chat_data)

    assert chat_response.status_code == 200
    response_json = chat_response.json()
    response_text = response_json.get("response", "")
    print(f"Received chat response for Excel: '{response_text}'")

    # Assert that the response contains the expected location
    assert (
        "Pennsylvania" in response_text
    ), f"Expected 'Pennsylvania' in response: {response_text}"


# def test_chat_with_single_url(client):
#     """
#     Tests the full pipeline for uploading and chatting with a single URL.
#     """
#     print("\n--- Running Test: Chat with Single URL ---")
#     upload_response = client.post(
#         "/file/upload",
#         data={
#             "urls": "https://en.wikipedia.org/wiki/Tesla_Model_3",
#             "username": "testuser",
#         },
#     )

#     assert upload_response.status_code == 200
#     upload_json = upload_response.json()
#     file_id = upload_json.get("file_id")
#     session_id = upload_json.get("session_id")
#     assert file_id and session_id
#     print(f"Uploaded URL content. Received file_id: {file_id}")

#     # Poll the status endpoint to ensure the URL content is processed
#     print(f"Polling for status of file_id: {file_id}...")
#     start_time = time.time()
#     while time.time() - start_time < 120:  # 2-minute timeout for URL processing
#         status_response = client.get(f"/embeddings/status/{file_id}")
#         assert status_response.status_code == 200
#         if status_response.json().get("can_chat"):
#             print(f"Success! URL content {file_id} is ready for chat.")
#             break
#         time.sleep(5)
#     else:
#         pytest.fail(f"Timeout waiting for URL content {file_id} to process.")

#     # Chat with the processed URL content
#     chat_data = {
#         "text": ["What is the Wheelbase for tesla model"],
#         "file_id": file_id,
#         "session_id": session_id,
#         "model_choice": "gpt_4o_mini",
#         "user_id": "testuser",
#     }
#     chat_response = client.post("/file/chat", json=chat_data)

#     assert chat_response.status_code == 200
#     response_json = chat_response.json()
#     response_text = response_json.get("response", "")
#     print(f"Received chat response for URL: '{response_text}'")

#     # Assert that the response contains the expected wheelbase measurement
#     assert any(
#         measurement in response_text
#         for measurement in ["113.8", "2,895", "113", "2895"]
#     ), f"Expected wheelbase measurement (113.8 inches or 2,895 mm) in response: {response_text}"


# def test_chat_with_multiple_urls(client):
#     """
#     Tests the full pipeline for uploading and chatting with multiple URLs.
#     """
#     print("\n--- Running Test: Chat with Multiple URLs ---")

#     # First upload the first URL (Elon Musk)
#     upload_response_1 = client.post(
#         "/file/upload",
#         data={
#             "urls": "https://en.wikipedia.org/wiki/Elon_Musk",
#             "username": "testuser",
#         },
#     )

#     assert upload_response_1.status_code == 200
#     upload_json_1 = upload_response_1.json()
#     file_id_1 = upload_json_1.get("file_id")
#     assert file_id_1
#     print(f"Uploaded first URL content. Received file_id: {file_id_1}")

#     # Wait for first URL to be processed
#     print(f"Polling for status of first URL file_id: {file_id_1}...")
#     start_time = time.time()
#     while time.time() - start_time < 120:
#         status_response = client.get(f"/embeddings/status/{file_id_1}")
#         assert status_response.status_code == 200
#         if status_response.json().get("can_chat"):
#             print(f"Success! First URL content {file_id_1} is ready for chat.")
#             break
#         time.sleep(5)
#     else:
#         pytest.fail(f"Timeout waiting for first URL content {file_id_1} to process.")

#     # Now upload the second URL with the existing file ID
#     upload_response_2 = client.post(
#         "/file/upload",
#         data={
#             "urls": "https://en.wikipedia.org/wiki/Tesla_Model_3",
#             "username": "testuser",
#             "existing_file_ids": file_id_1,
#         },
#     )

#     assert upload_response_2.status_code == 200
#     upload_json_2 = upload_response_2.json()
#     file_ids = upload_json_2.get("file_ids")
#     session_id = upload_json_2.get("session_id")
#     assert file_ids and len(file_ids) == 2 and session_id
#     print(f"Uploaded second URL content. Received file_ids: {file_ids}")

#     # Wait for the new (second) URL to be processed
#     new_file_id = next(fid for fid in file_ids if fid != file_id_1)
#     print(f"Polling for status of second URL file_id: {new_file_id}...")
#     start_time = time.time()
#     while time.time() - start_time < 120:
#         status_response = client.get(f"/embeddings/status/{new_file_id}")
#         assert status_response.status_code == 200
#         if status_response.json().get("can_chat"):
#             print(f"Success! Second URL content {new_file_id} is ready for chat.")
#             break
#         time.sleep(5)
#     else:
#         pytest.fail(f"Timeout waiting for second URL content {new_file_id} to process.")

#     # Chat with both URLs
#     chat_data = {
#         "text": [
#             "From Elon Musk page, in which month was he born? "
#             "From Tesla Model 3 page, what is the Wheelbase for tesla model?"
#         ],
#         "file_ids": file_ids,
#         "session_id": session_id,
#         "model_choice": "gpt_4o_mini",
#         "user_id": "testuser",
#     }
#     chat_response = client.post("/file/chat", json=chat_data)

#     assert chat_response.status_code == 200
#     response_json = chat_response.json()
#     response_text = response_json.get("response", "")
#     print(f"Received chat response for multiple URLs: '{response_text}'")

#     # Assert that the response contains references to both expected answers
#     # For Elon Musk: June or 6
#     # For Tesla Model 3: wheelbase measurements
#     assert any(
#         month in response_text for month in ["June", "6"]
#     ), f"Expected 'June' or '6' for Elon Musk birth month in response: {response_text}"

#     assert any(
#         measurement in response_text
#         for measurement in ["113.8", "2,895", "113", "2895"]
#     ), f"Expected wheelbase measurement (113.8 inches or 2,895 mm) for Tesla in response: {response_text}"


def test_generate_dalle3_images(client):
    """
    Tests the DALL-E 3 image generation endpoint.
    """
    print("\n--- Running Test: Generate DALL-E 3 Images ---")

    # Define the test prompt (content policy safe)
    test_prompt = "A peaceful mountain landscape with a serene lake reflecting the sky, digital art style"

    # Prepare request data for DALL-E 3
    request_data = {
        "prompt": test_prompt,
        "size": "1024x1024",
        "n": 1,  # DALL-E 3 only supports 1 image
        "model_choice": "dall-e-3",
    }

    # Call the image generation endpoint
    response = client.post("/image/generate", json=request_data)

    # Verify response is successful
    assert response.status_code == 200
    response_data = response.json()
    print(f"Received DALL-E 3 response: {response_data.keys()}")

    # Check most important response elements
    assert response_data["success"] is True
    assert "image_url" in response_data
    assert "dall-e" in response_data["model"].lower()

    print(f"Successfully generated DALL-E 3 image with model: {response_data['model']}")


def test_generate_imagen3_images(client):
    """
    Tests the Imagen 3 image generation endpoint.
    """
    print("\n--- Running Test: Generate Imagen 3 Images ---")

    # Define the test prompt (content policy safe)
    test_prompt = (
        "A beautiful garden with colorful flowers and butterflies, watercolor style"
    )

    # Prepare request data for Imagen 3
    request_data = {
        "prompt": test_prompt,
        "size": "1024x1024",
        "n": 2,  # Imagen supports multiple images
        "model_choice": "imagen-3.0-generate-002",
    }

    # Call the image generation endpoint
    response = client.post("/image/generate", json=request_data)

    # Verify response is successful
    assert response.status_code == 200
    response_data = response.json()
    print(f"Received Imagen 3 response: {response_data.keys()}")

    # Check most important response elements
    assert response_data["success"] is True
    assert "image_urls" in response_data
    assert "imagen" in response_data["model"].lower()
    assert len(response_data["image_urls"]) == 2  # Requested 2 images

    print(
        f"Successfully generated {len(response_data['image_urls'])} Imagen 3 images "
        f"with model: {response_data['model']}"
    )
