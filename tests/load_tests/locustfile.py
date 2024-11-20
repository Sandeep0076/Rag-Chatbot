import os
import random

from locust import HttpUser, between, task

from tests.load_tests.helpers import get_random_pdf_file

BEARER_TOKEN = os.getenv("idtoken")
ENV = os.getenv("ENV", "local")


# a list of file ids to save for later usage (e.g. /file/chat)
if os.path.exists("tests/resources/latest_file_ids.txt"):
    os.remove("tests/resources/latest_file_ids.txt")


class FastAPILoadTest(HttpUser):
    wait_time = between(1, 2)  # Users wait between 1 and 2 seconds

    # prompts for pdf chat
    with open("tests/resources/chat-pdf.prompts.txt", "r", encoding="utf-8") as file:
        # read each line, strip newlines, and return as a list
        pdf_prompts = [line.strip() for line in file.readlines()]

    with open("tests/resources/chat.prompts.txt", "r", encoding="utf-8") as file:
        # read each line, strip newlines, and return as a list
        prompts = [line.strip() for line in file.readlines()]

    @task(weight=10)
    def upload_file_and_create_embeddings(self):
        # randomly choose a pdf file
        selected_file_path = get_random_pdf_file(source_dir="tests/resources")

        # Open the file in binary mode
        with open(selected_file_path, "rb") as file:
            file_data = {
                "file": (selected_file_path.split("/")[-1], file, "application/pdf"),
                "is_image": (None, "false"),  # Treat `is_image` as a form field
                "username": (
                    None,
                    "zloch@netrtl.com",
                ),  # Treat `username` as a form field
            }
            # Bearer token
            headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
            response = self.client.post(
                "/file/upload",
                files=file_data,
                headers=headers if ENV != "local" else None,
            )

            # Check if the response was successful
            if response.status_code == 200:
                # Parse the response JSON and extract the file_id
                response_data = response.json()
                file_id = response_data.get("file_id")

                with open("tests/resources/latest_file_ids.txt", "a") as fids:
                    fids.write(f"{file_id}\n")
            else:
                raise Exception(response.reason)

            # data for /embeddings/create-endpoint
            data = {"file_id": file_id, "is_image": "false"}

            # Bearer token
            headers = {
                "Authorization": f"Bearer {BEARER_TOKEN}",
                "Content-Type": "application/json",  # Ensure the request content type is JSON
            }

            self.client.post(
                "/embeddings/create",
                json=data,
                headers=headers if ENV != "local" else None,
            )
            print(f"Create embeddings for {file_id}")

    @task(weight=5)
    def available_models(self):
        """"""
        # Bearer token
        headers = {
            "Authorization": f"Bearer {BEARER_TOKEN}",
            "Content-Type": "application/json",  # Ensure the request content type is JSON
        }

        self.client.get(
            "/available-models", headers=headers if ENV != "local" else None
        )

    # Simulate multiple users accessing different endpoints in random order
    @task(3)
    def random_scenario(self):
        scenario = random.choice(
            [self.upload_file_and_create_embeddings, self.available_models]
        )
        scenario()
