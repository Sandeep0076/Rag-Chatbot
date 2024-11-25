import os
import random

from locust import HttpUser, between, task

from tests.load_tests.helpers import get_random_file_id_from_file

BEARER_TOKEN = os.getenv("idtoken")
ENV = os.getenv("ENV", "local")

# a list of file ids to pick up from
file_ids = list()

# prompts for pdf chat
with open("tests/resources/chat-pdf.prompts.txt", "r", encoding="utf-8") as file:
    # read each line, strip newlines, and return as a list
    pdf_prompts = [line.strip() for line in file.readlines()]


class FastAPILoadTest(HttpUser):
    wait_time = between(1, 2)  # Users wait between 1 and 2 seconds

    @task(weight=10)
    def chat(self):
        """"""
        prompt = random.choice(pdf_prompts)
        file_id = get_random_file_id_from_file(
            file_path="tests/resources/latest_file_ids.txt"
        )

        # example data for the chat endpoint
        data = {
            "text": [prompt],
            "file_id": file_id,
            "model_choice": "gpt_4o_mini",
            "user_id": "zloch@netrtl.com",
        }

        # Bearer token
        headers = {
            "Authorization": f"Bearer {BEARER_TOKEN}",
            "Content-Type": "application/json",  # Ensure the request content type is JSON
        }

        self.client.post("/file/chat", json=data, headers=headers)

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
        scenario = random.choice([self.chat, self.available_models])
        scenario()
