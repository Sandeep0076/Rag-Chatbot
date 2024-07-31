import base64
import json
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

# Replace with your FastAPI endpoint
API_URL = "http://localhost:8080"

st.title("RAG Chatbot")

# Cleanup button
if st.button("Cleanup Files"):
    response = requests.post(f"{API_URL}/file/cleanup")
    if response.status_code == 200:
        st.success("Cleanup completed successfully")
    else:
        st.error(f"Cleanup failed: {response.text}")

# File upload
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "jpg", "png"])
contain_multimedia = st.checkbox("Contains multimedia")

if uploaded_file is not None:
    files = {"file": uploaded_file}
    data = {"contain_multimedia": json.dumps(contain_multimedia)}
    response = requests.post(f"{API_URL}/file/upload", files=files, data=data)
    if response.status_code == 200:
        file_id = response.json()["file_id"]
        st.success(f"File uploaded successfully. File ID: {file_id}")

        # Preprocess button
        if st.button("Preprocess File"):
            preprocess_response = requests.post(
                f"{API_URL}/file/preprocess",
                json={"file_id": file_id, "contain_multimedia": contain_multimedia},
            )
            if preprocess_response.status_code == 200:
                st.success("File preprocessed successfully")
            else:
                st.error(f"Preprocessing failed: {preprocess_response.text}")
    else:
        st.error("File upload failed")

# Get available models
response = requests.get(f"{API_URL}/available-models")
if response.status_code == 200:
    available_models = response.json()["models"]
else:
    available_models = ["gpt-3.5-turbo"]  # Fallback option

# Chat interface
file_id = st.text_input("Enter File ID")
model_choice = st.selectbox("Select Model", options=available_models)
user_input = st.text_input("Enter your message")

if st.button("Send"):
    if file_id and user_input:
        payload = {"text": user_input, "file_id": file_id, "model_choice": model_choice}
        response = requests.post(f"{API_URL}/file/chat", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.write("Response:", result["response"])

            if "chart_data" in result:
                chart_data = result["chart_data"]
                image_data = base64.b64decode(chart_data)
                image = Image.open(BytesIO(image_data))
                st.image(image, caption="Generated Chart")
        else:
            st.error(f"Chat request failed: {response.text}")
    else:
        st.warning("Please enter both File ID and message")
