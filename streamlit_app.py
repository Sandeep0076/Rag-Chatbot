import base64
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

API_URL = "http://localhost:8080"


def initialize_session_state():
    if "file_uploaded" not in st.session_state:
        st.session_state.file_uploaded = False
    if "file_id" not in st.session_state:
        st.session_state.file_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "available_models" not in st.session_state:
        response = requests.get(f"{API_URL}/available-models")
        if response.status_code == 200:
            st.session_state.available_models = response.json()["models"]
        else:
            st.session_state.available_models = ["gpt-3.5-turbo"]
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = st.session_state.available_models[0]


def cleanup_files():
    if st.button("Cleanup Files"):
        response = requests.post(f"{API_URL}/file/cleanup")
        if response.status_code == 200:
            st.success("Cleanup completed successfully")
            st.session_state.file_uploaded = False
            st.session_state.file_id = None
            st.session_state.messages = []
        else:
            st.error(f"Cleanup failed: {response.text}")


def handle_file_upload():
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "jpg", "png"])
    # contain_multimedia = st.checkbox("Contains multimedia")

    if uploaded_file is not None and not st.session_state.file_uploaded:
        if st.button("Upload and Process File"):
            files = {"file": uploaded_file}
            data = {"contain_multimedia": str("false")}

            with st.spinner("Uploading and preprocessing file..."):
                upload_response = requests.post(
                    f"{API_URL}/file/upload", files=files, data=data
                )
                if upload_response.status_code == 200:
                    upload_result = upload_response.json()
                    file_id = upload_result["file_id"]
                    st.success(
                        f"File uploaded and preprocessed successfully. File ID: {file_id}"
                    )
                    st.session_state.file_id = file_id
                    st.session_state.file_uploaded = True
                else:
                    st.error(
                        f"File upload and preprocessing failed: {upload_response.text}"
                    )


def display_chat_interface():
    if st.session_state.file_uploaded:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "chart_data" in message:
                    image_data = base64.b64decode(message["chart_data"]["chart_data"])
                    image = Image.open(BytesIO(image_data))
                    st.image(image, caption=message["chart_data"]["chart_title"])

        user_input = st.chat_input("Enter your message")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.spinner("Processing your request..."):
                payload = {
                    "text": user_input,
                    "file_id": st.session_state.file_id,
                    "model_choice": st.session_state.model_choice,
                }
                response = requests.post(f"{API_URL}/file/chat", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    ai_message = {"role": "assistant", "content": result["response"]}
                    if "chart_data" in result:
                        ai_message["chart_data"] = result["chart_data"]
                    st.session_state.messages.append(ai_message)

                    with st.chat_message("assistant"):
                        st.write(result["response"])
                        if "chart_data" in result:
                            chart_data = result["chart_data"]["chart_data"]
                            image_data = base64.b64decode(chart_data)
                            image = Image.open(BytesIO(image_data))
                            st.image(image, caption=result["chart_data"]["chart_title"])
                else:
                    st.error(f"Chat request failed: {response.text}")
    else:
        st.warning("Please upload and process a file first")


def main():
    st.title("RAG Chatbot")
    initialize_session_state()
    cleanup_files()
    handle_file_upload()
    st.session_state.model_choice = st.selectbox(
        "Select Model", options=st.session_state.available_models, key="model_select"
    )
    display_chat_interface()


if __name__ == "__main__":
    main()
