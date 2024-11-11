import base64
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

API_URL = "http://localhost:8080"


def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()


def cleanup_files():
    response = requests.post(
        f"{API_URL}/file/cleanup",
        json={"is_manual": True},
        headers={"Content-Type": "application/json"},
    )
    if response.status_code == 200:
        st.success("Cleanup completed successfully")
        reset_session()
    else:
        st.error(f"Cleanup failed: {response.text}")


def handle_file_upload():
    st.session_state.file_type = st.radio(
        "Select file type:", ["PDF", "CSV", "Image"], horizontal=True
    )

    is_image = st.session_state.file_type == "Image"
    if is_image:
        file_types = ["jpg", "png"]
    elif st.session_state.file_type in ["CSV"]:
        file_types = ["xlsx", "xls", "csv"]
    else:  # PDF
        file_types = ["pdf"]

    uploaded_file = st.file_uploader(
        f"Choose a {st.session_state.file_type} file", type=file_types
    )

    if uploaded_file is not None:
        if st.button("Upload and Process File"):
            with st.spinner("Uploading and processing file..."):
                files = {"file": uploaded_file}
                data = {
                    "is_image": str(is_image),
                    "username": st.session_state.username,
                }
                upload_response = requests.post(
                    f"{API_URL}/file/upload", files=files, data=data
                )

                if upload_response.status_code == 200:
                    upload_result = upload_response.json()
                    file_id = upload_result["file_id"]
                    st.session_state.file_id = file_id
                    st.session_state.file_uploaded = True

                    # Step 2: Create embeddings or prepare SQLite DB
                    with st.spinner("Creating embeddings or preparing database..."):
                        if st.session_state.file_type in ["PDF", "Image"]:
                            embed_response = requests.post(
                                f"{API_URL}/embeddings/create",
                                json={
                                    "file_id": file_id,
                                    "is_image": is_image,
                                },
                            )
                            if embed_response.status_code == 200:
                                st.success(
                                    "File processed and embeddings created successfully."
                                )
                        else:  # CSV/Excel
                            st.success(
                                "File processed and SQLite database created successfully."
                            )

                    if is_image:
                        st.session_state.uploaded_image = uploaded_file

                    # Reset messages when a new file is uploaded
                    st.session_state.messages = []
                else:
                    st.error(f"File upload failed: {upload_response.text}")

    # Display image in sidebar if it's an image file
    if is_image and st.session_state.uploaded_image is not None:
        with st.sidebar:
            st.subheader("Uploaded Image:")
            img = Image.open(st.session_state.uploaded_image)
            st.image(img, use_column_width=True)


def display_chat_interface():
    if st.session_state.file_uploaded and st.session_state.file_id:
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
                # Chat request
                chat_payload = {
                    "text": user_input,
                    "file_id": st.session_state.file_id,
                    "model_choice": st.session_state.model_choice,
                }
                chat_response = requests.post(f"{API_URL}/file/chat", json=chat_payload)

                if chat_response.status_code == 200:
                    chat_result = chat_response.json()

                    # Display chat response
                    ai_message = {
                        "role": "assistant",
                        "content": f"{chat_result['response']}\n\nModel: **{st.session_state.model_choice}**",
                    }
                    if "chart_data" in chat_result:
                        ai_message["chart_data"] = chat_result["chart_data"]
                    st.session_state.messages.append(ai_message)

                    with st.chat_message("assistant"):
                        st.markdown(chat_result["response"])
                        st.markdown(f"Model: **{st.session_state.model_choice}**")
                        if "chart_data" in chat_result:
                            chart_data = chat_result["chart_data"]["chart_data"]
                            image_data = base64.b64decode(chart_data)
                            image = Image.open(BytesIO(image_data))
                            st.image(
                                image, caption=chat_result["chart_data"]["chart_title"]
                            )

                    # Display image in sidebar if it's an image file
                    if (
                        st.session_state.file_type == "Image"
                        and st.session_state.uploaded_image is not None
                    ):
                        with st.sidebar:
                            st.subheader("Uploaded Image:")
                            img = Image.open(st.session_state.uploaded_image)
                            st.image(img, use_column_width=True)

                    # Nearest neighbors request (only for PDF and Image files)
                    if st.session_state.file_type in ["PDF", "Image"]:
                        neighbors_payload = {
                            "text": user_input,
                            "file_id": st.session_state.file_id,
                            "n_neighbors": 3,
                        }
                        neighbors_response = requests.post(
                            f"{API_URL}/file/neighbors", json=neighbors_payload
                        )
                        if neighbors_response.status_code == 200:
                            neighbors_result = neighbors_response.json()
                            with st.sidebar:
                                st.subheader("Nearest Neighbors:")
                                for i, neighbor in enumerate(
                                    neighbors_result["neighbors"], 1
                                ):
                                    st.write(f"{i}. {neighbor}")
                else:
                    st.error(f"Request failed: {chat_response.text}")
    else:
        st.warning("Please upload and process a file first")


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
            st.session_state.available_models = ["gpt_4o_mini", "gemini-pro"]
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = "gpt_4o_mini"
    if "file_type" not in st.session_state:
        st.session_state.file_type = "PDF"
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None


def on_model_change():
    st.session_state.model_choice = st.session_state.temp_model_choice


def initialize_model(model_choice):
    if (
        not st.session_state.model_initialized
        or st.session_state.model_choice != model_choice
    ):
        st.session_state.model_choice = model_choice
        if st.session_state.file_id:
            response = requests.post(
                f"{API_URL}/model/initialize",
                json={
                    "model_choice": model_choice,
                    "file_id": st.session_state.file_id,
                },
            )
            if response.status_code == 200:
                st.session_state.model_initialized = True
                st.success(f"Model {model_choice} initialized successfully")
            else:
                st.error(f"Model initialization failed: {response.text}")
        else:
            st.session_state.model_initialized = False
            st.success(
                f"Model {model_choice} selected. Please upload a file to initialize."
            )


def main():
    st.title("RAG Chatbot")

    username = st.text_input("Enter your username:")
    st.session_state.username = username
    initialize_session_state()

    # Add New Chat button at the top right
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("New Chat"):
            cleanup_files()
            st.rerun()

    # Model selection dropdown - Fixed version
    st.selectbox(
        "Select Model",
        options=st.session_state.available_models,
        index=st.session_state.available_models.index(st.session_state.model_choice),
        key="temp_model_choice",
        on_change=on_model_change,
    )

    handle_file_upload()
    display_chat_interface()


if __name__ == "__main__":
    main()
