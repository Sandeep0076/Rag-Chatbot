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
            st.session_state.available_models = ["gpt-3.5-turbo", "gemini-pro"]
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = st.session_state.available_models[0]
    if "file_type" not in st.session_state:
        st.session_state.file_type = "pdf"


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
    st.session_state.file_type = st.radio(
        "Select file type:", ["PDF", "Image"], horizontal=True
    )

    is_image = st.session_state.file_type == "Image"
    file_types = ["jpg", "png"] if is_image else ["pdf"]
    uploaded_file = st.file_uploader(
        f"Choose a {st.session_state.file_type} file", type=file_types
    )

    if uploaded_file is not None:
        if st.button("Upload and Process File"):
            with st.spinner("Uploading and preprocessing file..."):
                # Step 1: Upload file
                files = {"file": uploaded_file}
                data = {
                    "is_image": str(is_image),
                }
                upload_response = requests.post(
                    f"{API_URL}/file/upload", files=files, data=data
                )

                if upload_response.status_code == 200:
                    upload_result = upload_response.json()
                    file_id = upload_result["file_id"]
                    st.success(upload_result["message"])

                    # Step 2: Initialize model
                    init_response = requests.post(
                        f"{API_URL}/model/initialize",
                        json={
                            "model_choice": st.session_state.model_choice,
                            "file_id": file_id,
                        },
                    )
                    if init_response.status_code != 200:
                        st.error(f"Model initialization failed: {init_response.text}")
                        return

                    # Step 3: Create embeddings
                    embed_response = requests.post(
                        f"{API_URL}/embeddings/create",
                        json={
                            "file_id": file_id,
                            "model_choice": st.session_state.model_choice,
                            "is_image": is_image,
                        },
                    )
                    if embed_response.status_code != 200:
                        st.error(f"Embedding creation failed: {embed_response.text}")
                        return

                    st.success("File processed and embeddings created successfully.")
                    st.session_state.file_id = file_id
                    st.session_state.file_uploaded = True
                    if is_image:
                        st.session_state.uploaded_image = uploaded_file

                    # Reset messages when a new file is uploaded
                    st.session_state.messages = []
                else:
                    st.error(f"File upload failed: {upload_response.text}")


def initialize_model(model_choice):
    if st.session_state.file_id:
        response = requests.post(
            f"{API_URL}/model/initialize",
            json={"model_choice": model_choice, "file_id": st.session_state.file_id},
        )
        if response.status_code == 200:
            st.session_state.model_choice = model_choice
            st.success(f"Model {model_choice} initialized successfully")
        else:
            st.error(f"Model initialization failed: {response.text}")
    else:
        st.session_state.model_choice = model_choice
        st.success(
            f"Model {model_choice} selected. Please upload a file to initialize."
        )


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
                        "content": chat_result["response"],
                    }
                    if "chart_data" in chat_result:
                        ai_message["chart_data"] = chat_result["chart_data"]
                    st.session_state.messages.append(ai_message)

                    with st.chat_message("assistant"):
                        st.write(chat_result["response"])
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

                    # Nearest neighbors request
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


def main():
    st.title("RAG Chatbot")
    initialize_session_state()

    # Model selection dropdown
    new_model_choice = st.selectbox(
        "Select Model",
        options=st.session_state.available_models,
        index=st.session_state.available_models.index(st.session_state.model_choice),
        key="model_select",
    )

    # Check if a new model is selected
    if new_model_choice != st.session_state.model_choice:
        initialize_model(new_model_choice)

    cleanup_files()
    handle_file_upload()
    display_chat_interface()


if __name__ == "__main__":
    main()
