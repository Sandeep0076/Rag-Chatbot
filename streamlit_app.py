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
    if "model_initialized" not in st.session_state:
        st.session_state.model_initialized = False


def cleanup_files():
    if st.button("Cleanup Files"):
        response = requests.post(f"{API_URL}/file/cleanup")
        if response.status_code == 200:
            st.success("Cleanup completed successfully")
            st.session_state.file_uploaded = False
            st.session_state.file_id = None
            st.session_state.messages = []
            st.session_state.model_initialized = False
        else:
            st.error(f"Cleanup failed: {response.text}")


def handle_file_upload():
    st.session_state.file_type = st.radio(
        "Select file type:", ["PDF", "Image"], horizontal=True
    )

    if st.session_state.file_type == "PDF":
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        is_image = False
    else:
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png"])
        is_image = True

    if uploaded_file is not None:
        if st.button("Upload and Process File"):
            files = {"file": uploaded_file}
            data = {"is_image": str(is_image)}
            if is_image:
                st.warning("Processing images may take longer. Please be patient.")

            with st.spinner("Uploading and preprocessing file..."):
                upload_response = requests.post(
                    f"{API_URL}/file/upload", files=files, data=data
                )

                if upload_response.status_code == 200:
                    upload_result = upload_response.json()
                    file_id = upload_result["file_id"]
                    st.success("File uploaded and preprocessed successfully.")
                    st.session_state.file_id = file_id
                    st.session_state.file_uploaded = True
                    if is_image:
                        st.session_state.uploaded_image = uploaded_file
                    # Reset messages when a new file is uploaded
                    st.session_state.messages = []
                    st.session_state.model_initialized = False

                    # Initialize the model after successful upload
                    try:
                        initialize_model()
                    except Exception as e:
                        st.error(f"Failed to initialize model: {str(e)}")
                        st.session_state.model_initialized = False
                else:
                    st.error(
                        f"File upload and preprocessing failed: {upload_response.text}"
                    )


def initialize_model():
    if st.session_state.file_uploaded and st.session_state.file_id:
        with st.spinner("Initializing model..."):
            response = requests.post(
                f"{API_URL}/model/initialize",
                json={
                    "file_id": st.session_state.file_id,
                    "model_choice": st.session_state.model_choice,
                },
            )
            if response.status_code == 200:
                st.success("Model initialized successfully")
                st.session_state.model_initialized = True
            else:
                raise Exception(f"Model initialization failed: {response.text}")


def display_chat_interface():
    if st.session_state.file_uploaded and st.session_state.model_initialized:
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
                chat_payload = {
                    "text": user_input,
                    "file_id": st.session_state.file_id,
                    "model_choice": st.session_state.model_choice,
                }
                chat_response = requests.post(f"{API_URL}/file/chat", json=chat_payload)

                if chat_response.status_code == 200:
                    chat_result = chat_response.json()
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

                    if (
                        st.session_state.file_type == "Image"
                        and st.session_state.uploaded_image is not None
                    ):
                        with st.sidebar:
                            st.subheader("Uploaded Image:")
                            img = Image.open(st.session_state.uploaded_image)
                            img_bytes = BytesIO()
                            img.save(img_bytes, format="PNG")
                            img_str = base64.b64encode(img_bytes.getvalue()).decode()
                            href = (
                                f'<a href="data:image/png;base64,{img_str}" target="_blank">'
                                f'<img src="data:image/png;base64,{img_str}" width="100%">'
                                "</a>"
                            )
                            st.markdown(href, unsafe_allow_html=True)
                else:
                    st.error(f"Request failed: {chat_response.text}")
    elif not st.session_state.file_uploaded:
        st.warning("Please upload and process a file first")
    elif not st.session_state.model_initialized:
        st.warning("Model is not initialized. Please wait or try reinitializing.")


def main():
    st.title("RAG Chatbot")
    initialize_session_state()

    # Model selection dropdown at the top
    selected_model = st.selectbox(
        "Select Model", options=st.session_state.available_models, key="model_select"
    )

    # Check if model selection has changed
    if selected_model != st.session_state.model_choice:
        st.session_state.model_choice = selected_model
        st.session_state.model_initialized = False  # Reset initialization flag
        initialize_model()  # Reinitialize with new model

    cleanup_files()
    handle_file_upload()
    display_chat_interface()


if __name__ == "__main__":
    main()
