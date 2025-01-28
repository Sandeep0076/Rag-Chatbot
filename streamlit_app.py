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
    # Ensure username is available before file upload
    if not st.session_state.username:
        st.warning("Please enter a username before uploading files")
        return

    st.session_state.file_type = st.radio(
        "Select file type:", ["PDF", "CSV/Excel", "Database", "Image"], horizontal=True
    )

    # Add Generate Visualization radio button
    st.session_state.generate_visualization = st.radio(
        "Generate Visualization",
        options=[False, True],
        index=0,  # Default to False
    )

    is_image = st.session_state.file_type == "Image"
    file_types = {
        "Image": ["jpg", "png"],
        "CSV/Excel": ["xlsx", "xls", "csv"],
        "Database": ["db", "sqlite"],
        "PDF": ["pdf"],
    }[st.session_state.file_type]

    # Display help text for database files
    if st.session_state.file_type == "Database":
        st.info(
            "Upload SQLite database files (.db or .sqlite) to chat with their contents. "
        )

    uploaded_file = st.file_uploader(
        f"Choose a {st.session_state.file_type} file", type=file_types
    )

    if uploaded_file is not None:
        if st.button("Upload and Process File"):
            if not st.session_state.username:
                st.error("Username is required. Please enter a username above.")
                return

            with st.spinner("Uploading and processing file..."):
                files = {"file": uploaded_file}
                data = {
                    "is_image": str(is_image),
                    "username": st.session_state.username,  # Make sure username is included
                }
                upload_response = requests.post(
                    f"{API_URL}/file/upload", files=files, data=data
                )

                if upload_response.status_code == 200:
                    upload_result = upload_response.json()
                    file_id = upload_result["file_id"]
                    st.session_state.file_id = file_id
                    st.session_state.file_uploaded = True

                    if upload_result.get("status") == "success":
                        if st.session_state.file_type == "Database":
                            st.success(
                                "Database processed successfully. You can now chat with its contents."
                            )
                        else:
                            st.success(
                                "File processed successfully and ready for chat."
                            )
                    elif upload_result.get("status") == "partial":
                        st.warning(upload_result["message"])
                    else:
                        st.info(upload_result["message"])

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
                # Handle table formatting if content appears to be a markdown table
                if (
                    isinstance(message["content"], str)
                    and "|" in message["content"]
                    and "\n" in message["content"]
                ):
                    st.markdown(message["content"])
                else:
                    st.write(message["content"])

        user_input = st.chat_input("Enter your message")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.spinner("Processing your request..."):
                # Get previous messages to include in history
                previous_messages = [
                    msg["content"]
                    for msg in st.session_state.messages[-5:]  # Get last 5 messages
                    if msg["role"] == "user"  # Only including user messages
                ]

                chat_payload = {
                    "text": previous_messages,  # This will include history and current message
                    "file_id": st.session_state.file_id,
                    "model_choice": st.session_state.model_choice,
                    "user_id": st.session_state.username,
                    "generate_visualization": st.session_state.generate_visualization,
                }
                chat_response = requests.post(f"{API_URL}/file/chat", json=chat_payload)

                if chat_response.status_code == 200:
                    chat_result = chat_response.json()

                    # Handle visualization data
                    if (
                        st.session_state.generate_visualization
                        and "chart_config" in chat_result
                    ):
                        try:
                            chart_config = chat_result["chart_config"]
                            # Create message for chat history
                            ai_message = {
                                "role": "assistant",
                                "content": (
                                    f"Generated {chart_config['chart_type']} "
                                    f"visualization: {chart_config['title']}"
                                ),
                            }
                            st.session_state.messages.append(ai_message)

                            with st.chat_message("assistant"):
                                # Create Plotly figure based on chart data
                                if chart_config["chart_type"].lower() == "line chart":
                                    import plotly.graph_objects as go

                                    fig = go.Figure()
                                    for dataset in chart_config["data"]["datasets"]:
                                        fig.add_trace(
                                            go.Scatter(
                                                x=dataset["x"],
                                                y=dataset["y"],
                                                name=dataset["label"],
                                            )
                                        )

                                    fig.update_layout(
                                        title=chart_config["title"],
                                        xaxis_title=chart_config["labels"]["x"],
                                        yaxis_title=chart_config["labels"]["y"],
                                    )

                                    st.plotly_chart(fig)
                                else:
                                    st.write(
                                        "Unsupported chart type:",
                                        chart_config["chart_type"],
                                    )
                        except KeyError:
                            # If visualization data format is incorrect, display as text
                            ai_message = {
                                "role": "assistant",
                                "content": str(chat_result),
                            }
                            st.session_state.messages.append(ai_message)
                            with st.chat_message("assistant"):
                                st.write(ai_message["content"])

                    # Handle regular text response
                    else:
                        ai_message = {
                            "role": "assistant",
                            "content": chat_result.get("response", str(chat_result)),
                        }
                        st.session_state.messages.append(ai_message)
                        with st.chat_message("assistant"):
                            st.write(ai_message["content"])
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
    if "generate_visualization" not in st.session_state:
        st.session_state.generate_visualization = False


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
