import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

from streamlit_image_generation import display_app_header, handle_image_generation

st.set_page_config(
    page_title="RTL-Deutschland RAG_CHATBOT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for glassy blue-purple gradient background (main + sidebar + buttons)

custom_css = """
<style>
    body, .stApp {
        background: linear-gradient(135deg, #cbe5fd 0%, #d7d8f8 60%, #b9b7f8 100%);
        background-attachment: fixed;
    }
    /* Glass effect for main content */
    .stApp > header, .stApp > div:first-child {
        background: rgba(255, 255, 255, 0.35);
        backdrop-filter: blur(8px) saturate(180%);
        -webkit-backdrop-filter: blur(8px) saturate(180%);
        border-radius: 16px;
        box-shadow: 0 4px 32px 0 rgba(31, 38, 135, 0.12);
    }
    /* Glassy gradient sidebar */
    section[data-testid="stSidebar"], .stSidebar {
        background: linear-gradient(135deg, #cbe5fd 0%, #d7d8f8 60%, #b9b7f8 100%) !important;
        background-attachment: fixed !important;
        backdrop-filter: blur(10px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(10px) saturate(180%) !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 32px 0 rgba(31, 38, 135, 0.10) !important;
        opacity: 0.95;
    }
    /* --- BUTTONS & RADIO --- */
    /* Style for radio buttons */
    div[data-baseweb="radio"] label {
        background: rgba(203, 229, 253, 0.7);
        border-radius: 12px;
        padding: 6px 14px;
        margin-bottom: 6px;
        color: #1E3A8A;
        font-weight: 500;
        transition: background 0.2s, color 0.2s;
        box-shadow: 0 1px 6px 0 rgba(31,38,135,0.07);
    }
    div[data-baseweb="radio"] label[data-checked="true"],
    div[data-baseweb="radio"] input[type="radio"]:checked + div {
        background: linear-gradient(90deg, #7ecbff 0%, #b9b7f8 100%);
        color: #fff;
        font-weight: 700;
        box-shadow: 0 2px 10px 0 rgba(31,38,135,0.13);
    }
    /* Style for navigation buttons (smaller size) */
    .nav-btn {
        padding: 4px 12px !important;
        font-size: 13px !important;
        height: 32px !important;
        min-width: 60px !important;
        border-radius: 16px !important;
        background: linear-gradient(90deg, #7ecbff 0%, #b9b7f8 100%) !important;
        color: #1E3A8A !important;
        font-weight: 500 !important;
        border: none !important;
        margin-right: 6px !important;
        box-shadow: 0 1px 6px 0 rgba(31,38,135,0.07) !important;
        transition: background 0.2s, color 0.2s;
    }
    .nav-btn.selected {
        background: linear-gradient(90deg, #b9b7f8 0%, #7ecbff 100%) !important;
        color: #fff !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 10px 0 rgba(31,38,135,0.13) !important;
    }
    /* Style for Streamlit tab buttons */
    button[data-baseweb="tab"] {
        background: rgba(203, 229, 253, 0.7);
        border-radius: 10px 10px 0 0;
        color: #1E3A8A;
        font-weight: 500;
        border: none;
        margin-right: 4px;
        padding: 8px 20px;
        transition: background 0.2s, color 0.2s;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, #7ecbff 0%, #b9b7f8 100%);
        color: #fff;
        font-weight: 700;
        box-shadow: 0 2px 10px 0 rgba(31,38,135,0.13);
    }
    /* Comprehensive styling for all chat-related elements */
    /* Make all chat containers transparent or glassy */
    .stChatFloatingInputContainer,
    .stChatFloatingInput,
    .stChatInput,
    .stChatContainer,
    .stChatMessage,
    section[data-testid="stChatMessageContainer"],
    div[data-testid="stChatMessageContainer"],
    footer.stChatInputContainer,
    .stTextInput > div,
    div[data-testid="stFormSubmitButton"] > div,
    /* Target all possible parent containers */
    div.main > div > div > div > div > div > footer,
    div.main > div > div > div > div > footer,
    div.main > div > div > div > footer,
    div.main > div > div > footer,
    .element-container:has(footer) {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Target the white background container specifically */
    div.stChatContainer > div:first-child,
    footer.stChatInputContainer,
    div.stChatFloatingInputContainer,
    /* Target dynamically loaded containers */
    div[data-testid="stFormSubmitButton"] > div,
    div.main div.element-container:has(footer) > div,
    div.main footer,
    div.main div:has(> footer) {
        background: rgba(255, 255, 255, 0.25) !important;
        background-color: rgba(255, 255, 255, 0.25) !important;
        backdrop-filter: blur(8px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(8px) saturate(180%) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(185, 183, 248, 0.3) !important;
        box-shadow: 0 4px 20px 0 rgba(31, 38, 135, 0.10) !important;
    }

    /* Style for all input fields in chat */
    div[data-testid="stChatInput"] input,
    div[data-testid="stTextInput"] input,
    .stChatFloatingInput input,
    .stChatInputContainer input,
    footer input,
    div.main input[type="text"] {
        background: rgba(255, 255, 255, 0.35) !important;
        backdrop-filter: blur(8px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(8px) saturate(180%) !important;
        border-radius: 12px !important;
        border: 1.5px solid #b9b7f8 !important;
        color: #1E3A8A !important;
        font-weight: 500;
        box-shadow: 0 2px 10px 0 rgba(31,38,135,0.10) !important;
    }

    /* Super aggressive targeting of ANY white backgrounds */
    div.main > div > div > div > div,
    div.block-container,
    div[data-testid="stVerticalBlock"] > div,
    div.stMarkdown,
    div.stFileUploader,
    .stFileUploader > div,
    div.stButton,
    div.stMarkdown div,
    div.element-container,
    div.stAlert,
    div.stSuccessAlert,
    div.stSpinner,
    /* Target chat message containers specifically */
    div.stChatMessageContent,
    div.stChatMessage,
    div[data-testid="stChatMessage"],
    div.stChatMessageContent > div,
    /* Target the bottom input area specifically */
    footer,
    footer > div,
    div[data-testid="stForm"],
    div[data-testid="stForm"] > div,
    /* Target absolutely all divs in the entire app */
    div.main div,
    div.stApp div,
    div {
        background-color: transparent !important;
    }

    /* Target the specific chat input container at the bottom */
    .stChatInputContainer,
    .stChatFloatingInputContainer,
    footer,
    form,
    div[data-testid="stForm"],
    div[data-baseweb],
    div[data-testid="stFormSubmitButton"],
    /* Target the chat message area */
    section[data-testid="stChatMessageContainer"] > div {
        background: rgba(255, 255, 255, 0.25) !important;
        backdrop-filter: blur(8px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(8px) saturate(180%) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(185, 183, 248, 0.3) !important;
    }

    /* Style user messages differently from assistant messages */
    /* User messages - blue gradient */
    div[data-testid="stChatMessage"]:nth-child(odd) > div {
        background: linear-gradient(90deg, rgba(126, 203, 255, 0.4) 0%, rgba(185, 183, 248, 0.3) 100%) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(126, 203, 255, 0.5) !important;
        box-shadow: 0 2px 10px 0 rgba(31, 38, 135, 0.1) !important;
        margin: 8px 0 !important;
        padding: 10px !important;
    }

    /* Assistant messages - purple gradient */
    div[data-testid="stChatMessage"]:nth-child(even) > div {
        background: linear-gradient(90deg, rgba(185, 183, 248, 0.4) 0%, rgba(126, 203, 255, 0.3) 100%) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(185, 183, 248, 0.5) !important;
        box-shadow: 0 2px 10px 0 rgba(31, 38, 135, 0.1) !important;
        margin: 8px 0 !important;
        padding: 10px !important;
    }

    /* Hide any default icons that might be causing duplication */
    div[data-testid="stChatMessage"] .stAvatar {
        /* Keep the built-in avatars, remove our custom ones */
    }

    /* Make all file uploader and success alerts match the theme */
    .uploadedFile,
    .uploadedFile > div,
    .row-widget.stButton,
    .stAlert,
    .stSuccessAlert,
    .css-1kyxreq,
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.25) !important;
        backdrop-filter: blur(8px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(8px) saturate(180%) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(185, 183, 248, 0.3) !important;
    }

    /* Force the entire app body to have the gradient background */
    body::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #cbe5fd 0%, #d7d8f8 60%, #b9b7f8 100%);
        z-index: -1;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

API_URL = "http://localhost:8080"


# Custom CSS for better styling
def apply_custom_css():
    st.markdown(
        """
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 600;
            color: #1E3A8A;
            margin-bottom: 1rem;
            text-align: center;
        }
        .subheader {
            font-size: 1.5rem;
            font-weight: 500;
            color: #2563EB;
            margin-bottom: 1rem;
        }
        .sidebar-header {
            font-size: 1.2rem;
            font-weight: 600;
            color: #1E3A8A;
            margin-top: 1rem;
        }
        .nav-item {
            padding: 0.5rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
            font-weight: 500;
            cursor: pointer;
        }
        .nav-item:hover {
            background-color: #E5E7EB;
        }
        .nav-item-active {
            background-color: #DBEAFE;
            color: #1E40AF;
        }
        .stButton > button {
            width: 100%;
            border-radius: 0.5rem;
            font-weight: 500;
        }
        .upload-section {
            background-color: #F3F4F6;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .chat-container {
            border: 1px solid #E5E7EB;
            border-radius: 0.5rem;
            padding: 0.5rem;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
            overflow-y: auto;
        }
        .file-info {
            background-color: #DBEAFE;
            padding: 0.25rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .stTextInput > div > div > input {
            border-radius: 0.5rem;
        }
        /* Navigation bar styling */
        .stRadio > div {
            display: flex;
            gap: 0px;
        }
        .stRadio > div > label {
            background-color: #f0f2f6;
            border-radius: 5px;
            padding: 10px 15px;
            cursor: pointer;
            margin-right: 10px;
            transition: background-color 0.3s;
            font-weight: 600;
        }
        .stRadio > div > div [data-baseweb="radio"] {
            margin-right: 20px;
        }
        .stRadio > div > div [data-baseweb="radio"]:checked + label {
            background-color: #dbeafe;
            color: #1E40AF;
            border-bottom: 2px solid #1E40AF;
        }
        .stRadio > div > div [data-baseweb="radio"]:hover + label {
            background-color: #dbeafe;
        }
        /* Navigation section styling */
        .nav-container {
            padding: 5px 0;
            margin-bottom: 10px;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


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

    # Visualization is now automatically detected by the backend
    # No need for a manual toggle

    is_image = st.session_state.file_type == "Image"
    file_types = {
        "Image": ["jpg", "png"],
        "CSV/Excel": ["xlsx", "xls", "csv"],
        "Database": ["db", "sqlite"],
        "PDF": ["pdf"],
        "Text": ["txt", "doc", "docx"],
        "URL": [],  # No file types for URL
    }[st.session_state.file_type]

    # Display help text for database files
    if st.session_state.file_type == "Database":
        st.info(
            "Upload SQLite database files (.db or .sqlite) to chat with their contents. "
        )

    # Handle URL input
    if st.session_state.file_type == "URL":
        st.info(
            "Enter one or more URLs separated by commas to chat with their contents."
        )
        url_input = st.text_area("Enter URLs (comma-separated for multiple URLs)")

        if url_input and st.button("Process URLs"):
            if not st.session_state.username:
                st.error("Username is required. Please enter a username above.")
                return

            with st.spinner("Processing URLs..."):
                # Use the existing file upload endpoint with is_url=True
                data = {
                    "username": st.session_state.username,
                    "is_url": "true",
                    "urls": url_input,
                }

                upload_response = requests.post(f"{API_URL}/file/upload", data=data)

                if upload_response.status_code == 200:
                    upload_result = upload_response.json()
                    file_id = upload_result["file_id"]
                    st.session_state.file_id = file_id
                    st.session_state.file_uploaded = True

                    st.success("URLs processed successfully and ready for chat.")

                    # Reset messages when new URLs are processed
                    st.session_state.messages = []
                else:
                    st.error(f"URL processing failed: {upload_response.text}")
    else:
        uploaded_file = st.file_uploader(
            f"Choose a {st.session_state.file_type} file", type=file_types
        )

    # Only show the upload button if we're not in URL mode and a file has been selected
    if st.session_state.file_type != "URL" and uploaded_file is not None:
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

    # Display image in a dedicated section if it's an image file
    if is_image and st.session_state.uploaded_image is not None:
        st.markdown('<div class="file-info">', unsafe_allow_html=True)
        st.subheader("Uploaded Image:")
        img = Image.open(st.session_state.uploaded_image)
        st.image(img, width=400)
        st.markdown("</div>", unsafe_allow_html=True)


def display_chat_interface():
    if st.session_state.file_uploaded and st.session_state.file_id:
        # Super-compact display with minimal info to save space
        st.markdown(
            f"<small>File: {st.session_state.file_id} ({st.session_state.file_type})</small>",
            unsafe_allow_html=True,
        )

        # CSS to eliminate any unnecessary gap
        st.markdown(
            """
        <style>
        .block-container {gap: 0.5rem !important;}
        div[data-testid="stVerticalBlock"] {gap: 0.5rem !important;}
        </style>
        """,
            unsafe_allow_html=True,
        )
        # Display messages if any exist
        if len(st.session_state.messages) > 0:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if (
                        isinstance(message["content"], str)
                        and "|" in message["content"]
                        and "\n" in message["content"]
                    ):
                        st.markdown(message["content"])
                    else:
                        st.write(message["content"])

        # Chat input - immediately below file info with no gap
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

                    # Handle visualization data - automatically detected by backend
                    if "chart_config" in chat_result:
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
                            # Removed unused image size and num_images selectors
                            # These were likely intended for image generation but not connected to any functionality
                            st.session_state.messages.append(ai_message)

                            with st.chat_message("assistant"):
                                fig = plot_chart(chart_config)
                                st.plotly_chart(fig)
                        except Exception as e:
                            st.error(f"Error creating chart: {str(e)}")
                            st.write("Raw chart data:", chart_config)
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
    # Initialize session variables if they don't exist
    if "file_uploaded" not in st.session_state:
        st.session_state.file_uploaded = False
    if "file_id" not in st.session_state:
        st.session_state.file_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "available_models" not in st.session_state:
        response = requests.get(f"{API_URL}/available-models")
        if response.status_code == 200:
            response_data = response.json()
            st.session_state.available_models = response_data["models"]
            # Safely handle model_types which may not exist in older API versions
            if "model_types" in response_data:
                st.session_state.model_types = response_data["model_types"]
            else:
                # Default categorization if API doesn't provide it
                # Identify image models by name pattern
                image_models = [
                    m
                    for m in st.session_state.available_models
                    if "dall-e" in m.lower() or "imagen" in m.lower()
                ]
                text_models = [
                    m
                    for m in st.session_state.available_models
                    if m not in image_models
                ]

                st.session_state.model_types = {
                    "text": text_models,
                    "image": image_models,
                    "size": "",
                    "n": 1,
                }
        else:
            st.session_state.available_models = ["gpt_4o_mini", "gemini-pro"]
            # Fallback model types if API call fails
            st.session_state.model_types = {
                "text": ["gpt_4o_mini", "gemini-pro"],
                "image": ["dall-e-3", "imagen-3.0-generate-002"],
            }
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = "gpt_4o_mini"
    if "file_type" not in st.session_state:
        st.session_state.file_type = "PDF"
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    if "model_initialized" not in st.session_state:
        st.session_state.model_initialized = False
    if "nav_option" not in st.session_state:
        st.session_state.nav_option = "Chat"
    # Visualization is now automatically detected by the backend
    # The generate_visualization flag is still kept in session state for API compatibility
    if "generate_visualization" not in st.session_state:
        st.session_state.generate_visualization = False
    if "username" not in st.session_state:
        st.session_state.username = ""


def on_model_change():
    st.session_state.model_choice = st.session_state.temp_model_choice


def initialize_model(model_choice):
    # Update the model choice in session state
    st.session_state.model_choice = model_choice

    # Make sure model_types exists in session state
    if "model_types" not in st.session_state:
        st.session_state.model_types = {"text": [], "image": []}

    # Check if this is an image generation model
    is_image_model = model_choice in st.session_state.model_types.get("image", [])

    # Store model type in session state
    st.session_state.current_model_type = "image" if is_image_model else "text"

    # If it's an image model, we don't need to initialize with a file_id
    if is_image_model:
        st.session_state.model_initialized = True
        return

    # For text models, we need to initialize only if we have a file_id
    if (
        not st.session_state.model_initialized
        or st.session_state.model_choice != model_choice
    ):
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


def create_line_chart(dataset, title, labels):
    fig = go.Figure()
    for data in dataset:
        fig.add_trace(
            go.Scatter(
                x=data["x"], y=data["y"], name=data["label"], mode="lines+markers"
            )
        )
    fig.update_layout(title=title, xaxis_title=labels["x"], yaxis_title=labels["y"])
    return fig


def create_bar_chart(data, title, labels, options=None):
    fig = go.Figure()

    # Handle simplified format (values/categories)
    if "values" in data and "categories" in data:
        fig.add_trace(go.Bar(x=data["categories"], y=data["values"], name="Value"))
    # Handle datasets format
    elif "datasets" in data:
        for dataset in data["datasets"]:
            fig.add_trace(go.Bar(x=dataset["x"], y=dataset["y"], name=dataset["label"]))
    else:
        raise ValueError(
            "Invalid data format for bar chart. Must have either 'values' and 'categories' or 'datasets'"
        )

    if options and options.get("stacked", False):
        fig.update_layout(barmode="stack")

    fig.update_layout(title=title, xaxis_title=labels["x"], yaxis_title=labels["y"])
    return fig


def create_pie_chart(data, title):
    fig = go.Figure(go.Pie(values=data["values"], labels=data["categories"], hole=0.3))
    fig.update_layout(title=title)
    return fig


def create_scatter_plot(dataset, title, labels, is_3d=False):
    fig = go.Figure()
    for data in dataset:
        if is_3d:
            fig.add_trace(
                go.Scatter3d(
                    x=data["x"],
                    y=data["y"],
                    z=data["z"],
                    name=data["label"],
                    mode="markers",
                )
            )
        else:
            scatter_args = {
                "x": data["x"],
                "y": data["y"],
                "name": data["label"],
                "mode": "markers",
            }
            if "size" in data:  # For bubble charts
                scatter_args["marker"] = {"size": data["size"]}
            if "color" in data:
                scatter_args["marker"] = scatter_args.get("marker", {})
                scatter_args["marker"]["color"] = data["color"]

            fig.add_trace(go.Scatter(**scatter_args))

    layout_args = {"title": title}
    if is_3d:
        layout_args.update(
            {
                "scene": {
                    "xaxis_title": labels["x"],
                    "yaxis_title": labels["y"],
                    "zaxis_title": labels["z"],
                }
            }
        )
    else:
        layout_args.update({"xaxis_title": labels["x"], "yaxis_title": labels["y"]})

    fig.update_layout(**layout_args)
    return fig


def create_heatmap(data, title, labels):
    fig = go.Figure(
        go.Heatmap(
            z=data["matrix"],
            x=data.get("x_categories"),
            y=data.get("y_categories"),
            colorscale=data.get("options", {}).get("color_palette", "Viridis"),
        )
    )
    fig.update_layout(title=title, xaxis_title=labels["x"], yaxis_title=labels["y"])
    return fig


def create_box_plot(dataset, title, labels):
    fig = go.Figure()
    for data in dataset:
        fig.add_trace(
            go.Box(
                x=data["x"] if "x" in data else None, y=data["y"], name=data["label"]
            )
        )
    fig.update_layout(title=title, xaxis_title=labels["x"], yaxis_title=labels["y"])
    return fig


def create_histogram(data, title, labels):
    fig = go.Figure(go.Histogram(x=data["values"], nbinsx=30))
    fig.update_layout(title=title, xaxis_title=labels["x"], yaxis_title="Count")
    return fig


def plot_chart(chart_config):
    """Create and return a plotly figure based on the chart configuration."""
    chart_type = chart_config["chart_type"].lower()
    title = chart_config["title"]
    data = chart_config["data"]
    labels = chart_config["labels"]
    options = chart_config.get("options", {})

    chart_creators = {
        "line chart": lambda: create_line_chart(data["datasets"], title, labels),
        "bar chart": lambda: create_bar_chart(data, title, labels, options),
        "pie chart": lambda: create_pie_chart(data, title),
        "scatter plot": lambda: create_scatter_plot(data["datasets"], title, labels),
        "3d scatter plot": lambda: create_scatter_plot(
            data["datasets"], title, labels, is_3d=True
        ),
        "bubble chart": lambda: create_scatter_plot(data["datasets"], title, labels),
        "heatmap": lambda: create_heatmap(data, title, labels),
        "box plot": lambda: create_box_plot(data["datasets"], title, labels),
        "histogram": lambda: create_histogram(data, title, labels),
    }

    if chart_type not in chart_creators:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    return chart_creators[chart_type]()


def render_navigation():
    """Render the top navigation bar with buttons."""
    # Make sure model_types exists in session state
    if "model_types" not in st.session_state:
        st.session_state.model_types = {"text": [], "image": []}

    # If current model is an image model, automatically switch to Image generation tab
    if (
        "current_model_type" in st.session_state
        and st.session_state.current_model_type == "image"
    ):
        st.session_state.nav_option = "Image generation"

    nav_options = ["Chat", "Image generation"]
    nav_cols = st.columns(len(nav_options))
    for i, nav in enumerate(nav_options):
        if nav_cols[i].button(
            nav, key=f"nav_{nav}", help=f"Go to {nav}", use_container_width=False
        ):
            st.session_state.nav_option = nav


def process_url_input(url_input):
    """Process URLs entered by the user."""
    if not st.session_state.username:
        st.error("Username is required. Please enter a username above.")
        return

    with st.spinner("Processing URLs..."):
        data = {
            "username": st.session_state.username,
            "is_url": "true",
            "urls": url_input,
        }

        upload_response = requests.post(f"{API_URL}/file/upload", data=data)

        if upload_response.status_code == 200:
            upload_result = upload_response.json()
            file_id = upload_result["file_id"]
            st.session_state.file_id = file_id
            st.session_state.file_uploaded = True

            st.success("URLs processed successfully and ready for chat.")
            st.session_state.messages = []
        else:
            st.error(f"URL processing failed: {upload_response.text}")


def process_file_upload(uploaded_file, is_image):
    """Process an uploaded file."""
    if not st.session_state.username:
        st.error("Username is required. Please enter a username above.")
        return

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

            if upload_result.get("status") == "success":
                if st.session_state.file_type == "Database":
                    st.success(
                        "Database processed successfully. You can now chat with its contents."
                    )
                else:
                    st.success("File processed successfully and ready for chat.")
            elif upload_result.get("status") == "partial":
                st.warning(upload_result["message"])
            else:
                st.info(upload_result["message"])

            if is_image:
                st.session_state.uploaded_image = uploaded_file

            st.session_state.messages = []
        else:
            st.error(f"File upload failed: {upload_response.text}")


def render_sidebar():
    """Render the sidebar components."""
    # Add New Chat button at the top
    if st.button("New Chat", key="new_chat_btn"):
        cleanup_files()
        st.rerun()

    # Show file type selection only when Chat is selected
    if st.session_state.nav_option == "Chat":
        st.markdown(
            '<div class="sidebar-header">Select file type:</div>',
            unsafe_allow_html=True,
        )
        st.session_state.file_type = st.selectbox(
            "Select file type:",
            ["PDF", "Text", "CSV/Excel", "Database", "Image", "URL"],
            key="file_type_select",
            label_visibility="collapsed",
        )

        # Move file uploader to sidebar (right after file type selection)
        # Always show file uploader in Chat mode regardless of username
        is_image = st.session_state.file_type == "Image"
        file_types = {
            "Image": ["jpg", "png"],
            "CSV/Excel": ["xlsx", "xls", "csv"],
            "Database": ["db", "sqlite"],
            "PDF": ["pdf"],
            "Text": ["txt", "doc", "docx"],
            "URL": [],  # No file types for URL
        }[st.session_state.file_type]

        # Display help text for database files
        if st.session_state.file_type == "Database":
            st.info(
                "Upload SQLite database files (.db or .sqlite) to chat with their contents. "
            )

        # Handle URL input
        if st.session_state.file_type == "URL":
            st.info(
                "Enter one or more URLs separated by commas to chat with their contents."
            )
            url_input = st.text_area("Enter URLs (comma-separated for multiple URLs)")

            if url_input and st.button("Process URLs"):
                process_url_input(url_input)
        else:
            uploaded_file = st.file_uploader(
                f"Choose a {st.session_state.file_type} file", type=file_types
            )

            # Only show the upload button if we're not in URL mode and a file has been selected
            if uploaded_file is not None and st.button("Upload and Process File"):
                process_file_upload(uploaded_file, is_image)

        # Display image in a dedicated section if it's an image file
        if is_image and st.session_state.uploaded_image is not None:
            st.markdown('<div class="file-info">', unsafe_allow_html=True)
            st.subheader("Uploaded Image:")
            img = Image.open(st.session_state.uploaded_image)
            st.image(img, width=200)  # Smaller for sidebar
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

    # User information section - always show
    st.markdown(
        '<div class="sidebar-header">User Information</div>', unsafe_allow_html=True
    )
    username = st.text_input("Enter your username:")
    st.session_state.username = username

    # Model selection dropdown - always show
    st.markdown(
        '<div class="sidebar-header">Model Selection</div>', unsafe_allow_html=True
    )
    st.selectbox(
        "Select Model",
        options=st.session_state.available_models,
        index=st.session_state.available_models.index(st.session_state.model_choice),
        key="temp_model_choice",
        on_change=on_model_change,
    )


def main():
    """Main application function with reduced complexity."""
    # Apply custom CSS
    apply_custom_css()

    # Initialize session state
    initialize_session_state()

    # Main content area - title
    display_app_header()

    # Top navigation bar
    render_navigation()

    # Add horizontal line for visual separation
    st.markdown("<hr/>", unsafe_allow_html=True)

    # Create sidebar with conditional content based on navigation selection
    with st.sidebar:
        render_sidebar()

    # Display different content based on navigation selection
    if st.session_state.nav_option == "Chat":
        # File upload is now handled in the sidebar
        display_chat_interface()
    elif st.session_state.nav_option == "Image generation":
        # Handle image generation interface and API calls
        handle_image_generation()
    elif st.session_state.nav_option == "Chart Generation":
        # Existing code for chart generation
        st.title("Chart Generation")
        st.write("Upload CSV/Excel files and generate visualizations.")
        # Keep the existing chart generation code as is
    elif st.session_state.nav_option == "Reference":
        # Existing code for reference section
        st.title("📚 Reference")
        st.markdown("### Welcome to the RAG Chatbot Reference Page")
        # Keep the existing reference section code as is


if __name__ == "__main__":
    main()
