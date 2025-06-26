import logging

import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

from streamlit_image_generation import display_app_header, handle_image_generation

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

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


def handle_file_uploader_change():
    uploader_key = "multi_file_uploader_static"
    if uploader_key in st.session_state and st.session_state[uploader_key] is not None:
        newly_selected_files = st.session_state[uploader_key]
        for uploaded_file_obj in newly_selected_files:
            if not any(
                f.name == uploaded_file_obj.name
                for f in st.session_state.uploaded_files_list
            ):
                st.session_state.uploaded_files_list.append(uploaded_file_obj)
                logging.info(
                    f"Appended {uploaded_file_obj.name} to session_state.uploaded_files_list via on_change."
                )
                # File was successfully added to the list
            else:
                logging.info(
                    f"{uploaded_file_obj.name} is already in the list, not re-adding via on_change."
                )

        # Explicitly clear the file uploader's value in session_state.
        # This tells the widget its current selection has been processed and it should be empty.
        st.session_state[uploader_key] = []

        # if files_actually_added_to_list:
        # Streamlit should naturally rerun if session_state that affects UI has changed.
        # Explicit rerun removed to see if it resolves any subtle state conflicts.


def handle_url_processing(url_input):
    """Handle URL input processing as a separate function."""
    if not url_input:
        return

    if not st.session_state.username:
        st.error("Username is required. Please enter a username above.")
        return

    with st.spinner("Processing URLs..."):
        data = {
            "username": st.session_state.username,
            "is_url": "true",  # Maintained for now, though backend might not strictly need it
            "urls": url_input,
        }

        upload_response = requests.post(f"{API_URL}/file/upload", data=data)

        if upload_response.status_code == 200:
            upload_result = upload_response.json()
            logging.info(f"Received upload_result: {upload_result}")
            if "multi_file_mode" in upload_result:
                logging.info(
                    f"upload_result['multi_file_mode'] value: {upload_result['multi_file_mode']},"
                    f" type: {type(upload_result['multi_file_mode'])}"
                )
            else:
                logging.info("upload_result does not contain 'multi_file_mode'")
            if "file_ids" in upload_result:
                logging.info(
                    f"upload_result['file_ids'] value: {upload_result['file_ids']},\n"
                    f"type: {type(upload_result['file_ids'])}"
                )
            else:
                logging.info("upload_result does not contain 'file_ids'")

            # Prioritize multi_file_mode if explicitly set to True by the backend
            if upload_result.get("multi_file_mode") is True:  # Explicit check for True
                if (
                    isinstance(upload_result.get("file_ids"), list)
                    and upload_result["file_ids"]
                ):
                    # Multi-URL processing successful
                    st.session_state.multi_file_mode = True
                    st.session_state.file_ids = list(upload_result["file_ids"])
                    st.session_state.file_id = st.session_state.file_ids[0]
                    logging.info(
                        f"Multi-URL mode (explicit). st.session_state.file_ids set to: {st.session_state.file_ids}"
                    )
                    st.success(
                        f"Processed {len(st.session_state.file_ids)} URLs successfully. Ready for multi-document chat."
                    )
                else:
                    # multi_file_mode was true, but file_ids was missing or empty - treat as error or single
                    logging.warning(
                        "multi_file_mode is True but file_ids is invalid. Falling back."
                    )
                    if "file_id" in upload_result and upload_result["file_id"]:
                        st.session_state.multi_file_mode = False
                        st.session_state.file_id = upload_result["file_id"]
                        st.session_state.file_ids = [upload_result["file_id"]]
                        logging.info(
                            f"Single-URL mode (fallback from multi).\n"
                            f"st.session_state.file_ids set to: {st.session_state.file_ids}"
                        )
                        st.success("URL processed successfully (fallback).")
                    else:
                        st.error(
                            "Error processing URLs: multi_file_mode True but no valid file_ids or file_id."
                        )
                        return
            elif (
                "file_id" in upload_result and upload_result["file_id"]
            ):  # Check this only if multi_file_mode was not explicitly True
                # Single URL processing successful
                st.session_state.multi_file_mode = False
                st.session_state.file_id = upload_result["file_id"]
                st.session_state.file_ids = [upload_result["file_id"]]
                logging.info(
                    f"Single-URL mode (explicit). st.session_state.file_ids set to: {st.session_state.file_ids}"
                )
                st.success("URL processed successfully and ready for chat.")
            else:
                # Fallback or error in response structure
                st.error(
                    "Error processing URLs: Response format incorrect or no file IDs found."
                )
                return

            st.session_state.file_uploaded = True
            st.session_state.messages = []  # Reset chat history
        else:
            st.error(f"URL processing failed: {upload_response.text}")


def display_uploaded_files_list():
    """Display the list of uploaded files with options to remove them."""
    if not st.session_state.uploaded_files_list:
        return

    st.markdown("---")
    st.markdown("#### Files for Chat:")
    for i, file_in_list in enumerate(st.session_state.uploaded_files_list):
        col1, col2 = st.columns([0.8, 0.2])
        col1.markdown(f"{i + 1}. **{file_in_list.name}**")
        if col2.button("Remove", key=f"remove_btn_{i}"):
            logging.info(f"Removing {file_in_list.name} from upload list.")
            file_to_remove_from_list = st.session_state.uploaded_files_list.pop(i)
            file_name_to_remove = file_to_remove_from_list.name
            logging.info(f"Removing {file_name_to_remove} from UI upload list.")

            # If this file was processed, remove its associated data
            if (
                "processed_file_map" in st.session_state
                and file_name_to_remove in st.session_state.processed_file_map
            ):
                file_id_to_cull = st.session_state.processed_file_map.pop(
                    file_name_to_remove
                )
                if file_id_to_cull in st.session_state.file_ids:
                    st.session_state.file_ids.remove(file_id_to_cull)
                if file_id_to_cull in st.session_state.file_names:
                    del st.session_state.file_names[file_id_to_cull]
                logging.info(
                    f"Removed processed file_id {file_id_to_cull} for {file_name_to_remove}."
                )

                # If the removed file was the currently 'active' single file_id, clear it or pick another
                if st.session_state.file_id == file_id_to_cull:
                    st.session_state.file_id = None
                    st.session_state.file_uploaded = (
                        False  # Mark no single file as 'active'
                    )
                    if (
                        st.session_state.file_ids
                    ):  # If other processed files exist, make the last one 'active'
                        st.session_state.file_id = st.session_state.file_ids[-1]
                        st.session_state.file_uploaded = True
            st.rerun()
    st.markdown(
        "<p><em>All files in this list will provide context during chat.</em></p>",
        unsafe_allow_html=True,
    )


def process_selected_files(uploaded_files, is_image):
    logging.info(
        f"[process_selected_files] Received {len(uploaded_files) if uploaded_files else 0} files from st.file_uploader."
    )
    """Process uploaded files that haven't been processed yet.
    Now batches multiple files in a single request for parallel processing.
    """
    if not st.session_state.username:
        st.error("Username is required. Please enter a username above.")
        return

    # Reset messages when new files are uploaded
    st.session_state.messages = []

    # Filter out already processed files
    files_to_process = []
    for uploaded_file_obj in uploaded_files:
        if uploaded_file_obj.name in st.session_state.get("processed_file_map", {}):
            file_id_check = st.session_state["processed_file_map"][
                uploaded_file_obj.name
            ]
            if file_id_check in st.session_state.get("file_ids", []):
                logging.info(
                    f"Skipping already processed file: {uploaded_file_obj.name} (ID: {file_id_check})"
                )
                continue
        files_to_process.append(uploaded_file_obj)

    if not files_to_process:
        st.info("All files in the list are already processed and ready for chat.")
        return

    # Process files in a single batch request (even for single files)
    logging.info(
        "[process_selected_files] Number of files in files_to_process: "
        f"{len(files_to_process)} before calling process_multiple_files."
    )
    if len(files_to_process) > 0:
        with st.spinner(
            f"Uploading and processing {len(files_to_process)} files in parallel..."
        ):
            # Always use the batch processing method for all files
            # This ensures we use the multi-file upload endpoint even for single files
            process_multiple_files(files_to_process, is_image)

        st.success(
            f"{len(files_to_process)} files have been processed and are ready for chat!"
        )


def process_multiple_files(uploaded_files, is_image):
    logging.info(
        f"[process_multiple_files] Received {len(uploaded_files)} files as parameter."
    )
    """Process multiple files in a single batch request for parallel processing."""
    logging.info(f"Processing {len(uploaded_files)} files in parallel batch")

    # FIXED: Always use 'files' parameter for batch processing, regardless of file count
    # Create proper format for sending multiple files to FastAPI - always as a list of tuples with 'files' key
    files_for_request = []

    # Always format as a list of tuples with 'files' parameter for proper batch processing
    for file_obj in uploaded_files:
        files_for_request.append(
            ("files", (file_obj.name, file_obj.getvalue(), file_obj.type))
        )

    logging.info(
        f"Preparing {len(uploaded_files)} file(s) with key 'files' for batch upload: "
        f"{[f[1][0] for f in files_for_request]}"
    )

    # Log the structure being sent to requests.post for debugging
    logging.debug(
        f"Payload for requests.post (batch upload structure with {len(uploaded_files)} files)"
    )

    logging.info(
        f"[process_multiple_files] Constructed files_for_request payload with {len(files_for_request)} files"
    )
    # Prepare the data portion of the request
    data = {
        "is_image": str(is_image),
        "username": st.session_state.username,
    }

    try:
        # Send the request with our properly formatted files_for_request payload
        upload_response = requests.post(
            f"{API_URL}/file/upload",
            files=files_for_request,  # Use the new payload structure
            data=data,
        )

        logging.debug(
            f"Request sent with {len(uploaded_files)} files. Check server logs for received parameters."
        )
        logging.debug(f"Response status code: {upload_response.status_code}")

        if upload_response.status_code == 200:
            upload_result = upload_response.json()
            logging.info(
                f"Upload successful with status code: {upload_response.status_code}"
            )

            # Process response whether multi-file or single file mode
            # The backend should always return multi_file_mode=True for our batch uploads
            if upload_result.get("multi_file_mode", False):
                # Handle multi-file response
                file_ids = upload_result.get("file_ids", [])
                filenames = upload_result.get("original_filenames", [])
                statuses = upload_result.get("statuses", [])

                logging.info(
                    f"Received {len(file_ids)} file IDs from parallel processing"
                )

                # Set multi_file_mode based on actual number of files
                st.session_state.multi_file_mode = len(file_ids) > 1

                # Store the session_id from the new upload batch
                session_id = upload_result.get("session_id")
                if session_id:
                    logging.info(f"New upload session created with ID: {session_id}")
                    # Reset file_ids list when new files are uploaded
                    # This is the key change for session management - each upload creates a new session
                    st.session_state.file_ids = []
                    st.session_state.current_session_id = session_id

                # Set file_id for single file compatibility (first file)
                st.session_state.file_id = file_ids[0] if file_ids else None
                st.session_state.file_uploaded = True

                # Map filenames to their respective file IDs for processed file tracking
                for i, file_id in enumerate(file_ids):
                    filename = (
                        filenames[i] if i < len(filenames) else uploaded_files[i].name
                    )
                    status = statuses[i] if i < len(statuses) else "success"

                    # Track processed files to avoid reprocessing
                    if "processed_file_map" not in st.session_state:
                        st.session_state.processed_file_map = {}
                    st.session_state.processed_file_map[filename] = file_id

                    # Add to uploaded files list for multi-file chat
                    if "uploaded_files" not in st.session_state or not isinstance(
                        st.session_state.uploaded_files, dict
                    ):
                        st.session_state.uploaded_files = {}

                    st.session_state.uploaded_files[file_id] = {
                        "name": filename,
                        "type": st.session_state.file_type,
                        "session_id": session_id,  # Track which session this file belongs to
                    }
                    st.session_state.file_names[file_id] = filename

                    # Add to file_ids list for the current session's files
                    st.session_state.file_ids.append(file_id)
                    logging.info(
                        f"Added file {filename} (ID: {file_id}) to current session - Status: {status}"
                    )
            else:
                # Handle legacy single file response (shouldn't happen with our new implementation)
                file_id = upload_result.get("file_id")
                filename = upload_result.get("original_filename")

                logging.warning(
                    f"Received unexpected single-file response for file: {filename}"
                )

                # Process it anyway for robustness
                if file_id:
                    # Same processing as above
                    st.session_state.file_id = file_id
                    st.session_state.file_uploaded = True

                    if "processed_file_map" not in st.session_state:
                        st.session_state.processed_file_map = {}
                    st.session_state.processed_file_map[filename] = file_id

                    if "uploaded_files" not in st.session_state or not isinstance(
                        st.session_state.uploaded_files, dict
                    ):
                        st.session_state.uploaded_files = {}

                    st.session_state.uploaded_files[file_id] = {
                        "name": filename,
                        "type": st.session_state.file_type,
                    }
                    st.session_state.file_names[file_id] = filename

                    if file_id not in st.session_state.file_ids:
                        st.session_state.file_ids.append(file_id)
                        logging.info(
                            f"Added file {filename} (ID: {file_id}) to active list (single mode)"
                        )

            return True
        else:
            st.error(f"Batch file upload failed: {upload_response.text}")
            return False
    except Exception as e:
        logging.error(f"Error during batch file upload: {str(e)}")
        st.error(f"Error during batch upload: {str(e)}")
        return False


def batch_upload_files(files_list, is_image=False):
    """Upload multiple files in a single request with the 'files' parameter."""
    if not files_list:
        return None

    logging.info(
        f"[batch_upload_files] Preparing batch upload for {len(files_list)} files"
    )

    # Prepare the files for the multipart/form-data request
    files_data = []
    for file in files_list:
        file_content = file.read()
        files_data.append(("files", (file.name, file_content, file.type)))
        file.seek(0)  # Reset file pointer

    # Add additional form data
    form_data = {
        "username": st.session_state.username,
        "is_image": str(is_image).lower(),
    }

    # Send the request with proper headers
    try:
        response = requests.post(
            f"{API_URL}/file/upload", files=files_data, data=form_data
        )
        logging.info(f"[batch_upload_files] Response status: {response.status_code}")
        return response
    except Exception as e:
        logging.error(f"[batch_upload_files] Error: {str(e)}")
        return None


def process_single_file(uploaded_file_obj, is_image):
    """Process a single file and update session state."""
    logging.info(f"Processing file: {uploaded_file_obj.name}")
    files = {"file": uploaded_file_obj}  # Use the object from the list
    data = {
        "is_image": str(is_image),
        "username": st.session_state.username,
    }
    upload_response = requests.post(f"{API_URL}/file/upload", files=files, data=data)

    if upload_response.status_code == 200:
        upload_result = upload_response.json()
        file_id = upload_result["file_id"]
        file_name = uploaded_file_obj.name  # Use uploaded_file_obj here
        session_id = upload_result.get("session_id")

        # Store session ID and reset file_ids for new session
        if session_id:
            logging.info(
                f"New single-file upload session created with ID: {session_id}"
            )
            # Reset file_ids list when new files are uploaded - key for session management
            st.session_state.file_ids = []
            st.session_state.current_session_id = session_id

        # Store in session state
        st.session_state.file_id = file_id
        st.session_state.file_uploaded = True
        # Single file upload so set multi_file_mode to False
        st.session_state.multi_file_mode = False

        # Add to uploaded files list for multi-file chat
        # The st.session_state.uploaded_files dict might be legacy, ensure it exists or handle gracefully
        if "uploaded_files" not in st.session_state or not isinstance(
            st.session_state.uploaded_files, dict
        ):
            st.session_state.uploaded_files = {}
        st.session_state.uploaded_files[file_id] = {
            "name": file_name,
            "type": st.session_state.file_type,
            "session_id": session_id,  # Track which session this file belongs to
        }
        st.session_state.file_names[file_id] = file_name

        # Add to file_ids list for current session
        st.session_state.file_ids.append(file_id)
        logging.info(f"Added file_id {file_id} to current session for chat.")

        # Update processed_file_map
        if "processed_file_map" not in st.session_state:
            st.session_state.processed_file_map = {}
        st.session_state.processed_file_map[file_name] = file_id

        if upload_result.get("status") == "success":
            if st.session_state.file_type == "Database":
                st.success(
                    "Database processed successfully. You can now chat with its contents."
                )
            else:
                st.success(f"{uploaded_file_obj.name} processed successfully.")
        elif upload_result.get("status") == "partial":
            st.warning(f"{uploaded_file_obj.name}: {upload_result['message']}")
        else:
            st.info(upload_result["message"])

        if is_image:
            st.session_state.uploaded_image = uploaded_file_obj

        return True  # Successfully processed a file
    else:
        st.error(
            f"File upload failed for {uploaded_file_obj.name}: {upload_response.text}"
        )
        return False


def handle_file_upload():
    """Handle file upload UI and logic."""
    logging.info("========== ENTERING handle_file_upload ===========")
    if not st.session_state.username:
        st.warning("Please enter a username before uploading files")
        return

    is_image = st.session_state.file_type == "Image"
    file_types = {
        "Image": ["jpg", "png"],
        "CSV/Excel": ["xlsx", "xls", "csv"],
        "Database": ["db", "sqlite"],
        "PDF": ["pdf"],
        "Text": ["txt", "doc", "docx"],
        "URL": [],
    }[st.session_state.file_type]

    if st.session_state.file_type == "Database":
        st.info(
            "Upload SQLite database files (.db or .sqlite) to chat with their contents."
        )

    if st.session_state.file_type == "URL":
        st.info(
            "Enter one or more URLs separated by commas to chat with their contents."
        )
        url_input = st.text_area("Enter URLs (comma-separated for multiple URLs)")
        if url_input and st.button("Process URLs"):
            handle_url_processing(url_input)
    else:
        if "uploaded_files_list" not in st.session_state:
            st.session_state.uploaded_files_list = []

        _ = st.file_uploader(
            f"Select or Add {st.session_state.file_type} file(s)",
            type=file_types,
            accept_multiple_files=True,
            key="multi_file_uploader_static",
            on_change=handle_file_uploader_change,
        )

        display_uploaded_files_list()
        uploaded_files = st.session_state.uploaded_files_list

        if uploaded_files and len(uploaded_files) > 0:
            logging.info(
                f"[handle_file_upload] Files selected: {len(uploaded_files)} - {[f.name for f in uploaded_files]}"
            )
            file_count = len(uploaded_files)
            upload_button_label = (
                f"Upload and Process ({file_count}) "
                f"Selected File{'s' if file_count > 1 else ''}"
            )
            logging.info(
                f"[handle_file_upload] Showing upload button with label: {upload_button_label}"
            )

            if st.button(upload_button_label):
                logging.info(
                    f"[handle_file_upload] UPLOAD BUTTON CLICKED - Processing {len(uploaded_files)} files"
                )
                with st.spinner(
                    f"Uploading and processing {len(uploaded_files)} files in parallel..."
                ):
                    # Call batch_upload_files to send all files in one request
                    upload_response = batch_upload_files(uploaded_files, is_image)

                    if upload_response and upload_response.status_code == 200:
                        try:
                            result = upload_response.json()
                            logging.info(
                                f"Batch upload successful with status code: {upload_response.status_code}"
                            )

                            file_ids = result.get("file_ids", [])
                            filenames = result.get("original_filenames", [])

                            if file_ids:
                                logging.info(
                                    f"Received {len(file_ids)} file IDs from batch upload"
                                )
                                # Fix: Set multi_file_mode based on actual number of files
                                st.session_state.multi_file_mode = len(file_ids) > 1

                                if "processed_file_map" not in st.session_state:
                                    st.session_state.processed_file_map = {}
                                if "file_ids" not in st.session_state:
                                    st.session_state.file_ids = []
                                if "file_names" not in st.session_state:
                                    st.session_state.file_names = {}

                                # Reset file_ids for new session to ensure clean state
                                st.session_state.file_ids = []

                                for i, file_id in enumerate(file_ids):
                                    filename = (
                                        filenames[i]
                                        if i < len(filenames)
                                        else uploaded_files[i].name
                                    )
                                    st.session_state.processed_file_map[
                                        filename
                                    ] = file_id
                                    st.session_state.file_ids.append(file_id)
                                    st.session_state.file_names[file_id] = filename

                                # Set file_id for single file compatibility
                                st.session_state.file_id = (
                                    file_ids[0] if file_ids else None
                                )
                                st.session_state.file_uploaded = True

                                # Store session_id if provided
                                session_id = result.get("session_id")
                                if session_id:
                                    st.session_state.current_session_id = session_id
                                    logging.info(f"Set session_id: {session_id}")

                                logging.info(
                                    f"Final session state: multi_file_mode={st.session_state.multi_file_mode}, "
                                    f"file_id={st.session_state.file_id}, file_ids={st.session_state.file_ids}"
                                )

                                st.success(
                                    f"{len(file_ids)} files processed successfully!"
                                )
                            else:
                                st.error("No file IDs returned from the server.")
                        except Exception as e:
                            logging.error(
                                f"Error processing batch upload response: {str(e)}"
                            )
                            st.error(f"Error processing server response: {str(e)}")
                    else:
                        error_msg = (
                            "Unknown error"
                            if not upload_response
                            else upload_response.text
                        )
                        logging.error(f"Batch upload failed: {error_msg}")
                        st.error(f"Failed to upload files: {error_msg}")

    # Display image in a dedicated section if it's an image file
    if is_image and st.session_state.uploaded_image is not None:
        st.markdown('<div class="file-info">', unsafe_allow_html=True)
        st.subheader("Uploaded Image:")
        img = Image.open(st.session_state.uploaded_image)
        st.image(img, width=400)
        st.markdown("</div>", unsafe_allow_html=True)


def display_chat_interface():
    # Check if we have files to chat with (either single file or multiple files)
    has_files = (st.session_state.file_uploaded and st.session_state.file_id) or (
        st.session_state.multi_file_mode and len(st.session_state.file_ids) > 0
    )

    if has_files:
        # Display current file info
        file_name = st.session_state.file_names.get(
            st.session_state.file_id, st.session_state.file_id
        )
        st.markdown(
            f"<small>File: {file_name} ({st.session_state.file_type})</small>",
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

                # Log current state before constructing payload
                logging.info(
                    "Preparing chat payload. Current state: "
                    f"multi_file_mode={st.session_state.get('multi_file_mode')}, "
                    f"file_id={st.session_state.get('file_id')}, "
                    f"file_ids={st.session_state.get('file_ids')}"
                )

                # Prepare chat payload based on mode (single or multi-file)
                chat_payload = {
                    "text": previous_messages,  # This will include history and current message
                    "model_choice": st.session_state.model_choice,
                    "user_id": st.session_state.username,
                    "generate_visualization": st.session_state.generate_visualization,
                    "session_id": st.session_state.current_session_id,  # Include current session ID for isolation
                }

                # Add temperature parameter if set
                if st.session_state.temperature is not None:
                    chat_payload["temperature"] = st.session_state.temperature

                if st.session_state.multi_file_mode and st.session_state.file_ids:
                    chat_payload["file_ids"] = st.session_state.file_ids
                    logging.info(
                        f"Multi-file mode. Sending all file_ids: {st.session_state.file_ids}"
                    )
                elif st.session_state.file_id:
                    chat_payload["file_id"] = st.session_state.file_id
                    logging.info(
                        f"Single-file mode. Sending file_id: {st.session_state.file_id}"
                    )
                else:
                    st.error("Error: No file context available for chat.")
                    # Potentially skip the API call or handle as an error state
                    return
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


def initialize_file_state():
    """Initialize session state variables related to file handling."""
    if "file_uploaded" not in st.session_state:
        st.session_state.file_uploaded = False
    if "file_id" not in st.session_state:
        st.session_state.file_id = None
    if "file_ids" not in st.session_state:
        st.session_state.file_ids = []
    if "multi_file_mode" not in st.session_state:
        st.session_state.multi_file_mode = False
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {}
    # Dictionary to store file_id -> filename mapping
    if "file_names" not in st.session_state:
        st.session_state.file_names = {}  # Maps file_id to original filename
    # Track the current upload session ID
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None

    # For managing the list of files selected by the user via uploader
    if "uploaded_files_list" not in st.session_state:
        st.session_state.uploaded_files_list = (
            []
        )  # List of UploadedFile objects for the UI file list

    # For mapping processed file names (from uploaded_files_list) to their backend file_ids
    if "processed_file_map" not in st.session_state:
        st.session_state.processed_file_map = {}  # Maps original_filename -> file_id

    # Ensure the old 'uploaded_files' dict (file_id -> {name, type}) is initialized if used
    if "uploaded_files" not in st.session_state or not isinstance(
        st.session_state.uploaded_files, dict
    ):
        st.session_state.uploaded_files = (
            {}
        )  # This stores processed file info by file_id


def initialize_messages_state():
    """Initialize chat message related state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def initialize_models_state():
    """Initialize models and retrieve available models from API."""
    if "available_models" not in st.session_state:
        response = requests.get(f"{API_URL}/available-models")
        if response.status_code == 200:
            response_data = response.json()
            st.session_state.available_models = response_data["models"]
            # Safely handle model_types which may not exist in older API versions
            if "model_types" in response_data:
                st.session_state.model_types = response_data["model_types"]
            else:
                setup_default_model_types()
        else:
            setup_fallback_models()

    if "model_choice" not in st.session_state:
        st.session_state.model_choice = "gpt_4o_mini"
    if "model_initialized" not in st.session_state:
        st.session_state.model_initialized = False


def setup_default_model_types():
    """Set up default model types based on model name patterns."""
    # Default categorization if API doesn't provide it
    # Identify image models by name pattern
    image_models = [
        m
        for m in st.session_state.available_models
        if "dall-e" in m.lower() or "imagen" in m.lower()
    ]
    text_models = [
        m for m in st.session_state.available_models if m not in image_models
    ]

    st.session_state.model_types = {
        "text": text_models,
        "image": image_models,
    }


def setup_fallback_models():
    """Set up fallback models if API call fails."""
    st.session_state.available_models = ["gpt_4o_mini", "gemini-pro"]
    # Fallback model types if API call fails
    st.session_state.model_types = {
        "text": ["gpt_4o_mini", "gemini-pro"],
        "image": [
            "dall-e-3",
            "imagen-3.0-generate-002",
        ],  # Assuming dall-e and imagen are your image models
    }


def initialize_ui_state():
    """Initialize UI related state variables."""
    if "file_type" not in st.session_state:
        st.session_state.file_type = "PDF"
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    if "nav_option" not in st.session_state:
        st.session_state.nav_option = "Chat"
    # Visualization is now automatically detected by the backend
    # The generate_visualization flag is still kept in session state for API compatibility
    if "generate_visualization" not in st.session_state:
        st.session_state.generate_visualization = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    # Initialize temperature parameter
    if "temperature" not in st.session_state:
        st.session_state.temperature = None  # None means use model defaults


def initialize_session_state():
    """Initialize all session state variables by calling specialized functions."""
    initialize_file_state()
    initialize_messages_state()
    initialize_models_state()
    initialize_ui_state()


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
            "is_url": "true",  # Maintained for now
            "urls": url_input,
        }

        upload_response = requests.post(f"{API_URL}/file/upload", data=data)

        if upload_response.status_code == 200:
            upload_result = upload_response.json()
            logging.info(f"Received upload_result: {upload_result}")
            if "multi_file_mode" in upload_result:
                logging.info(
                    f"upload_result['multi_file_mode'] value: {upload_result['multi_file_mode']},\n"
                    f"type: {type(upload_result['multi_file_mode'])}"
                )
            else:
                logging.info("upload_result does not contain 'multi_file_mode'")
            if "file_ids" in upload_result:
                logging.info(
                    f"upload_result['file_ids'] value: {upload_result['file_ids']},\n"
                    f"type: {type(upload_result['file_ids'])}"
                )
            else:
                logging.info("upload_result does not contain 'file_ids'")

            # Prioritize multi_file_mode if explicitly set to True by the backend
            if upload_result.get("multi_file_mode") is True:  # Explicit check for True
                if (
                    isinstance(upload_result.get("file_ids"), list)
                    and upload_result["file_ids"]
                ):
                    # Multi-URL processing successful
                    st.session_state.multi_file_mode = True
                    st.session_state.file_ids = list(upload_result["file_ids"])
                    st.session_state.file_id = st.session_state.file_ids[0]
                    logging.info(
                        f"Multi-URL mode (explicit). st.session_state.file_ids set to: {st.session_state.file_ids}"
                    )
                    st.success(
                        f"Processed {len(st.session_state.file_ids)} URLs successfully. Ready for multi-document chat."
                    )
                else:
                    # multi_file_mode was true, but file_ids was missing or empty
                    logging.warning(
                        "multi_file_mode is True but file_ids is invalid. Falling back."
                    )
                    if "file_id" in upload_result and upload_result["file_id"]:
                        st.session_state.multi_file_mode = False
                        st.session_state.file_id = upload_result["file_id"]
                        st.session_state.file_ids = [upload_result["file_id"]]
                        logging.info(
                            f"Single-URL mode (fallback from multi).\n"
                            f"st.session_state.file_ids set to: {st.session_state.file_ids}"
                        )
                        st.success("URL processed successfully (fallback).")
                    else:
                        st.error(
                            "Error processing URLs: multi_file_mode True but no valid file_ids or file_id."
                        )
                        return
            elif (
                "file_id" in upload_result and upload_result["file_id"]
            ):  # Check this only if multi_file_mode was not explicitly True
                # Single URL processing successful
                st.session_state.multi_file_mode = False
                st.session_state.file_id = upload_result["file_id"]
                st.session_state.file_ids = [upload_result["file_id"]]
                logging.info(
                    f"Single-URL mode (explicit). st.session_state.file_ids set to: {st.session_state.file_ids}"
                )
                st.success("URL processed successfully and ready for chat.")
            else:
                # Fallback or error in response structure
                st.error(
                    "Error processing URLs: Response format incorrect or no file IDs found."
                )
                return

            st.session_state.file_uploaded = True
            st.session_state.messages = []  # Reset chat history
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
            file_name = uploaded_file.name

            # Store in session state
            st.session_state.file_id = file_id
            st.session_state.file_uploaded = True

            # Add to uploaded files list for multi-file chat
            st.session_state.uploaded_files[file_id] = {
                "name": file_name,
                "type": st.session_state.file_type,
            }
            st.session_state.file_names[file_id] = file_name

            # Always add the file_id to our list of files for multi-file context
            if file_id not in st.session_state.file_ids:
                st.session_state.file_ids.append(file_id)
                logging.info(f"Added file {file_id} to available files list")

            if upload_result.get("status") == "success":
                if st.session_state.file_type == "Database":
                    st.success(
                        "Database processed successfully. You can now chat with its contents."
                    )
                else:
                    st.success(f"{uploaded_file.name} processed successfully.")
            elif upload_result.get("status") == "partial":
                st.warning(f"{uploaded_file.name}: {upload_result['message']}")
            else:
                st.info(upload_result["message"])

            if is_image:
                st.session_state.uploaded_image = uploaded_file
        else:
            st.error(
                f"File upload failed for {uploaded_file.name}: {upload_response.text}"
            )

        # When all files are processed, add a final success message
        if len(st.session_state.uploaded_files) > 1:
            st.success(
                f"All {len(st.session_state.uploaded_files)} files processed successfully!"
            )

        # Reset chat messages when new files are uploaded
        st.session_state.messages = []


def render_sidebar():
    """Render the sidebar components."""
    # Add New Chat button at the top
    if st.button("New Chat", key="new_chat_btn"):
        cleanup_files()
        st.rerun()

    # Don't force multi_file_mode - let it be set based on actual upload results
    # multi_file_mode is set correctly during file upload based on number of files

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
            # Allow multiple files to be selected
            uploaded_files = st.file_uploader(
                f"Choose {st.session_state.file_type} file(s)",  # Updated label
                type=file_types,
                accept_multiple_files=True,  # Enable multi-file upload
            )

            # Process uploaded files if any are selected and button is pressed
            if uploaded_files:  # Check if the list is not empty
                if st.button(
                    f"Upload and Process ({len(uploaded_files)}) Selected File(s)"
                ):  # Updated button label
                    # FIXED: Use batch processing instead of calling process_file_upload in a loop
                    # This sends all files in a single request with the 'files' parameter
                    logging.info(
                        f"Processing {len(uploaded_files)} files in batch mode"
                    )
                    with st.spinner(
                        f"Uploading and processing {len(uploaded_files)} files in parallel..."
                    ):
                        success = process_multiple_files(uploaded_files, is_image)
                        if success:
                            st.success(
                                f"{len(uploaded_files)} files processed successfully!"
                            )
                        else:
                            st.error("Error processing files. Please check the logs.")
                    st.rerun()  # Rerun to update UI after processing all files

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

    # Temperature control section
    st.markdown(
        '<div class="sidebar-header">Temperature Settings</div>', unsafe_allow_html=True
    )

    # Add help text for temperature
    st.markdown(
        "<small>Temperature controls randomness: 0.0 = focused, 1.0 = creative</small>",
        unsafe_allow_html=True,
    )

    # Temperature control with checkbox for auto mode
    use_auto_temperature = st.checkbox(
        "Use model defaults",
        value=st.session_state.temperature is None,
        help="Let the system choose optimal temperature based on model type (OpenAI: 0.5, Gemini: 0.8)",
    )

    if use_auto_temperature:
        st.session_state.temperature = None
        st.markdown(
            '<small style="color: #666;">Using automatic temperature based on model</small>',
            unsafe_allow_html=True,
        )
    else:
        # Temperature slider (0.0 to 2.0, step 0.1, default 0.7)
        temperature_value = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7
            if st.session_state.temperature is None
            else st.session_state.temperature,
            step=0.1,
            help="Higher values make output more random, lower values more focused",
        )
        st.session_state.temperature = temperature_value


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
