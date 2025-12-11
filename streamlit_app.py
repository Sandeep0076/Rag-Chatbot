import logging
import warnings

import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

from streamlit_components.custom_gpt_creator import display_custom_gpt_creator
from streamlit_components.custom_gpt_prompts import (
    CUSTOM_GPT_GENERAL_PROMPT,
    DOCUMENT_GROUNDING_PROMPT,
)
from streamlit_image_generation import handle_image_generation

# Define API URL as it's used across functions
API_URL = "http://localhost:8080"
# Preferred default text model (non-image)
DEFAULT_TEXT_MODEL = "claude-sonnet-4-5"

# Suppress warnings from google-cloud-aiplatform and vertexai
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="google.cloud.aiplatform"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="vertexai._model_garden._model_garden_models"
)
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

# Modern Neumorphic Design System - Inspired by Reference UI

custom_css = """
<style>
    /* === ROOT VARIABLES === */
    :root {
        --bg-primary: #f5f7fa;
        --bg-secondary: #ffffff;
        --bg-card: #ffffff;
        --color-primary: #64748b;
        --color-secondary: #94a3b8;
        --color-accent: #0ea5e9;
        --color-text: #475569;
        --color-text-muted: #94a3b8;
        --border-radius: 24px;
        --border-radius-lg: 32px;
        --shadow-soft: 0 4px 20px -2px rgba(148, 163, 184, 0.1), 0 8px 16px -4px rgba(148, 163, 184, 0.1);
        --shadow-card: 0 8px 32px -4px rgba(148, 163, 184, 0.15), 0 16px 24px -8px rgba(148, 163, 184, 0.1);
        --shadow-inset: inset 0 2px 8px rgba(148, 163, 184, 0.1);
        --gradient-primary: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%);
        --gradient-soft: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    }

         /* === BASE STYLES === */
     body, .stApp {
         background: var(--bg-primary) !important;
         font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
         color: var(--color-text) !important;
         margin: 0 !important;
         padding: 0 !important;
         width: 100% !important;
     }

     /* Remove any default Streamlit margins/padding */
     .main {
         padding: 0 !important;
         margin: 0 !important;
         width: 100% !important;
     }

         /* === MAIN CONTAINER === */
     .main .block-container {
         background: transparent !important;
         padding: 1rem !important;
         max-width: 100% !important;
         width: 100% !important;
     }

    /* === SIDEBAR STYLING === */
    section[data-testid="stSidebar"], .stSidebar {
        background: var(--bg-secondary) !important;
        border: none !important;
        box-shadow: var(--shadow-card) !important;
        border-radius: 0 var(--border-radius) var(--border-radius) 0 !important;
    }

    .stSidebar > div {
        background: transparent !important;
        padding: 1.5rem !important;
    }

         /* === HEADER STYLING === */
     .main-header {
         background: var(--bg-card) !important;
         border-radius: var(--border-radius) !important;
         padding: 1rem 1rem !important;
         margin: 0 0 1rem 0 !important;
         box-shadow: var(--shadow-card) !important;
         text-align: center !important;
         border: 1px solid rgba(148, 163, 184, 0.1) !important;
         width: 100% !important;
         max-width: 100% !important;
     }

     .main-header h1 {
         background: var(--gradient-primary) !important;
         -webkit-background-clip: text !important;
         background-clip: text !important;
         -webkit-text-fill-color: transparent !important;
         font-weight: 500 !important;
         font-size: 1.5rem !important;
         margin-bottom: 0.3rem !important;
         letter-spacing: -0.75px !important;
         line-height: 1.1 !important;
     }

    /* === BUTTON STYLING === */
    .stButton > button {
        background: linear-gradient(145deg, #f0f4f8, #d6e4ed) !important;
        color: var(--color-text) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: var(--border-radius) !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: 8px 8px 16px rgba(148, 163, 184, 0.15), -8px -8px 16px rgba(255, 255, 255, 0.7) !important;
        transition: all 0.3s ease !important;
        height: auto !important;
        min-height: 48px !important;
    }

    .stButton > button:hover {
        background: linear-gradient(145deg, #e2e8f0, #cbd5e0) !important;
        box-shadow: 6px 6px 12px rgba(148, 163, 184, 0.2), -6px -6px 12px rgba(255, 255, 255, 0.8) !important;
        transform: translateY(-1px) !important;
    }

    .stButton > button:active {
        background: linear-gradient(145deg, #cbd5e0, #e2e8f0) !important;
        box-shadow: inset 4px 4px 8px rgba(148, 163, 184, 0.2), inset -4px -4px 8px rgba(255, 255, 255, 0.7) !important;
        transform: translateY(0) !important;
    }

    /* === NAVIGATION BUTTONS === */
    .nav-container {
        background: var(--bg-card) !important;
        border-radius: var(--border-radius) !important;
        padding: 0.75rem !important;
        margin-bottom: 1rem !important;
        box-shadow: var(--shadow-card) !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
    }

    .nav-container button {
        background: var(--bg-card) !important;
        color: var(--color-text) !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        border-radius: var(--border-radius) !important;
        padding: 0.5rem 1.5rem !important;
        margin-right: 0.5rem !important;
        box-shadow: var(--shadow-soft) !important;
        transition: all 0.3s ease !important;
        font-weight: 500 !important;
    }

    .nav-container button:hover,
    .nav-container button[data-selected="true"] {
        background: var(--gradient-primary) !important;
        color: white !important;
        transform: translateY(-1px) !important;
    }

    /* === CARDS AND CONTAINERS === */
    .modern-card {
        background: var(--bg-card) !important;
        border-radius: var(--border-radius) !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
        box-shadow: var(--shadow-card) !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
    }

    /* === INPUT FIELDS === */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div > div,
    div[data-testid="stChatInput"] input {
        background: var(--bg-card) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: var(--border-radius) !important;
        padding: 1rem 1.5rem !important;
        color: var(--color-text) !important;
        font-size: 0.95rem !important;
        box-shadow: var(--shadow-inset) !important;
        transition: all 0.3s ease !important;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    div[data-testid="stChatInput"] input:focus {
        border-color: var(--color-accent) !important;
        box-shadow: var(--shadow-inset), 0 0 0 3px rgba(14, 165, 233, 0.1) !important;
        outline: none !important;
    }

    /* === CHAT INTERFACE === */
    .stChatInputContainer,
    div[data-testid="stChatInputContainer"] {
        background: var(--bg-card) !important;
        border-radius: var(--border-radius) !important;
        padding: 1.5rem !important;
        margin: 2rem 0 1rem 0 !important;
        box-shadow: var(--shadow-card) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
    }

    /* Style the chat input itself to be more visible and stick to bottom */
    div[data-testid="stChatInput"] {
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        background: var(--bg-card) !important;
        border-radius: var(--border-radius) !important;
        padding: 1rem !important;
        box-shadow: var(--shadow-card) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        position: relative !important;
        width: 100% !important;
        z-index: 1 !important;
    }

    div[data-testid="stChatInput"] > div {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        position: relative !important;
        overflow: hidden !important;
    }

    /* Make the actual input field more visible */
    div[data-testid="stChatInput"] input {
        background: var(--bg-card) !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        border-radius: var(--border-radius) !important;
        padding: 1rem 1.5rem 1rem 1.5rem !important;
        padding-right: 3.5rem !important; /* Leave room for send button */
        color: var(--color-text) !important;
        font-size: 1rem !important;
        box-shadow: var(--shadow-inset) !important;
        flex: 1;
        border: none !important;
    }

    /* Position the send button container inside the chat input */
    div[data-testid="stChatInput"] button {
        position: absolute !important;
        right: 0.5rem !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        background: transparent !important;
        border: none !important;
        padding: 0.25rem !important;
        border-radius: 50% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        z-index: 10 !important;
    }

    /* Chat input focus state */
    div[data-testid="stChatInput"] input:focus {
        border-color: var(--color-accent) !important;
        box-shadow: var(--shadow-inset), 0 0 0 3px rgba(14, 165, 233, 0.1) !important;
        outline: none !important;
    }

    /* === CHAT MESSAGES === */
    div[data-testid="stChatMessage"] {
        margin: 1rem 0 !important;
    }

    div[data-testid="stChatMessage"] > div {
        background: var(--bg-card) !important;
        border-radius: var(--border-radius) !important;
        padding: 1.5rem !important;
        box-shadow: var(--shadow-card) !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
        margin: 0 !important;
    }

    /* User messages - keep white/light */
    div[data-testid="stChatMessage"]:has([data-testid="user-message"]) > div,
    div[data-testid="stChatMessage"]:nth-child(odd) > div {
        background: var(--bg-card) !important;
        border-left: 4px solid #cbd5e0 !important;
    }

    /* Assistant messages - light gray, darker than user messages */
    div[data-testid="stChatMessage"]:has([data-testid="assistant-message"]) > div,
    div[data-testid="stChatMessage"]:nth-child(even) > div {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
        border-left: 4px solid #94a3b8 !important;
    }

    /* === CHAT AVATARS/ICONS === */
    /* Make chat avatars darker and more visible */
    div[data-testid="stChatMessage"] img,
    div[data-testid="stChatMessage"] svg,
    div[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"],
    div[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
        opacity: 1 !important;
        filter: brightness(0.6) contrast(1.5) !important;
        background: rgba(71, 85, 105, 0.8) !important;
        border-radius: 50% !important;
        padding: 8px !important;
        color: white !important;
    }

    /* Target the avatar container */
    div[data-testid="stChatMessage"] > div:first-child,
    .stChatMessage [data-testid="chatAvatarIcon-user"],
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        background: rgba(71, 85, 105, 0.8) !important;
        border-radius: 50% !important;
        min-width: 40px !important;
        min-height: 40px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    /* Make the icons inside avatars darker */
    div[data-testid="stChatMessage"] svg path {
        fill: white !important;
        stroke: white !important;
    }

    /* === FILE UPLOAD AREA === */
    .stFileUploader {
        background: var(--bg-card) !important;
        border: 2px dashed rgba(148, 163, 184, 0.3) !important;
        border-radius: var(--border-radius) !important;
        padding: 2rem !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
        box-shadow: var(--shadow-soft) !important;
    }

    .stFileUploader:hover {
        border-color: var(--color-accent) !important;
        background: linear-gradient(135deg, #eff6ff 0%, #f8fafc 100%) !important;
    }

    [data-testid="stFileUploadDropzone"] {
        background: transparent !important;
        border: none !important;
    }

         /* === SIDEBAR STYLING CONTINUED === */

    .sidebar-header {
        color: var(--color-text) !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid rgba(148, 163, 184, 0.1) !important;
    }

    /* === ALERTS AND NOTIFICATIONS === */
    .stAlert,
    .stSuccess,
    .stError,
    .stWarning,
    .stInfo {
        border-radius: var(--border-radius) !important;
        border: none !important;
        box-shadow: var(--shadow-soft) !important;
        padding: 1rem 1.5rem !important;
    }

    /* === EXPANDER === */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border-radius: var(--border-radius) !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
        box-shadow: var(--shadow-soft) !important;
    }

    .streamlit-expanderContent {
        background: var(--bg-card) !important;
        border-radius: 0 0 var(--border-radius) var(--border-radius) !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
        border-top: none !important;
        box-shadow: var(--shadow-soft) !important;
    }

         /* === SELECTBOX AND DROPDOWN === */
     .stSelectbox > div > div {
         background: transparent !important;
         border: none !important;
         box-shadow: none !important;
     }

     /* Clean, minimal selectbox - just text but keep functionality */
     .stSelectbox div[data-baseweb="select"] {
         background: transparent !important;
         border: none !important;
         box-shadow: none !important;
         min-height: auto !important;
         display: flex !important;
         align-items: center !important;
         cursor: pointer !important;
     }

     /* Control container - no styling, center text, keep clickable */
     .stSelectbox div[data-baseweb="select"] > div {
         background: transparent !important;
         border: none !important;
         padding: 4px 0 !important;
         min-height: auto !important;
         display: flex !important;
         align-items: center !important;
         cursor: pointer !important;
         width: 100% !important;
     }

     /* Selected value styling - clean text, centered, clickable */
     .stSelectbox div[data-baseweb="select"] span {
         color: var(--color-text) !important;
         font-weight: 500 !important;
         font-size: 1rem !important;
         display: flex !important;
         align-items: center !important;
         cursor: pointer !important;
         pointer-events: all !important;
     }

     /* Dropdown arrow - minimal but visible and clickable */
     .stSelectbox svg {
         color: var(--color-secondary) !important;
         width: 16px !important;
         height: 16px !important;
         margin-left: 8px !important;
         cursor: pointer !important;
         pointer-events: all !important;
     }

     /* Ensure the entire selectbox area is clickable */
     .stSelectbox {
         cursor: pointer !important;
     }

     .stSelectbox > div {
         cursor: pointer !important;
     }

     .stSelectbox > div > div {
         cursor: pointer !important;
     }

     /* Dropdown menu - only style the popup */
     .stSelectbox [data-baseweb="popover"] {
         background: var(--bg-card) !important;
         border-radius: 12px !important;
         box-shadow: var(--shadow-card) !important;
         border: 1px solid rgba(148, 163, 184, 0.15) !important;
         margin-top: 4px !important;
     }

     /* Dropdown options */
     .stSelectbox [data-baseweb="popover"] > div {
         background: var(--bg-card) !important;
         border-radius: 12px !important;
         padding: 8px !important;
     }

     .stSelectbox [role="option"] {
         background: transparent !important;
         color: var(--color-text) !important;
         padding: 0.75rem 1rem !important;
         border-radius: 8px !important;
         margin: 2px 0 !important;
         transition: all 0.2s ease !important;
         font-size: 0.95rem !important;
         cursor: pointer !important;
     }

     .stSelectbox [role="option"]:hover {
         background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%) !important;
         color: var(--color-accent) !important;
     }

     /* Remove visual styling but keep functionality */
     .stSelectbox > div > div > div,
     .stSelectbox [data-baseweb="select"] > div > div,
     .stSelectbox [data-baseweb="select"] > div,
     .stSelectbox div {
         background: transparent !important;
         border: none !important;
         box-shadow: none !important;
     }

     /* Only hide truly empty divs, not functional elements */
     .stSelectbox > div > div > div:empty {
         display: none !important;
     }

    /* === SLIDER === */
    .stSlider > div > div > div > div {
        background: var(--color-accent) !important;
    }

    /* === CHECKBOX === */
    .stCheckbox > label > div {
        background: var(--bg-card) !important;
        border-radius: 6px !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        box-shadow: var(--shadow-inset) !important;
    }

    /* === REMOVE UNWANTED BACKGROUNDS === */
    div.main > div > div > div > div,
    div.block-container,
    div[data-testid="stVerticalBlock"] > div,
    div.element-container,
    div.stMarkdown,
    div.stSpinner {
        background: transparent !important;
    }

    /* === TYPOGRAPHY === */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--color-text) !important;
        font-weight: 600 !important;
    }

    .stMarkdown p {
        color: var(--color-text) !important;
        line-height: 1.6 !important;
    }

    /* === CUSTOM UTILITIES === */
    .glass-effect {
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        background: rgba(255, 255, 255, 0.9) !important;
    }

    .soft-shadow {
        box-shadow: var(--shadow-soft) !important;
    }

    .card-shadow {
        box-shadow: var(--shadow-card) !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


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


def reset_session(preserve_username: bool = True):
    """Reset session while optionally preserving username."""
    preserved_username = st.session_state.get("username") if preserve_username else ""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()
    if preserve_username and preserved_username:
        st.session_state.username = preserved_username
        # Re-sync query params after reset
        st.query_params["username"] = preserved_username


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

            # CRITICAL FIX: Extract and set session_id for URL uploads
            session_id = upload_result.get("session_id")
            logging.info(f"Extracted session_id from URL upload result: {session_id}")
            if session_id:
                st.session_state.current_session_id = session_id
                logging.info(
                    f"Successfully set session_id in session state: {session_id}"
                )
            else:
                logging.error("No session_id found in URL upload response!")
                st.error(
                    "URL processing succeeded but no session ID was returned. Please try again."
                )

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
            # Removed st.rerun() to avoid RerunData error
            # Session state changes trigger automatic rerun
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
                    }
                    st.session_state.file_names[file_id] = filename

                    # Add to file_ids list for the current session's files
                    st.session_state.file_ids.append(file_id)
                    logging.info(
                        f"Added file {filename} (ID: {file_id}) to current session - Status: {status}"
                    )
            else:
                # Handle single file response (multi_file_mode=False)
                file_id = upload_result.get("file_id")
                filename = upload_result.get("original_filename")

                logging.info(f"Received single-file response for file: {filename}")

                # CRITICAL FIX: Extract and set session_id for single file uploads
                session_id = upload_result.get("session_id")
                logging.info(
                    f"Extracted session_id from single file upload result: {session_id}"
                )
                if session_id:
                    st.session_state.current_session_id = session_id
                    logging.info(
                        f"Successfully set session_id in session state: {session_id}"
                    )
                else:
                    logging.error("No session_id found in single file upload response!")
                    st.error(
                        "Upload succeeded but no session ID was returned. Please try again."
                    )

                # Process file info
                if file_id:
                    # Reset file_ids for new session
                    st.session_state.file_ids = []

                    # Set session state
                    st.session_state.file_id = file_id
                    st.session_state.file_uploaded = True
                    st.session_state.multi_file_mode = False

                    if "processed_file_map" not in st.session_state:
                        st.session_state.processed_file_map = {}
                    st.session_state.processed_file_map[filename] = file_id

                    if "uploaded_files" not in st.session_state or not isinstance(
                        st.session_state.uploaded_files, dict
                    ):
                        st.session_state.uploaded_files = {}

                    st.session_state.uploaded_files[file_id] = {
                        "name": filename,
                    }
                    st.session_state.file_names[file_id] = filename

                    # Add to file_ids list for current session
                    st.session_state.file_ids.append(file_id)
                    logging.info(
                        f"Added file {filename} (ID: {file_id}) to current session (single mode)"
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


def _get_file_types_for_upload():
    """Get allowed file types for mixed uploads."""
    return [
        "pdf",
        "txt",
        "doc",
        "docx",
        "csv",
        "xls",
        "xlsx",
        "db",
        "sqlite",
        "jpg",
        "jpeg",
        "png",
        "gif",
        "bmp",
        "webp",
    ]


def _display_file_type_info():
    """Display information for supported files."""
    st.info(
        "Supported: PDF, TXT/DOC/DOCX, CSV/XLS/XLSX, DB/SQLITE, and common images (JPG/PNG/GIF/BMP/WEBP)."
    )


def _handle_url_input():
    """Handle URL input and processing."""
    st.info("Enter one or more URLs separated by commas to chat with their contents.")
    url_input = st.text_area("Enter URLs (comma-separated for multiple URLs)")
    if url_input and st.button("Process URLs"):
        handle_url_processing(url_input)


def _handle_existing_file_ids_input():
    """Handle existing file IDs input and processing."""
    st.info("Enter existing file IDs to chat with files that already have embeddings.")
    existing_file_ids_input = st.text_area(
        "Enter File IDs (comma or newline separated for multiple files)",
        placeholder="file-id-1, file-id-2\nfile-id-3",
    )
    if existing_file_ids_input and st.button("Process Existing File IDs"):
        handle_existing_file_ids_processing(existing_file_ids_input)


def _setup_file_uploader(file_types):
    """Setup file uploader and get uploaded files."""
    if "uploaded_files_list" not in st.session_state:
        st.session_state.uploaded_files_list = []

    st.markdown("### Upload New Files")
    _ = st.file_uploader(
        "Select files",
        type=file_types,
        accept_multiple_files=True,
        key="multi_file_uploader_static",
        on_change=handle_file_uploader_change,
    )

    display_uploaded_files_list()
    return st.session_state.uploaded_files_list


def _get_existing_file_ids_input():
    """Get existing file IDs input from user."""
    st.markdown("### Or Use Existing File IDs")
    return st.text_area(
        "Enter existing File IDs (optional, can combine with new uploads)",
        placeholder="file-id-1, file-id-2",
    )


def _calculate_process_items(uploaded_files, existing_file_ids_input, urls_input=None):
    """Calculate what items need to be processed."""
    has_new_files = uploaded_files and len(uploaded_files) > 0
    has_existing_file_ids = existing_file_ids_input and existing_file_ids_input.strip()
    has_urls = bool(urls_input and urls_input.strip())

    process_items = []
    if has_new_files:
        process_items.append(
            f"{len(uploaded_files)} new file{'s' if len(uploaded_files) > 1 else ''}"
        )
    if has_existing_file_ids:
        file_ids_list = existing_file_ids_input.replace("\n", ",").split(",")
        existing_count = len([fid.strip() for fid in file_ids_list if fid.strip()])
        process_items.append(
            f"{existing_count} existing file ID{'s' if existing_count > 1 else ''}"
        )
    if has_urls:
        url_count = len(
            [
                url.strip()
                for url in urls_input.replace("\n", ",").split(",")
                if url.strip()
            ]
        )
        process_items.append(f"{url_count} URL{'s' if url_count > 1 else ''}")

    return has_new_files, has_existing_file_ids, has_urls, process_items


def _update_session_state_with_files(file_ids, filenames):
    """Update session state with processed file information."""
    logging.info(f"Received {len(file_ids)} file IDs from enhanced upload")

    # Set multi_file_mode based on actual number of files
    st.session_state.multi_file_mode = len(file_ids) > 1

    # Initialize session state dictionaries if needed
    if "processed_file_map" not in st.session_state:
        st.session_state.processed_file_map = {}
    if "file_ids" not in st.session_state:
        st.session_state.file_ids = []
    if "file_names" not in st.session_state:
        st.session_state.file_names = {}

    # Reset file_ids for new session to ensure clean state
    st.session_state.file_ids = []

    # Process each file ID and filename
    for i, file_id in enumerate(file_ids):
        filename = filenames[i] if i < len(filenames) else f"file_{file_id}"
        st.session_state.processed_file_map[filename] = file_id
        st.session_state.file_ids.append(file_id)
        st.session_state.file_names[file_id] = filename

    # Set file_id for single file compatibility
    st.session_state.file_id = file_ids[0] if file_ids else None
    st.session_state.file_uploaded = True


def _process_upload_response(upload_response):
    """Process the upload response and update session state."""
    if upload_response and upload_response.status_code == 200:
        try:
            result = upload_response.json()
            logging.info(
                f"Enhanced upload successful with status code: {upload_response.status_code}"
            )

            file_ids = result.get("file_ids", [])
            filenames = result.get("original_filenames", [])

            if file_ids:
                _update_session_state_with_files(file_ids, filenames)

                # Store session_id if provided
                session_id = result.get("session_id")
                if session_id:
                    st.session_state.current_session_id = session_id
                    logging.info(
                        f"Successfully set session_id in session state: {session_id}"
                    )

                logging.info(
                    f"Final session state: multi_file_mode={st.session_state.multi_file_mode}, "
                    f"file_id={st.session_state.file_id}, file_ids={st.session_state.file_ids}"
                )

                st.success(f"{len(file_ids)} files processed successfully!")
                return True
            else:
                st.error("No file IDs returned from the server.")
                return False
        except Exception as e:
            logging.error(f"Error processing enhanced upload response: {str(e)}")
            st.error(f"Error processing server response: {str(e)}")
            return False
    else:
        error_msg = _parse_error_response(upload_response)
        logging.error(f"Enhanced upload failed: {error_msg}")
        st.error(f"Failed to process: {error_msg}")
        return False


def _handle_file_processing(
    uploaded_files, existing_file_ids_input, is_image, urls_input=None
):
    """Handle the file processing logic."""
    (
        has_new_files,
        has_existing_file_ids,
        has_urls,
        process_items,
    ) = _calculate_process_items(uploaded_files, existing_file_ids_input, urls_input)

    if has_new_files or has_existing_file_ids or has_urls:
        button_label = "Process " + " and ".join(process_items)

        if st.button(button_label):
            logging.info(
                "[handle_file_upload] PROCESS BUTTON CLICKED - Processing mixed content"
            )
            with st.spinner("Processing files and file IDs..."):
                upload_response = enhanced_batch_upload(
                    uploaded_files, existing_file_ids_input, is_image, urls_input
                )
                _process_upload_response(upload_response)


def _display_uploaded_image():
    """Display uploaded image if present (optional)."""
    if st.session_state.uploaded_image is not None:
        st.markdown('<div class="file-info">', unsafe_allow_html=True)
        st.subheader("Uploaded Image:")
        img = Image.open(st.session_state.uploaded_image)
        st.image(img, width=400)
        st.markdown("</div>", unsafe_allow_html=True)


def handle_file_upload():
    """Handle file upload UI and logic."""
    logging.info("========== ENTERING handle_file_upload ===========")

    if not st.session_state.username:
        st.warning("Please enter a username before uploading files")
        return

    # Mixed uploads allowed; image preview shown only when a single image is uploaded for display
    is_image = False
    file_types = _get_file_types_for_upload()

    _display_file_type_info()

    uploaded_files = _setup_file_uploader(file_types)
    existing_file_ids_input = _get_existing_file_ids_input()
    _handle_file_processing(uploaded_files, existing_file_ids_input, is_image)

    _display_uploaded_image()


def _generate_initial_greeting():
    """Generate initial greeting for Custom GPT based on name and system prompt."""
    gpt_name = st.session_state.get("custom_gpt_name", "Assistant")
    system_prompt = st.session_state.get("custom_gpt_system_prompt", "")

    # Keywords that indicate a proactive GPT
    proactive_keywords = ["tutor", "teacher", "coach", "trainer", "instructor", "guide"]

    # Check if GPT should be proactive
    is_proactive = any(
        keyword in gpt_name.lower() or keyword in system_prompt.lower()
        for keyword in proactive_keywords
    )

    if is_proactive:
        # Proactive greeting - takes initiative
        return (
            f"Hello! I'm {gpt_name}. I'm here to help you learn and grow.\n"
            "What topic would you like to explore today?"
        )
    else:
        # Reactive greeting - waits for user input
        return f"Hello! I'm {gpt_name}. How can I help you today?"


def _display_conversation_starters(conversation_starters):
    """Display conversation starter buttons and handle clicks."""
    st.markdown("---")
    st.markdown("### 💬 Conversation Starters")
    st.markdown(
        "<p style='color: var(--color-text-muted); margin-bottom: 1rem;'>"
        "Choose a question to get started:</p>",
        unsafe_allow_html=True,
    )

    # Create three columns for the buttons
    col1, col2, col3 = st.columns(3)

    # Button 1
    with col1:
        if st.button(
            conversation_starters[0],
            key="conv_starter_0",
            use_container_width=True,
            help="Click to ask this question",
        ):
            st.session_state.messages.append(
                {"role": "user", "content": conversation_starters[0]}
            )
            _process_conversation_starter(conversation_starters[0])
            st.rerun()

    # Button 2
    with col2:
        if st.button(
            conversation_starters[1],
            key="conv_starter_1",
            use_container_width=True,
            help="Click to ask this question",
        ):
            st.session_state.messages.append(
                {"role": "user", "content": conversation_starters[1]}
            )
            _process_conversation_starter(conversation_starters[1])
            st.rerun()

    # Button 3
    with col3:
        if st.button(
            conversation_starters[2],
            key="conv_starter_2",
            use_container_width=True,
            help="Click to ask this question",
        ):
            st.session_state.messages.append(
                {"role": "user", "content": conversation_starters[2]}
            )
            _process_conversation_starter(conversation_starters[2])
            st.rerun()


def _process_conversation_starter(question):
    """Process a conversation starter question by sending it to the chat API."""
    with st.spinner("Processing your request..."):
        previous_messages = [
            msg["content"]
            for msg in st.session_state.messages[-5:]
            if msg["role"] == "user"
        ]

        if not st.session_state.current_session_id:
            st.error("Error: No session ID available. Please upload the file again.")
            return

        chat_payload = _get_chat_payload(previous_messages)
        logging.info(f"Sending chat payload for conversation starter: {chat_payload}")

        # Determine endpoint based on custom_gpt_mode and file presence
        is_custom_gpt = st.session_state.get("custom_gpt_mode", False)

        # For custom GPT, check custom_gpt_document_ids; for regular chat, check file_ids
        if is_custom_gpt:
            has_files = bool(st.session_state.get("custom_gpt_document_ids"))
        else:
            has_files = bool(st.session_state.get("file_ids"))

        if is_custom_gpt and not has_files:
            # Custom GPT without files - use Anthropic endpoint
            endpoint = f"{API_URL}/chat/anthropic"
            logging.info("Using /chat/anthropic endpoint (Custom GPT without files)")
        else:
            # Custom GPT with files OR regular file chat - use file/chat endpoint
            endpoint = f"{API_URL}/file/chat"
            logging.info(
                f"Using /file/chat endpoint (Custom GPT: {is_custom_gpt}, Has files: {has_files})"
            )

        chat_response = requests.post(endpoint, json=chat_payload)
        _handle_chat_response(chat_response)


def display_chat_interface():
    # Check if we're in custom GPT mode or regular file chat mode
    is_custom_gpt = st.session_state.get("custom_gpt_mode", False)

    # In custom GPT mode, we may or may not have files
    # In regular mode, we need files
    has_files = (st.session_state.file_uploaded and st.session_state.file_id) or (
        st.session_state.multi_file_mode and len(st.session_state.file_ids) > 0
    )

    # Custom GPT can work without files
    can_chat = is_custom_gpt or has_files

    if can_chat:
        # Display custom GPT info if in custom GPT mode
        if is_custom_gpt:
            st.markdown(
                f"<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); "
                f"padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>"
                f"<h3 style='color: white; margin: 0;'>🤖 {st.session_state.custom_gpt_name}</h3>"
                f"<small style='color: rgba(255,255,255,0.8);'>Custom GPT Mode Active</small>",
                unsafe_allow_html=True,
            )

            # Show document info if documents are attached
            if st.session_state.custom_gpt_document_names:
                doc_names = st.session_state.custom_gpt_document_names
                doc_count = len(doc_names)
                more_text = f" and {doc_count - 3} more" if doc_count > 3 else ""
                st.markdown(
                    f"<div style='color: white;'><small>📚 Reference Documents: "
                    f"{', '.join(doc_names[:3])}{more_text}"
                    f"</small></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='color: white;'><small>ℹ️ Using general knowledge "
                    "(no documents attached)</small></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Display current file info in regular mode
            file_name = st.session_state.file_names.get(
                st.session_state.file_id, st.session_state.file_id
            )
            st.markdown(
                f"<small>File: {file_name}</small>",
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

        # Display conversation starters for Custom GPT if messages are empty
        if is_custom_gpt and len(st.session_state.messages) == 0:
            conversation_starters = st.session_state.get(
                "custom_gpt_conversation_starters", []
            )
            if conversation_starters and len(conversation_starters) >= 3:
                _display_conversation_starters(conversation_starters)
            else:
                # Fall back to default greeting if no conversation starters
                initial_greeting = _generate_initial_greeting()
                st.session_state.messages.append(
                    {"role": "assistant", "content": initial_greeting}
                )

        # Display messages first
        if False and len(st.session_state.messages) > 0:
            for message in st.session_state.messages:
                # If the assistant returned a chart configuration, render the chart *outside*
                # of the styled chat container so none of the custom CSS interferes.
                if "chart" in message:
                    # Optional explanatory text
                    if message.get("content"):
                        st.write(message["content"])
                    try:
                        fig = plot_chart(message["chart"])
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error rendering chart: {str(e)}")
                        st.write("Raw chart data:", message["chart"])
                else:
                    with st.chat_message(message["role"]):
                        if (
                            isinstance(message["content"], str)
                            and "|" in message["content"]
                            and "\n" in message["content"]
                        ):
                            st.markdown(message["content"])
                        else:
                            st.write(message["content"])

        # Handle user input first (this processes the submission)
        user_input = st.chat_input("Enter your message")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.spinner("Processing your request..."):
                # Send full chat history so backend can resolve context properly
                previous_messages = [
                    msg["content"] for msg in st.session_state.messages
                ]

                if not st.session_state.current_session_id:
                    st.error(
                        "Error: No session ID available. Please upload the file again."
                    )
                    return

                chat_payload = _get_chat_payload(previous_messages)
                logging.info(f"Sending chat payload: {chat_payload}")

                # Determine endpoint based on custom_gpt_mode and file presence
                is_custom_gpt = st.session_state.get("custom_gpt_mode", False)

                # For custom GPT, check custom_gpt_document_ids; for regular chat, check file_ids
                if is_custom_gpt:
                    has_files = bool(st.session_state.get("custom_gpt_document_ids"))
                else:
                    has_files = bool(st.session_state.get("file_ids"))

                if is_custom_gpt and not has_files:
                    # Custom GPT without files - use Anthropic endpoint
                    endpoint = f"{API_URL}/chat/anthropic"
                    logging.info(
                        "Using /chat/anthropic endpoint (Custom GPT without files)"
                    )
                else:
                    # Custom GPT with files OR regular file chat - use file/chat endpoint
                    endpoint = f"{API_URL}/file/chat"
                    logging.info(
                        f"Using /file/chat endpoint (Custom GPT: {is_custom_gpt}, Has files: {has_files})"
                    )

                chat_response = requests.post(endpoint, json=chat_payload)
                _handle_chat_response(chat_response)

            st.rerun()

        _display_messages()

    else:
        st.warning("Please upload and process a file first")


def _display_messages():
    """Display the chat messages and charts."""
    for message in st.session_state.messages:
        # Render chart messages outside of the chat container to avoid CSS conflicts
        if "chart" in message:
            try:
                fig = plot_chart(message["chart"])
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering chart: {str(e)}")
                st.write("Raw chart data:", message["chart"])

            # Show textual summary below the chart (if provided)
            if message.get("content"):
                st.write(message["content"])

            # Show intermediate steps for chart messages if available
            if "intermediate_steps" in message:
                if message["intermediate_steps"]:
                    with st.expander("🔍 View Intermediate Steps", expanded=False):
                        st.text(message["intermediate_steps"])
                else:
                    with st.expander("🔍 View Intermediate Steps", expanded=False):
                        st.info(
                            "No agent used. Direct answer provided from database summary."
                        )
        else:
            with st.chat_message(message["role"]):
                if (
                    isinstance(message["content"], str)
                    and "|" in message["content"]
                    and "\n" in message["content"]
                ):
                    st.markdown(message["content"])
                else:
                    st.write(message["content"])

                # Show intermediate steps for assistant messages if available
                if message["role"] == "assistant" and "intermediate_steps" in message:
                    if message["intermediate_steps"]:
                        with st.expander("🔍 View Intermediate Steps", expanded=False):
                            st.text(message["intermediate_steps"])
                    else:
                        with st.expander("🔍 View Intermediate Steps", expanded=False):
                            st.info(
                                "No agent used. Direct answer provided from database summary."
                            )


def _handle_chat_response(chat_response):
    """Handle the response from the chat API."""
    if chat_response.status_code == 200:
        chat_result = chat_response.json()
        if "chart_config" in chat_result:
            try:
                chart_config = chat_result["chart_config"]
                summary = chat_result.get("summary")
                ai_message = {
                    "role": "assistant",
                    "content": (
                        summary
                        if summary
                        else (
                            f"Generated {chart_config['chart_type']} "
                            f"visualization: {chart_config['title']}"
                        )
                    ),
                    "chart": chart_config,
                }
                # Include intermediate steps if available
                if "intermediate_steps" in chat_result:
                    ai_message["intermediate_steps"] = chat_result["intermediate_steps"]
                st.session_state.messages.append(ai_message)
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")
                st.write("Raw chart data:", chat_result.get("chart_config"))
        else:
            ai_message = {
                "role": "assistant",
                "content": chat_result.get("response", str(chat_result)),
            }
            # Include intermediate steps if available
            if "intermediate_steps" in chat_result:
                ai_message["intermediate_steps"] = chat_result["intermediate_steps"]
            st.session_state.messages.append(ai_message)
    else:
        # Parse structured error if available
        error_msg = _parse_error_response(chat_response)
        st.error(f"Chat failed: {error_msg}")


def _get_chat_payload(previous_messages):
    """Construct the payload for the chat API request."""
    chat_payload = {
        "text": previous_messages,
        "model_choice": st.session_state.model_choice,
        "user_id": st.session_state.username,
        "generate_visualization": st.session_state.generate_visualization,
        "session_id": st.session_state.current_session_id,
    }

    if st.session_state.temperature is not None:
        chat_payload["temperature"] = st.session_state.temperature

    # Add custom GPT parameters if in custom GPT mode
    if st.session_state.get("custom_gpt_mode", False):
        chat_payload["custom_gpt"] = True
        if st.session_state.custom_gpt_system_prompt:
            # Append document grounding prompt if documents are attached
            system_prompt = st.session_state.custom_gpt_system_prompt
            if st.session_state.custom_gpt_document_ids:
                system_prompt = f"{system_prompt}\n\n{DOCUMENT_GROUNDING_PROMPT}"
                logging.info(
                    "Custom GPT mode with documents: appended document grounding prompt"
                )
            # Always append general prompt at the end
            system_prompt = f"{system_prompt}\n\n{CUSTOM_GPT_GENERAL_PROMPT}"
            chat_payload["system_prompt"] = system_prompt
            logging.info("Custom GPT mode enabled with custom system prompt")

        # Use custom GPT document IDs if available
        if st.session_state.custom_gpt_document_ids:
            # Handle single vs multi-file: backend expects file_id for single, file_ids for multi
            if len(st.session_state.custom_gpt_document_ids) == 1:
                chat_payload["file_id"] = st.session_state.custom_gpt_document_ids[0]
                logging.info(
                    f"Custom GPT mode with 1 document (single-file): {chat_payload['file_id']}"
                )
            else:
                chat_payload["file_ids"] = st.session_state.custom_gpt_document_ids
                logging.info(
                    f"Custom GPT mode with {len(st.session_state.custom_gpt_document_ids)} documents (multi-file)"
                )
        # If no custom documents but custom GPT mode, allow chat without file context
        # (GPT will use its general knowledge)
    else:
        # Regular chat mode - backward compatible
        chat_payload["custom_gpt"] = False

        if st.session_state.multi_file_mode and st.session_state.file_ids:
            chat_payload["file_ids"] = st.session_state.file_ids
            logging.info(
                f"Multi-file mode. Sending all file_ids: {chat_payload['file_ids']}"
            )
        elif st.session_state.file_id:
            chat_payload["file_id"] = st.session_state.file_id
            logging.info(
                f"Single-file mode. Sending file_id: {chat_payload['file_id']}"
            )
        else:
            st.error("Error: No file context available for chat.")
            return None

    return chat_payload


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
        available_text_models = st.session_state.model_types.get(
            "text", st.session_state.available_models
        )
        default_text_model = (
            DEFAULT_TEXT_MODEL
            if DEFAULT_TEXT_MODEL in available_text_models
            else (available_text_models[0] if available_text_models else None)
        )
        st.session_state.model_choice = (
            default_text_model or st.session_state.available_models[0]
        )

    # Always ensure temp_model_choice is properly initialized
    if "temp_model_choice" not in st.session_state:
        st.session_state.temp_model_choice = st.session_state.model_choice

    if "model_initialized" not in st.session_state:
        st.session_state.model_initialized = False


def setup_default_model_types():
    """Set up default model types based on model name patterns."""
    # Default categorization if API doesn't provide it
    # Identify image models by name pattern
    image_models = [
        m
        for m in st.session_state.available_models
        if "dall-e" in m.lower() or "imagen" in m.lower() or "nanobanana" in m.lower()
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
    st.session_state.available_models = [
        DEFAULT_TEXT_MODEL,
        "gpt_4o_mini",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]
    # Fallback model types if API call fails
    st.session_state.model_types = {
        "text": [
            DEFAULT_TEXT_MODEL,
            "gpt_4o_mini",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ],
        "image": [
            "dall-e-3",
            "imagen-3.0-generate-002",
            "NanoBanana",
        ],  # Assuming dall-e and imagen are your image models
    }


def initialize_ui_state():
    """Initialize UI related state variables."""
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    if "nav_option" not in st.session_state:
        st.session_state.nav_option = "Chat"
    if "generate_visualization" not in st.session_state:
        st.session_state.generate_visualization = False
    # Load username from query params if not already set
    if "username" not in st.session_state or not st.session_state.username:
        if "username" in st.query_params and st.query_params["username"]:
            st.session_state.username = st.query_params["username"]
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "temperature" not in st.session_state:
        st.session_state.temperature = None

    # Initialize custom GPT session state variables
    if "custom_gpt_mode" not in st.session_state:
        st.session_state.custom_gpt_mode = False
    if "custom_gpt_system_prompt" not in st.session_state:
        st.session_state.custom_gpt_system_prompt = None
    if "custom_gpt_name" not in st.session_state:
        st.session_state.custom_gpt_name = ""
    if "custom_gpt_document_ids" not in st.session_state:
        st.session_state.custom_gpt_document_ids = []
    if "custom_gpt_document_names" not in st.session_state:
        st.session_state.custom_gpt_document_names = []


def initialize_session_state():
    """Initialize all session state variables by calling specialized functions."""
    initialize_file_state()
    initialize_messages_state()
    initialize_models_state()
    initialize_ui_state()


def on_model_change():
    # Safely handle temp_model_choice in case it doesn't exist
    if (
        hasattr(st.session_state, "temp_model_choice")
        and st.session_state.temp_model_choice
    ):
        st.session_state.model_choice = st.session_state.temp_model_choice
    else:
        # Fallback to current model choice if temp_model_choice is missing
        st.session_state.temp_model_choice = st.session_state.model_choice


def initialize_model(model_choice):
    # Update the model choice in session state
    st.session_state.model_choice = model_choice

    # Make sure model_types exists in session state
    if "model_types" not in st.session_state:
        st.session_state.model_types = {"text": [], "image": []}

    # Check if this is an image generation model
    is_image_model = model_choice in st.session_state.model_types.get("image", [])

    # Store model type in session state
    previous_model_type = st.session_state.get("current_model_type", "text")
    st.session_state.current_model_type = "image" if is_image_model else "text"

    # Auto-switch to Image generation tab only when model changes TO an image model
    # Don't switch if user is already on Chat tab with a text model
    if is_image_model and previous_model_type != "image":
        st.session_state.nav_option = "Image generation"
    elif not is_image_model and previous_model_type == "image":
        # If switching from image to text model, go to Chat tab
        st.session_state.nav_option = "Chat"

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
    """Render the top navigation bar with horizontal modern buttons."""
    # Make sure model_types exists in session state
    if "model_types" not in st.session_state:
        st.session_state.model_types = {"text": [], "image": []}

    # Don't auto-switch here - let initialize_model handle it when model changes
    # This prevents overriding user's explicit navigation choices

    # Create navigation container with modern styling
    st.markdown(
        "<div class='nav-container' style='margin-bottom: 1.5rem; text-align: center; width: 100%;'>",
        unsafe_allow_html=True,
    )

    # Create horizontal layout for navigation buttons
    nav_cols = st.columns([2, 1, 1, 1, 2])  # Center the three buttons horizontally

    # Place each button in its own column for horizontal layout
    with nav_cols[1]:
        if st.button(
            "Chat", key="nav_Chat", help="Go to Chat", use_container_width=True
        ):
            st.session_state.nav_option = "Chat"

    with nav_cols[2]:
        if st.button(
            "Image generation",
            key="nav_Image_generation",
            help="Go to Image generation",
            use_container_width=True,
        ):
            st.session_state.nav_option = "Image generation"
            # Auto-select default image model when switching to Image generation
            image_models = st.session_state.model_types.get("image", [])
            if image_models:
                # Try to find imagen-4 first, otherwise use the first image model
                default_image_model = None
                for model in image_models:
                    if "imagen-4" in model.lower():
                        default_image_model = model
                        break
                if not default_image_model:
                    default_image_model = image_models[0]

                # Only change if current model is not already an image model
                if st.session_state.model_choice not in image_models:
                    st.session_state.model_choice = default_image_model
                    st.session_state.temp_model_choice = default_image_model
                    st.session_state.current_model_type = "image"

    # Properly close container div
    with nav_cols[3]:
        if st.button(
            "Custom GPT",
            key="nav_Custom_GPT",
            help="Go to Custom GPT Creator",
            use_container_width=True,
        ):
            st.session_state.nav_option = "Custom GPT"

    st.markdown("</div>", unsafe_allow_html=True)


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

            # CRITICAL FIX: Extract and set session_id for URL uploads
            session_id = upload_result.get("session_id")
            logging.info(f"Extracted session_id from URL upload result: {session_id}")
            if session_id:
                st.session_state.current_session_id = session_id
                logging.info(
                    f"Successfully set session_id in session state: {session_id}"
                )
            else:
                logging.error("No session_id found in URL upload response!")
                st.error(
                    "URL processing succeeded but no session ID was returned. Please try again."
                )

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
            }
            st.session_state.file_names[file_id] = file_name

            # Always add the file_id to our list of files for multi-file context
            if file_id not in st.session_state.file_ids:
                st.session_state.file_ids.append(file_id)
                logging.info(f"Added file {file_id} to available files list")

            if upload_result.get("status") == "success":
                st.success(f"{uploaded_file.name} processed successfully.")
            elif upload_result.get("status") == "partial":
                st.warning(f"{uploaded_file.name}: {upload_result['message']}")
            else:
                st.info(upload_result["message"])

            # CRITICAL FIX: Extract and set session_id for single file uploads
            session_id = upload_result.get("session_id")
            logging.info(
                f"Extracted session_id from single file upload result: {session_id}"
            )
            if session_id:
                st.session_state.current_session_id = session_id
                logging.info(
                    f"Successfully set session_id in session state: {session_id}"
                )
            else:
                logging.error("No session_id found in single file upload response!")
                st.error(
                    "File upload succeeded but no session ID was returned. Please try again."
                )

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
    _render_model_selection()
    _render_new_chat_button()

    if st.session_state.nav_option == "Chat":
        _render_chat_file_interface()

    _render_user_information()
    _render_temperature_settings()


def main():
    """Main application function with reduced complexity."""
    # Apply custom CSS
    apply_custom_css()

    # Initialize session state
    initialize_session_state()

    # Top navigation bar (moved to very top)
    render_navigation()

    # Use proper sidebar for collapsibility
    with st.sidebar:
        render_sidebar()

    # Main content area
    # Display different content based on navigation selection
    if st.session_state.nav_option == "Chat":
        display_chat_interface()
    elif st.session_state.nav_option == "Image generation":
        handle_image_generation()
    elif st.session_state.nav_option == "Custom GPT":
        display_custom_gpt_creator()
    elif st.session_state.nav_option == "Chart Generation":
        st.title("Chart Generation")
        st.write("Upload CSV/Excel files and generate visualizations.")
    elif st.session_state.nav_option == "Reference":
        st.title("📚 Reference")
        st.markdown("### Welcome to the RAG Chatbot Reference Page")


def _parse_error_response(upload_response):
    """Parse error response from API and extract meaningful error message with code and key."""
    # IMPORTANT: requests.Response is falsy for HTTP status >= 400.
    # We only want to treat it as missing when it's actually None.
    if upload_response is None:
        logging.error("_parse_error_response: No upload_response provided")
        return "Unknown error - no response from server"

    logging.info(f"_parse_error_response: Status code: {upload_response.status_code}")
    logging.info(f"_parse_error_response: Response text: {upload_response.text[:500]}")

    try:
        # Try to parse JSON error response from API
        error_data = upload_response.json()
        logging.info(f"_parse_error_response: Parsed JSON: {error_data}")

        if isinstance(error_data, dict):
            # First check for structured error format (code, key, message)
            code = error_data.get("code") or error_data.get("error_code")
            key = error_data.get("key") or error_data.get("error_key")
            message = error_data.get("message")

            # If we have structured error, format it nicely
            if code and key and message:
                error_msg = f"Error {code}: {key} - {message}"
                logging.info(
                    f"_parse_error_response: Using structured error: {error_msg}"
                )
                return error_msg

            # Fallback to legacy error handling
            if "error" in error_data:
                error_msg = error_data["error"]
                logging.info(f"_parse_error_response: Using 'error' field: {error_msg}")
                return error_msg
            elif "message" in error_data:
                error_msg = error_data["message"]
                logging.info(
                    f"_parse_error_response: Using 'message' field: {error_msg}"
                )
                return error_msg
            elif "detail" in error_data and isinstance(error_data["detail"], dict):
                # Handle HTTPException detail format
                detail = error_data["detail"]
                # Check if detail has structured error
                detail_code = detail.get("code") or detail.get("error_code")
                detail_key = detail.get("key") or detail.get("error_key")
                detail_message = detail.get("message")

                if detail_code and detail_key and detail_message:
                    error_msg = f"Error {detail_code}: {detail_key} - {detail_message}"
                    logging.info(
                        f"_parse_error_response: Using structured error from detail: {error_msg}"
                    )
                    return error_msg

                if "error" in detail:
                    error_msg = detail["error"]
                    logging.info(
                        f"_parse_error_response: Using detail.error: {error_msg}"
                    )
                    return error_msg
                elif "message" in detail:
                    error_msg = detail["message"]
                    logging.info(
                        f"_parse_error_response: Using detail.message: {error_msg}"
                    )
                    return error_msg
                else:
                    error_msg = str(detail)
                    logging.info(
                        f"_parse_error_response: Using str(detail): {error_msg}"
                    )
                    return error_msg
            elif "detail" in error_data:
                error_msg = str(error_data["detail"])
                logging.info(f"_parse_error_response: Using detail string: {error_msg}")
                return error_msg
            else:
                error_msg = upload_response.text
                logging.info(
                    f"_parse_error_response: Fallback to response.text: {error_msg}"
                )
                return error_msg
        else:
            error_msg = upload_response.text
            logging.info(
                f"_parse_error_response: Non-dict response, using text: {error_msg}"
            )
            return error_msg
    except Exception as e:
        # Fallback to text if JSON parsing fails
        error_msg = upload_response.text
        logging.error(
            f"_parse_error_response: JSON parsing failed: {str(e)}, using text: {error_msg}"
        )
        return error_msg


def enhanced_batch_upload(
    files_list, existing_file_ids_input, is_image=False, urls_input=None
):
    """Handle batch file uploads to the backend API."""
    file_count = len(files_list) if files_list else 0
    has_file_ids = bool(existing_file_ids_input and existing_file_ids_input.strip())
    has_urls = bool(urls_input and urls_input.strip())

    logging.info(
        f"[enhanced_batch_upload] Processing: files={file_count}, "
        f"file_ids={has_file_ids}, urls={has_urls}"
    )

    # Prepare form data
    form_data = {"is_image": str(is_image), "username": st.session_state.username}

    # Prepare files data
    files_data = []
    if files_list:
        for f in files_list:
            files_data.append(("files", (f.name, f.getvalue(), f.type)))

    # Add URLs if present
    if has_urls:
        form_data["urls"] = urls_input

    if has_file_ids:
        form_data["existing_file_ids"] = existing_file_ids_input.strip()

    # Send the request
    try:
        response = requests.post(
            f"{API_URL}/file/upload",
            files=files_data if files_data else None,
            data=form_data,
        )
        logging.info(f"[enhanced_batch_upload] Response status: {response.status_code}")
        return response
    except Exception as e:
        logging.error(f"[enhanced_batch_upload] Error: {str(e)}")
        return None


def handle_existing_file_ids_processing(existing_file_ids_input):
    """Handle processing of existing file IDs only."""
    logging.info(
        f"[handle_existing_file_ids_processing] Processing file IDs: {existing_file_ids_input}"
    )

    if not st.session_state.username:
        st.warning("Please enter a username before processing file IDs")
        return

    with st.spinner("Processing existing file IDs..."):
        # Use enhanced upload with no files, only existing file IDs
        upload_response = enhanced_batch_upload(
            None, existing_file_ids_input, False, None
        )

        if upload_response and upload_response.status_code == 200:
            try:
                result = upload_response.json()
                logging.info(
                    f"File IDs processing successful with status code: {upload_response.status_code}"
                )

                file_ids = result.get("file_ids", [])
                filenames = result.get("original_filenames", [])

                if file_ids:
                    logging.info(
                        f"Received {len(file_ids)} file IDs from existing file processing"
                    )
                    # Set multi_file_mode based on actual number of files
                    st.session_state.multi_file_mode = len(file_ids) > 1

                    if "processed_file_map" not in st.session_state:
                        st.session_state.processed_file_map = {}
                    if "file_ids" not in st.session_state:
                        st.session_state.file_ids = []
                    if "file_names" not in st.session_state:
                        st.session_state.file_names = {}

                    # Reset file_ids for new session
                    st.session_state.file_ids = []

                    for i, file_id in enumerate(file_ids):
                        filename = (
                            filenames[i] if i < len(filenames) else f"file_{file_id}"
                        )
                        st.session_state.processed_file_map[filename] = file_id
                        st.session_state.file_ids.append(file_id)
                        st.session_state.file_names[file_id] = filename

                    # Set file_id for single file compatibility
                    st.session_state.file_id = file_ids[0] if file_ids else None
                    st.session_state.file_uploaded = True

                    # Store session_id if provided
                    session_id = result.get("session_id")
                    if session_id:
                        st.session_state.current_session_id = session_id
                        logging.info(
                            f"Successfully set session_id in session state: {session_id}"
                        )

                    st.success(
                        f"{len(file_ids)} existing files processed successfully!"
                    )
                else:
                    st.error("No file IDs returned from the server.")
            except Exception as e:
                logging.error(f"Error processing file IDs response: {str(e)}")
                st.error(f"Error processing server response: {str(e)}")
        else:
            error_msg = _parse_error_response(upload_response)
            logging.error(f"File IDs processing failed: {error_msg}")
            st.error(f"Failed to process file IDs: {error_msg}")


def _render_new_chat_button():
    """Render the New Chat button with modern styling."""
    if st.button("New Chat", key="new_chat_btn"):
        cleanup_files()


def _render_file_type_selection():
    """Deprecated: file type selection removed."""
    pass


def _get_file_types_config():
    """Get file types configuration."""
    return [
        "pdf",
        "txt",
        "doc",
        "docx",
        "csv",
        "xls",
        "xlsx",
        "db",
        "sqlite",
        "jpg",
        "jpeg",
        "png",
        "gif",
        "bmp",
        "webp",
    ]


def _render_database_info():
    """Render database file information."""
    # No-op: file type selection removed; info covered in _display_file_type_info
    pass


def _process_url_and_file_ids(url_input, existing_file_ids_input):
    """Process both URLs and existing file IDs from the sidebar."""
    with st.spinner("Processing URLs and file IDs..."):
        upload_response = enhanced_batch_upload(
            [], existing_file_ids_input, False, url_input
        )
        _process_sidebar_upload_response(upload_response)


def _process_sidebar_upload_response(upload_response):
    """Process upload response in sidebar context."""
    if upload_response and upload_response.status_code == 200:
        try:
            result = upload_response.json()
            file_ids = result.get("file_ids", [])
            filenames = result.get("original_filenames", [])

            if file_ids:
                _update_sidebar_session_state(file_ids, filenames, result)
                st.success(f"{len(file_ids)} items processed successfully!")
                st.session_state.messages = []  # Reset chat history
            else:
                st.error("No file IDs returned from the server.")
        except Exception as e:
            logging.error(f"Error processing response: {str(e)}")
            st.error(f"Error processing server response: {str(e)}")
    else:
        error_msg = _parse_error_response(upload_response)
        logging.error(f"Processing failed: {error_msg}")
        st.error(f"Failed to process: {error_msg}")


def _update_sidebar_session_state(file_ids, filenames, result):
    """Update session state from sidebar upload."""
    st.session_state.multi_file_mode = len(file_ids) > 1
    st.session_state.file_ids = file_ids
    st.session_state.file_id = file_ids[0] if file_ids else None
    st.session_state.file_uploaded = True

    # Initialize session state dicts if needed
    if "file_names" not in st.session_state:
        st.session_state.file_names = {}
    if "processed_file_map" not in st.session_state:
        st.session_state.processed_file_map = {}

    # Store file names
    for i, file_id in enumerate(file_ids):
        filename = filenames[i] if i < len(filenames) else f"file_{file_id}"
        st.session_state.file_names[file_id] = filename
        st.session_state.processed_file_map[filename] = file_id

    # Store session_id if provided
    session_id = result.get("session_id")
    if session_id:
        st.session_state.current_session_id = session_id


def _render_url_interface():
    """Render URL input interface."""
    st.info("Enter one or more URLs separated by commas to chat with their contents.")
    url_input = st.text_area("Enter URLs (comma-separated for multiple URLs)")

    st.markdown("**file ids:**")
    existing_file_ids_input = st.text_input(
        "Enter existing File IDs (comma-separated)",
        placeholder="file-id-1, file-id-2",
        label_visibility="collapsed",
        key="url_file_ids",
    )

    has_urls = url_input and url_input.strip()
    has_existing_file_ids = existing_file_ids_input and existing_file_ids_input.strip()

    if has_urls or has_existing_file_ids:
        process_items = []
        if has_urls:
            url_count = len(
                [
                    url.strip()
                    for url in url_input.replace("\n", ",").split(",")
                    if url.strip()
                ]
            )
            process_items.append(f"{url_count} URL{'s' if url_count > 1 else ''}")
        if has_existing_file_ids:
            file_ids_list = existing_file_ids_input.replace("\n", ",").split(",")
            existing_count = len([fid.strip() for fid in file_ids_list if fid.strip()])
            process_items.append(
                f"{existing_count} existing file ID{'s' if existing_count > 1 else ''}"
            )

        button_label = "Process " + " and ".join(process_items)

        if st.button(button_label, key="process_url_and_fileids"):
            _process_url_and_file_ids(url_input, existing_file_ids_input)


def _render_file_uploader_interface(existing_file_ids_input=None):
    """Render the file uploader and process sidebar uploads."""
    with st.form("sidebar_file_uploader_form"):
        uploaded_files = st.file_uploader(
            "Upload files",
            type=_get_file_types_config(),
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Upload")

        if submitted:
            # Check if we have files to upload or existing file IDs to process
            has_files = uploaded_files and len(uploaded_files) > 0
            has_existing_file_ids = (
                existing_file_ids_input and existing_file_ids_input.strip()
            )

            if has_files or has_existing_file_ids:
                with st.spinner("Processing files and file IDs..."):
                    upload_response = enhanced_batch_upload(
                        uploaded_files,
                        existing_file_ids_input,
                        is_image=False,
                        urls_input=None,
                    )
                    _process_sidebar_upload_response(upload_response)
            else:
                st.warning("Please upload files or enter existing file IDs")


def _render_uploaded_image_sidebar():
    """Render uploaded image in sidebar."""
    if st.session_state.uploaded_image is not None:
        st.markdown('<div class="file-info">', unsafe_allow_html=True)
        st.subheader("Uploaded Image:")
        img = Image.open(st.session_state.uploaded_image)
        st.image(img, width=200)  # Smaller for sidebar
        st.markdown("</div>", unsafe_allow_html=True)


def _render_chat_file_interface():
    """Render file interface for Chat mode with modern styling."""
    st.markdown('<div class="sidebar-header">File Upload</div>', unsafe_allow_html=True)

    # Always show file ID input field
    st.markdown("**Existing File IDs:**")
    existing_file_ids_input = st.text_input(
        "Enter existing File IDs (comma-separated)",
        placeholder="file-id-1, file-id-2",
        label_visibility="collapsed",
        key="sidebar_file_ids",
    )

    _render_file_uploader_interface(existing_file_ids_input)

    _render_uploaded_image_sidebar()


def _render_user_information():
    """Render user information section with modern styling (persistent username)."""
    st.markdown(
        '<div class="sidebar-header">User Information</div>', unsafe_allow_html=True
    )
    current_val = st.session_state.get("username", "")
    username_input = st.text_input(
        "Enter your username:",
        label_visibility="collapsed",
        placeholder="Enter your username",
        value=current_val,
        key="username_input",
    )
    # Only update if user provided a non-empty value (prevents blank overwrite on rerun)
    if username_input:
        if username_input != current_val:
            st.session_state.username = username_input
            # Persist in URL query params
            st.query_params["username"] = username_input
    # Ensure session_state.username remains accessible even if input cleared
    elif not st.session_state.get("username"):
        st.session_state.username = ""


def _render_model_selection():
    """Render model selection section with modern styling."""
    st.markdown(
        '<div class="sidebar-header">Model Selection</div>', unsafe_allow_html=True
    )

    # Ensure temp_model_choice is properly synchronized
    if "temp_model_choice" not in st.session_state:
        st.session_state.temp_model_choice = st.session_state.model_choice

    # Get the current index safely
    try:
        current_index = st.session_state.available_models.index(
            st.session_state.model_choice
        )
    except ValueError:
        # If current model choice is not in available models, default to first one
        current_index = 0
        st.session_state.model_choice = st.session_state.available_models[0]
        st.session_state.temp_model_choice = st.session_state.model_choice

    st.selectbox(
        "Select Model",
        options=st.session_state.available_models,
        index=current_index,
        key="temp_model_choice",
        on_change=on_model_change,
        label_visibility="collapsed",
    )


def _render_temperature_settings():
    """Render temperature settings section with modern styling."""
    st.markdown(
        '<div class="sidebar-header">Temperature Settings</div>', unsafe_allow_html=True
    )

    st.markdown(
        "<small style='color: var(--color-text-muted);'>"
        "Temperature controls randomness: 0.0 = focused, 1.0 = creative</small>",
        unsafe_allow_html=True,
    )

    use_auto_temperature = st.checkbox(
        "Use model defaults",
        value=st.session_state.temperature is None,
        help="Let the system choose optimal temperature based on model type (OpenAI: 0.5, Gemini: 0.8)",
    )

    if use_auto_temperature:
        st.session_state.temperature = None
        st.markdown(
            '<small style="color: var(--color-text-muted);">Using automatic temperature based on model</small>',
            unsafe_allow_html=True,
        )
    else:
        temperature_value = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7
            if st.session_state.temperature is None
            else st.session_state.temperature,
            step=0.1,
            help="Higher values make output more random, lower values more focused",
            label_visibility="collapsed",
        )
        st.session_state.temperature = temperature_value


if __name__ == "__main__":
    main()
