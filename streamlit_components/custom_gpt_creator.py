import json
import logging
import os
import uuid

import requests
import streamlit as st
from PIL import Image

from streamlit_components.custom_gpt_prompts import (
    ANSWER_RESOLUTION_PROMPT,
    AUDIENCE_UNDERSTANDING_PROMPT,
    CAPABILITIES_PROMPT,
    CONVERSATION_STARTERS_PROMPT,
    KNOWLEDGE_CONTEXT_PROMPT,
    PURPOSE_CLARIFICATION_PROMPT,
)

API_URL = "http://localhost:8080"


def _detect_language_via_api(text):
    """
    Detect language by calling the backend API endpoint.

    Args:
        text (str): Text to analyze for language detection

    Returns:
        str: Detected language (e.g., 'English', 'German')
    """
    try:
        response = requests.post(
            f"{API_URL}/language/detect",
            json={"text": text},
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            result = response.json()
            detected_language = result.get("language", "German")
            logging.info(f"[CUSTOM_GPT] Language detected via API: {detected_language}")
            return detected_language
        else:
            logging.warning(
                f"[CUSTOM_GPT] Language detection API failed with status {response.status_code}, "
                f"defaulting to German"
            )
            return "German"
    except Exception as e:
        logging.error(
            f"[CUSTOM_GPT] Language detection API error: {str(e)}, defaulting to German"
        )
        return "German"


def _call_azure_api_generic(
    prompt_template,
    replacements,
    response_key,
    spinner_text,
    step_name=None,
    previous_step_examples=None,
):
    """
    Generic function to call Azure OpenAI API for any GPT creation step.

    Args:
        prompt_template (str): The prompt template to use
        replacements (dict): Dictionary of placeholder replacements {placeholder: value}
        response_key (str): Base key for storing response in session_state
            (will create {response_key}_response and {response_key}_data)
        spinner_text (str): Text to show in the spinner while processing
        step_name (str, optional): Name of the step for backend logging
        previous_step_examples (list, optional): Examples from previous step to include in prompt
    """
    # For prompts after the first one, add language if not already present
    # Always ensure language placeholder available (for first step we pre-compute)
    language = st.session_state.get("custom_gpt_language", "English")
    if "{LANGUAGE}" not in replacements:
        replacements["{LANGUAGE}"] = language

    # Add previous step examples if provided
    if previous_step_examples and "{PREVIOUS_STEP_EXAMPLES}" in prompt_template:
        examples_str = "\n".join([f"- {ex}" for ex in previous_step_examples])
        replacements["{PREVIOUS_STEP_EXAMPLES}"] = examples_str
    elif "{PREVIOUS_STEP_EXAMPLES}" in prompt_template:
        # If placeholder exists but no examples provided, use empty string
        replacements["{PREVIOUS_STEP_EXAMPLES}"] = "No examples from previous step."

    # Format the prompt with all replacements
    formatted_prompt = prompt_template
    for placeholder, value in replacements.items():
        formatted_prompt = formatted_prompt.replace(placeholder, value)

    with st.spinner(spinner_text):
        try:
            # Prepare request payload
            request_payload = {
                "model": "gpt_4_1",
                "message": formatted_prompt,
                "temperature": 1,
            }
            # Add step_name for backend logging
            if step_name:
                request_payload["step_name"] = step_name

            # Call the Azure OpenAI API
            response = requests.post(
                f"{API_URL}/chat/azure",
                json=request_payload,
                headers={
                    "Content-Type": "application/json",
                },
            )

            if response.status_code == 200:
                # Get plain text response (not streaming)
                full_response = response.text

                # Store the response
                setattr(st.session_state, f"{response_key}_response", full_response)

                # Try to parse JSON response
                try:
                    # Clean the response - remove markdown code blocks if present
                    cleaned_response = full_response.strip()
                    if "```json" in cleaned_response:
                        cleaned_response = (
                            cleaned_response.split("```json")[1].split("```")[0].strip()
                        )
                    elif "```" in cleaned_response:
                        cleaned_response = (
                            cleaned_response.split("```")[1].split("```")[0].strip()
                        )

                    parsed_data = json.loads(cleaned_response)
                    setattr(st.session_state, f"{response_key}_data", parsed_data)

                    # Extract and store language from first prompt response
                    if response_key == "purpose_clarification" and isinstance(
                        parsed_data, dict
                    ):
                        model_language = parsed_data.get("language")
                        local_language = st.session_state.get("custom_gpt_language")
                        if model_language and model_language != local_language:
                            # Keep local detection but log mismatch
                            logging.warning(
                                f"[CUSTOM_GPT] Model reported language "
                                f"'{model_language}' but detector selected "
                                f"'{local_language}'. Using detector result."
                            )
                        else:
                            logging.info(
                                f"[CUSTOM_GPT] Confirmed language: {local_language}"
                            )
                except json.JSONDecodeError:
                    # If JSON parsing fails, store raw response
                    setattr(
                        st.session_state,
                        f"{response_key}_data",
                        {"raw_response": full_response},
                    )
            else:
                st.error(
                    f"API call failed with status {response.status_code}: {response.text}"
                )
        except Exception as e:
            st.error(f"Error calling API: {str(e)}")


def _get_azure_response_sync(
    prompt_template, replacements, spinner_text, step_name=None
):
    """
    Synchronous version of Azure API call that returns the response text directly.
    Used for answer resolution.

    Args:
        prompt_template (str): The prompt template to use
        replacements (dict): Dictionary of placeholder replacements
        spinner_text (str): Text to show in the spinner
        step_name (str, optional): Name of the step for backend logging
    """
    # Format the prompt
    formatted_prompt = prompt_template
    for placeholder, value in replacements.items():
        formatted_prompt = formatted_prompt.replace(placeholder, value)

    with st.spinner(spinner_text):
        try:
            # Prepare request payload
            request_payload = {
                "model": "gpt_4_1",
                "message": formatted_prompt,
                "temperature": 0.7,  # Lower temperature for resolution
            }
            # Add step_name for backend logging
            if step_name:
                request_payload["step_name"] = step_name

            response = requests.post(
                f"{API_URL}/chat/azure",
                json=request_payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = response.text.strip()
                return result
            else:
                st.error(f"API call failed: {response.text}")
                return None
        except Exception as e:
            st.error(f"Error calling API: {str(e)}")
            return None


def _resolve_user_input(user_input, question_text, previous_step_key, step_name=None):
    """
    Resolve user input against examples from the previous step.

    Args:
        user_input (str): The user's raw input
        question_text (str): The question that was asked
        previous_step_key (str): Key to retrieve previous step's data from session_state
        step_name (str, optional): Name of the step for backend logging

    Returns:
        str: Resolved answer text
    """
    # If input is empty, return empty
    if not user_input or not user_input.strip():
        return ""

    # Get examples from the previous step's data
    data = getattr(st.session_state, f"{previous_step_key}_data", None)
    examples_list = []
    if data and isinstance(data, dict) and "examples" in data:
        examples_list = data["examples"]

    # If no examples to reference, return input as is
    if not examples_list:
        return user_input

    # Format examples for the prompt
    examples_str = "\n".join([f"- {ex}" for ex in examples_list])

    # Call API to resolve
    resolved_response = _get_azure_response_sync(
        prompt_template=ANSWER_RESOLUTION_PROMPT,
        replacements={
            "{QUESTION}": question_text,
            "{EXAMPLES_LIST}": examples_str,
            "{USER_INPUT}": user_input,
        },
        spinner_text="Understanding your answer...",
        step_name=f"{step_name}_answer_resolution"
        if step_name
        else "answer_resolution",
    )

    return resolved_response if resolved_response else user_input


def _display_step_response(response_key):
    """
    Generic function to display the response from any GPT creation step.

    Args:
        response_key (str): Base key for retrieving response from session_state
            (expects {response_key}_response and {response_key}_data)
    """
    response = getattr(st.session_state, f"{response_key}_response", None)
    data = getattr(st.session_state, f"{response_key}_data", None)

    if response:
        if data and isinstance(data, dict):
            if "question" in data:
                st.write(data["question"])
                if "examples" in data:
                    st.markdown("**Examples:**")
                    for example in data["examples"]:
                        st.markdown(f"- {example}")
            else:
                st.write(response)
        else:
            st.write(response)


def _initialize_session_state():
    """Initialize all session state variables."""
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0

    # Initialize language (default to 'English')
    if "custom_gpt_language" not in st.session_state:
        st.session_state.custom_gpt_language = "English"

    # Initialize conversation starters
    if "custom_gpt_conversation_starters" not in st.session_state:
        st.session_state.custom_gpt_conversation_starters = []

    # API responses
    for key in [
        "purpose_clarification",
        "audience_understanding",
        "tone_style",
        "capabilities",
        "knowledge_context",
        "examples",
    ]:
        if f"{key}_response" not in st.session_state:
            setattr(st.session_state, f"{key}_response", None)
        if f"{key}_data" not in st.session_state:
            setattr(st.session_state, f"{key}_data", None)

    # User inputs
    user_input_keys = [
        "last_initial_response",
        "last_purpose_response",
        "target_audience",
        "last_tone_style",
        "top_capabilities",
        "avoid_doing",
        "specialized_knowledge",
        "ideal_interaction",
        "custom_instructions",
    ]
    for key in user_input_keys:
        if key not in st.session_state:
            setattr(st.session_state, key, "")


def _display_header_and_icon():
    """Display the icon and title header."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        icon_paths = [
            os.path.join(os.path.dirname(__file__), "assets", "custom_gpt_icon.png"),
            "assets/custom_gpt_icon.png",
            os.path.join(os.getcwd(), "assets", "custom_gpt_icon.png"),
        ]

        icon_loaded = False
        for icon_path in icon_paths:
            try:
                normalized_path = os.path.normpath(icon_path)
                if os.path.exists(normalized_path):
                    img = Image.open(normalized_path)
                    st.image(img, width=80, use_container_width=False)
                    icon_loaded = True
                    break
            except Exception:
                continue

        if not icon_loaded:
            try:
                st.image(
                    "assets/custom_gpt_icon.png", width=80, use_container_width=False
                )
            except Exception:
                st.markdown(
                    '<div style="text-align: center; margin-bottom: 10px; font-size: 48px;">🤖</div>',
                    unsafe_allow_html=True,
                )

        st.markdown(
            "<h1 style='text-align: center;'>Create a Custom GPT</h1>",
            unsafe_allow_html=True,
        )


def _display_initial_idea_step():
    """Display Step 0: Initial Idea."""
    if st.session_state.current_step < 0:
        return
    st.header("Initial Idea")
    st.write("What would you like to create a GPT for?")

    if st.session_state.current_step == 0:
        st.markdown(
            "**For example, you might say:**\n\n"
            '- "I need a customer service assistant for my e-commerce store"\n\n'
            '- "I want a coding tutor that explains concepts to beginners"\n\n'
            "Don't worry about being perfect - just describe your idea in your own words, "
            "and I'll ask follow-up questions to refine it.",
        )
        user_initial_response = st.text_area(
            "Enter your initial idea here...",
            value=st.session_state.last_initial_response,
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button(
                "Next →",
                key="next_initial",
                use_container_width=True,
                disabled=not user_initial_response,
            ):
                st.session_state.last_initial_response = user_initial_response

                if st.session_state.purpose_clarification_response is None:
                    # Determine language before first API call
                    detected = _detect_language_via_api(user_initial_response)
                    st.session_state.custom_gpt_language = detected
                    logging.info(
                        f"[CUSTOM_GPT] Language detection for initial idea: '{detected}'"
                    )
                    _call_azure_api_generic(
                        prompt_template=PURPOSE_CLARIFICATION_PROMPT,
                        replacements={
                            "{USER_INITIAL_RESPONSE}": user_initial_response,
                            "{LANGUAGE}": detected,
                        },
                        response_key="purpose_clarification",
                        spinner_text="Generating purpose clarification questions...",
                        step_name="purpose_clarification",
                    )
                st.session_state.current_step = 1
                st.rerun()

        if (
            st.session_state.last_initial_response
            and user_initial_response != st.session_state.last_initial_response
        ):
            # Reset subsequent steps if user edits the initial idea after progressing
            st.session_state.purpose_clarification_response = None
            st.session_state.purpose_clarification_data = None
            st.session_state.audience_understanding_response = None
            st.session_state.audience_understanding_data = None
            st.session_state.last_purpose_response = ""
            st.session_state.current_step = 0
    else:
        st.markdown(f"**Your answer:** {st.session_state.last_initial_response}")
        st.markdown("---")


def _display_purpose_clarification_step(user_initial_response):
    """Display Step 1: Purpose Clarification."""
    if st.session_state.current_step < 1:
        return

    st.subheader("Purpose Clarification")
    _display_step_response("purpose_clarification")

    if st.session_state.current_step == 1:
        problem_to_solve = st.text_area(
            "What specific problems should this GPT solve?",
            value=st.session_state.last_purpose_response,
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(
                "Next →",
                key="next_purpose",
                use_container_width=True,
                disabled=not problem_to_solve,
            ):
                if (
                    st.session_state.last_purpose_response
                    and problem_to_solve != st.session_state.last_purpose_response
                ):
                    st.session_state.audience_understanding_response = None
                    st.session_state.audience_understanding_data = None

                # Resolve the user input
                resolved_purpose = _resolve_user_input(
                    user_input=problem_to_solve,
                    question_text="What specific problems should this GPT solve?",
                    previous_step_key="purpose_clarification",
                    step_name="purpose_clarification",
                )
                st.session_state.last_purpose_response = resolved_purpose

                if st.session_state.audience_understanding_response is None:
                    # Extract examples from purpose clarification step
                    purpose_data = getattr(
                        st.session_state, "purpose_clarification_data", None
                    )
                    previous_examples = []
                    if (
                        purpose_data
                        and isinstance(purpose_data, dict)
                        and "examples" in purpose_data
                    ):
                        previous_examples = purpose_data["examples"]

                    _call_azure_api_generic(
                        prompt_template=AUDIENCE_UNDERSTANDING_PROMPT,
                        replacements={
                            "{USER_INITIAL_RESPONSE}": user_initial_response,
                            "{PURPOSE_RESPONSE}": resolved_purpose,
                        },
                        response_key="audience_understanding",
                        spinner_text="Analyzing your purpose and generating audience questions...",
                        step_name="audience_understanding",
                        previous_step_examples=previous_examples,
                    )
                st.session_state.current_step = 2
                st.rerun()
        with col2:
            if st.button(
                "Skip →",
                key="skip_purpose",
                use_container_width=True,
            ):
                # Skip this step and set empty value
                st.session_state.last_purpose_response = ""
                st.session_state.audience_understanding_response = None
                st.session_state.audience_understanding_data = None
                # Extract examples from purpose clarification step even when skipped
                purpose_data = getattr(
                    st.session_state, "purpose_clarification_data", None
                )
                previous_examples = []
                if (
                    purpose_data
                    and isinstance(purpose_data, dict)
                    and "examples" in purpose_data
                ):
                    previous_examples = purpose_data["examples"]
                _call_azure_api_generic(
                    prompt_template=AUDIENCE_UNDERSTANDING_PROMPT,
                    replacements={
                        "{USER_INITIAL_RESPONSE}": user_initial_response,
                        "{PURPOSE_RESPONSE}": "",
                    },
                    response_key="audience_understanding",
                    spinner_text="Skipping purpose; generating audience questions...",
                    step_name="audience_understanding",
                    previous_step_examples=previous_examples,
                )
                st.session_state.current_step = 2
                st.rerun()
    else:
        st.markdown(f"**Your answer:** {st.session_state.last_purpose_response}")
        st.markdown("---")


def _display_audience_understanding_step():
    """Display Step 2: Audience Understanding."""
    if st.session_state.current_step < 2:
        return

    st.subheader("Audience Understanding")
    _display_step_response("audience_understanding")

    if st.session_state.current_step == 2:
        target_audience = st.text_area(
            "Who will be using this GPT?",
            value=st.session_state.target_audience,
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(
                "Next →",
                key="next_audience",
                use_container_width=True,
                disabled=not target_audience,
            ):
                # Resolve the user input
                resolved_audience = _resolve_user_input(
                    user_input=target_audience,
                    question_text="Who will be using this GPT?",
                    previous_step_key="audience_understanding",
                    step_name="audience_understanding",
                )
                st.session_state.target_audience = resolved_audience
                st.session_state.current_step = 3
                st.rerun()
        with col2:
            if st.button(
                "Skip →",
                key="skip_audience",
                use_container_width=True,
            ):
                # Skip this step and set empty value
                st.session_state.target_audience = ""
                st.session_state.current_step = 3
                st.rerun()
    else:
        st.markdown(f"**Your answer:** {st.session_state.target_audience}")
        st.markdown("---")


def _display_tone_style_step(user_initial_response):
    """Display Step 3: Tone & Style."""
    if st.session_state.current_step < 3:
        return

    # Get detected language
    detected = st.session_state.get("custom_gpt_language", "English")

    # Call tone options API before displaying step
    if st.session_state.tone_style_response is None:
        with st.spinner("Loading tone options..."):
            try:
                response = requests.get(
                    f"{API_URL}/custom-gpt/tone-options",
                    params={"language": detected},
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
                response.raise_for_status()
                st.session_state.tone_style_response = response.json()
                # Parse and store data
                st.session_state.tone_style_data = st.session_state.tone_style_response
            except Exception as e:
                st.error(f"Error loading tone options: {str(e)}")
                return

    st.subheader("Tone & Style")
    _display_step_response("tone_style")

    if st.session_state.current_step == 3:
        tone_style_input = st.text_area(
            "How should your GPT communicate?",
            value=st.session_state.last_tone_style,
            height=100,
            placeholder="Describe the communication style or select from examples above...",
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(
                "Next →",
                key="next_tone",
                use_container_width=True,
                disabled=not tone_style_input,
            ):
                if (
                    st.session_state.last_tone_style
                    and tone_style_input != st.session_state.last_tone_style
                ):
                    st.session_state.capabilities_response = None
                    st.session_state.capabilities_data = None

                # Resolve the user input
                resolved_tone = _resolve_user_input(
                    user_input=tone_style_input,
                    question_text="How should your GPT communicate?",
                    previous_step_key="tone_style",
                    step_name="tone_style",
                )
                st.session_state.last_tone_style = resolved_tone

                if st.session_state.capabilities_response is None:
                    # Extract examples from tone style step
                    tone_data = getattr(st.session_state, "tone_style_data", None)
                    previous_examples = []
                    if (
                        tone_data
                        and isinstance(tone_data, dict)
                        and "examples" in tone_data
                    ):
                        previous_examples = tone_data["examples"]

                    _call_azure_api_generic(
                        prompt_template=CAPABILITIES_PROMPT,
                        replacements={
                            "{USER_INITIAL_RESPONSE}": user_initial_response,
                            "{PURPOSE_RESPONSE}": st.session_state.last_purpose_response,
                            "{AUDIENCE_RESPONSE}": st.session_state.target_audience,
                            "{TONE_STYLE_RESPONSE}": resolved_tone,
                        },
                        response_key="capabilities",
                        spinner_text="Analyzing requirements and generating capability questions...",
                        step_name="capabilities",
                        previous_step_examples=previous_examples,
                    )
                st.session_state.current_step = 4
                st.rerun()
        with col2:
            if st.button(
                "Skip →",
                key="skip_tone",
                use_container_width=True,
            ):
                # Skip this step and set empty value
                st.session_state.last_tone_style = ""
                st.session_state.capabilities_response = None
                st.session_state.capabilities_data = None
                # Extract examples from tone style step
                tone_data = getattr(st.session_state, "tone_style_data", None)
                previous_examples = []
                if (
                    tone_data
                    and isinstance(tone_data, dict)
                    and "examples" in tone_data
                ):
                    previous_examples = tone_data["examples"]
                _call_azure_api_generic(
                    prompt_template=CAPABILITIES_PROMPT,
                    replacements={
                        "{USER_INITIAL_RESPONSE}": user_initial_response,
                        "{PURPOSE_RESPONSE}": st.session_state.last_purpose_response,
                        "{AUDIENCE_RESPONSE}": st.session_state.target_audience,
                        "{TONE_STYLE_RESPONSE}": "",
                    },
                    response_key="capabilities",
                    spinner_text="Skipping tone; generating capability questions...",
                    step_name="capabilities",
                    previous_step_examples=previous_examples,
                )
                st.session_state.current_step = 4
                st.rerun()
    else:
        st.markdown(f"**Your answer:** {st.session_state.last_tone_style}")
        st.markdown("---")


def _display_capabilities_definition_step():
    """Display Step 4: Capabilities Definition."""
    if st.session_state.current_step < 4:
        return

    st.subheader("Capabilities Definition")
    _display_step_response("capabilities")

    if st.session_state.current_step == 4:
        top_capabilities = st.text_area(
            "What are the top 3-5 things this GPT must be able to do? (one per line)",
            value=st.session_state.top_capabilities,
        )
        avoid_doing = st.text_area(
            "Are there any things it should specifically avoid or refuse to do?",
            value=st.session_state.avoid_doing,
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(
                "Next →",
                key="next_capabilities",
                use_container_width=True,
                disabled=not top_capabilities,
            ):
                if (
                    st.session_state.top_capabilities
                    and top_capabilities != st.session_state.top_capabilities
                ):
                    st.session_state.knowledge_context_response = None
                    st.session_state.knowledge_context_data = None

                # Resolve inputs
                resolved_capabilities = _resolve_user_input(
                    user_input=top_capabilities,
                    question_text="What are the top 3-5 things this GPT must be able to do?",
                    previous_step_key="capabilities",
                    step_name="capabilities",
                )
                resolved_avoid = _resolve_user_input(
                    user_input=avoid_doing,
                    question_text="Are there any things it should specifically avoid or refuse to do?",
                    previous_step_key="capabilities",
                    step_name="capabilities",
                )

                st.session_state.top_capabilities = resolved_capabilities
                st.session_state.avoid_doing = resolved_avoid

                if st.session_state.knowledge_context_response is None:
                    # Extract examples from capabilities step
                    capabilities_data = getattr(
                        st.session_state, "capabilities_data", None
                    )
                    previous_examples = []
                    if (
                        capabilities_data
                        and isinstance(capabilities_data, dict)
                        and "examples" in capabilities_data
                    ):
                        previous_examples = capabilities_data["examples"]

                    _call_azure_api_generic(
                        prompt_template=KNOWLEDGE_CONTEXT_PROMPT,
                        replacements={
                            "{USER_INITIAL_RESPONSE}": st.session_state.last_initial_response,
                            "{PURPOSE_RESPONSE}": st.session_state.last_purpose_response,
                            "{AUDIENCE_RESPONSE}": st.session_state.target_audience,
                            "{TONE_STYLE_RESPONSE}": st.session_state.last_tone_style,
                            "{CAPABILITIES_RESPONSE}": resolved_capabilities,
                        },
                        response_key="knowledge_context",
                        spinner_text="Analyzing capabilities and generating knowledge questions...",
                        step_name="knowledge_context",
                        previous_step_examples=previous_examples,
                    )
                st.session_state.current_step = 5
                st.rerun()
        with col2:
            if st.button(
                "Skip →",
                key="skip_capabilities",
                use_container_width=True,
            ):
                # Skip this step and set empty values
                st.session_state.top_capabilities = ""
                st.session_state.avoid_doing = ""
                st.session_state.knowledge_context_response = None
                st.session_state.knowledge_context_data = None
                # Extract examples from capabilities step
                capabilities_data = getattr(st.session_state, "capabilities_data", None)
                previous_examples = []
                if (
                    capabilities_data
                    and isinstance(capabilities_data, dict)
                    and "examples" in capabilities_data
                ):
                    previous_examples = capabilities_data["examples"]
                _call_azure_api_generic(
                    prompt_template=KNOWLEDGE_CONTEXT_PROMPT,
                    replacements={
                        "{USER_INITIAL_RESPONSE}": st.session_state.last_initial_response,
                        "{PURPOSE_RESPONSE}": st.session_state.last_purpose_response,
                        "{AUDIENCE_RESPONSE}": st.session_state.target_audience,
                        "{TONE_STYLE_RESPONSE}": st.session_state.last_tone_style,
                        "{CAPABILITIES_RESPONSE}": "",
                    },
                    response_key="knowledge_context",
                    spinner_text="Skipping capabilities; generating knowledge questions...",
                    step_name="knowledge_context",
                    previous_step_examples=previous_examples,
                )
                st.session_state.current_step = 5
                st.rerun()
    else:
        st.markdown(f"**Your answer:** {st.session_state.top_capabilities}")
        if st.session_state.avoid_doing:
            st.markdown(f"**Things to avoid:** {st.session_state.avoid_doing}")
        st.markdown("---")


def _display_knowledge_context_step():
    """Display Step 5: Knowledge & Context."""
    if st.session_state.current_step < 5:
        return

    st.subheader("Knowledge & Context")
    _display_step_response("knowledge_context")

    if st.session_state.current_step == 5:
        specialized_knowledge = st.text_area(
            "Does this GPT need specialized knowledge, or are there specific terms or jargon it should use?",
            value=st.session_state.specialized_knowledge,
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(
                "Next →",
                key="next_knowledge",
                use_container_width=True,
                disabled=not specialized_knowledge,
            ):
                if (
                    st.session_state.specialized_knowledge
                    and specialized_knowledge != st.session_state.specialized_knowledge
                ):
                    st.session_state.examples_response = None
                    st.session_state.examples_data = None

                # Resolve input
                resolved_knowledge = _resolve_user_input(
                    user_input=specialized_knowledge,
                    question_text=(
                        "Does this GPT need specialized knowledge, or are there "
                        "specific terms or jargon it should use?"
                    ),
                    previous_step_key="knowledge_context",
                    step_name="knowledge_context",
                )

                st.session_state.specialized_knowledge = resolved_knowledge
                st.session_state.current_step = 6
                st.rerun()
        with col2:
            if st.button(
                "Skip →",
                key="skip_knowledge",
                use_container_width=True,
            ):
                # Skip this step and set empty value
                st.session_state.specialized_knowledge = ""
                st.session_state.current_step = 6
                st.rerun()
    else:
        st.markdown(f"**Your answer:** {st.session_state.specialized_knowledge}")
        st.markdown("---")


def _display_examples_collection_step():
    """Display Step 6: Custom Instructions or Additional Requirements."""
    if st.session_state.current_step < 6:
        return

    st.subheader("Custom Instructions or Additional Requirements")
    st.info(
        "📝 Add any specific instructions, constraints, or requirements that "
        "haven't been covered in the previous steps. This is optional."
    )

    if st.session_state.current_step == 6:
        custom_instructions = st.text_area(
            "Custom Instructions (Optional)",
            value=st.session_state.custom_instructions,
            height=200,
            placeholder=(
                "Examples:\n"
                "- Always cite sources when referencing specific information\n"
                "- Use bullet points for lists of 3 or more items\n"
                "- Avoid using technical jargon unless specifically asked\n"
                "- Keep responses under 500 words unless more detail is requested"
            ),
            help=(
                "Use this field to add any additional context, rules, or "
                "instructions that are important for your GPT."
            ),
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(
                "Finish and Generate System Prompt →",
                key="finish_and_generate",
                use_container_width=True,
                type="primary",
            ):
                st.session_state.custom_instructions = custom_instructions
                st.session_state.current_step = 7
                st.rerun()
        with col2:
            if st.button(
                "Skip →",
                key="skip_custom_instructions",
                use_container_width=True,
            ):
                st.session_state.custom_instructions = ""
                st.session_state.current_step = 7
                st.rerun()
    else:
        if st.session_state.custom_instructions:
            st.markdown(
                f"**Custom instructions:** {st.session_state.custom_instructions}"
            )
        else:
            st.markdown("**Custom instructions:** None provided")
        st.markdown("---")


def _generate_conversation_starters():
    """Generate 3 conversation starter questions based on the custom GPT configuration."""
    # Check if already generated
    if (
        hasattr(st.session_state, "custom_gpt_conversation_starters")
        and st.session_state.custom_gpt_conversation_starters
    ):
        return

    try:
        # Get detected language (default to 'English' if not set)
        language = st.session_state.get("custom_gpt_language", "English")

        # Prepare structured input similar to system prompt generation
        structured_input = f"""
user_initial_response: {st.session_state.last_initial_response}
problems_to_solve: {st.session_state.last_purpose_response}
target_users: {st.session_state.target_audience}
tone_style: {st.session_state.last_tone_style}
must_do_capabilities: {st.session_state.top_capabilities}
specialized_knowledge: {st.session_state.specialized_knowledge}
example_interaction: {st.session_state.get('ideal_interaction', '')}
"""

        # Format the prompt with replacements
        formatted_prompt = CONVERSATION_STARTERS_PROMPT.replace("{LANGUAGE}", language)
        formatted_prompt = f"{formatted_prompt}\n\n{structured_input}"

        # Call the Azure OpenAI API
        with st.spinner("Generating conversation starter questions..."):
            response = requests.post(
                f"{API_URL}/chat/azure",
                json={
                    "model": "gpt_4_1",
                    "message": formatted_prompt,
                    "temperature": 1,
                },
                headers={
                    "Content-Type": "application/json",
                },
            )

            if response.status_code == 200:
                full_response = response.text

                # Try to parse JSON response
                try:
                    # Clean the response - remove markdown code blocks if present
                    cleaned_response = full_response.strip()
                    if "```json" in cleaned_response:
                        cleaned_response = (
                            cleaned_response.split("```json")[1].split("```")[0].strip()
                        )
                    elif "```" in cleaned_response:
                        cleaned_response = (
                            cleaned_response.split("```")[1].split("```")[0].strip()
                        )

                    parsed_data = json.loads(cleaned_response)

                    # Extract questions from the response
                    if isinstance(parsed_data, dict) and "questions" in parsed_data:
                        questions = parsed_data["questions"]
                        if isinstance(questions, list) and len(questions) >= 3:
                            st.session_state.custom_gpt_conversation_starters = (
                                questions[:3]
                            )
                            logging.info("Successfully generated conversation starters")
                        else:
                            logging.warning(f"Unexpected questions format: {questions}")
                            st.session_state.custom_gpt_conversation_starters = []
                    else:
                        logging.warning(f"Unexpected response format: {parsed_data}")
                        st.session_state.custom_gpt_conversation_starters = []
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse conversation starters JSON: {e}")
                    logging.error(f"Raw response: {full_response[:500]}")
                    st.session_state.custom_gpt_conversation_starters = []
            else:
                logging.error(
                    f"API call failed with status {response.status_code}: {response.text}"
                )
                st.session_state.custom_gpt_conversation_starters = []
    except Exception as e:
        logging.error(f"Error generating conversation starters: {str(e)}")
        st.session_state.custom_gpt_conversation_starters = []


def _display_system_prompt_step():
    """Auto-generate system prompt after Step 6 and advance to settings with a success flag."""
    # Only act when exactly on step 7
    if st.session_state.current_step != 7:
        return

    # Already advanced; do nothing
    if st.session_state.get("_system_prompt_advanced_once"):
        return

    # Generate if not present
    if not hasattr(st.session_state, "system_prompt_generator_response"):
        with st.spinner("Generating system prompt..."):
            try:
                # Get detected language (default to 'English' if not set)
                language = st.session_state.get("custom_gpt_language", "English")
                response = requests.post(
                    f"{API_URL}/gpt/generate-system-prompt",
                    json={
                        "initial_idea": st.session_state.last_initial_response,
                        "purpose": st.session_state.last_purpose_response,
                        "audience": st.session_state.target_audience,
                        "tone": st.session_state.last_tone_style,
                        "capabilities": st.session_state.top_capabilities,
                        "constraints": st.session_state.avoid_doing or None,
                        "knowledge": st.session_state.specialized_knowledge,
                        "example_interaction": st.session_state.get(
                            "ideal_interaction", ""
                        ),
                        "custom_instructions": st.session_state.custom_instructions
                        or None,
                        "language": language,
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.system_prompt_generator_response = data.get(
                        "system_prompt", ""
                    )
                    # Extract and store GPT name from response
                    st.session_state.custom_gpt_name = data.get(
                        "gpt_name", st.session_state.last_initial_response
                    )
                else:
                    st.session_state.system_prompt_generator_response = None
                    st.error(f"Failed to generate system prompt: {response.text}")
            except Exception as e:
                st.session_state.system_prompt_generator_response = None
                st.error(f"Error generating system prompt: {e}")

    # Success path
    if st.session_state.system_prompt_generator_response:
        st.session_state.generated_system_prompt = (
            st.session_state.system_prompt_generator_response
        )
        st.session_state.system_prompt_generated_success = True

        # Generate conversation starters after system prompt is created
        _generate_conversation_starters()

        st.session_state.current_step = 8
        st.session_state._system_prompt_advanced_once = True
        st.rerun()
    else:
        # Allow manual retry without advancing
        if st.button("Retry system prompt generation", key="retry_sys_prompt_btn"):
            if hasattr(st.session_state, "system_prompt_generator_response"):
                delattr(st.session_state, "system_prompt_generator_response")
            st.rerun()


def _parse_upload_response(data, uploaded_files):
    """Parse upload response and extract file IDs, names, and session ID."""
    file_ids, file_names, session_id = [], [], None

    # Handle both single and multi-file responses
    if data.get("multi_file_mode", False):
        # Multi-file response
        returned_file_ids = data.get("file_ids", [])
        returned_filenames = data.get("original_filenames", [])

        for i, fid in enumerate(returned_file_ids):
            if fid:
                file_ids.append(fid)
                filename = (
                    returned_filenames[i]
                    if i < len(returned_filenames)
                    else getattr(uploaded_files[i], "name", fid)
                )
                file_names.append(filename)

        # Get session_id from response
        session_id = data.get("session_id")
    else:
        # Single file response (backward compatibility)
        fid = data.get("file_id")
        if fid:
            file_ids.append(fid)
            filename = data.get("original_filename") or getattr(
                uploaded_files[0], "name", fid
            )
            file_names.append(filename)

        session_id = data.get("session_id")

    return file_ids, file_names, session_id


def _extract_error_message(resp):
    """Extract error message from response."""
    error_msg = f"Failed to upload documents (Status {resp.status_code})"
    try:
        error_data = resp.json()
        if isinstance(error_data, dict):
            detail = error_data.get("detail", {})
            if isinstance(detail, dict):
                error_msg = detail.get("message", error_msg)
            elif isinstance(detail, str):
                error_msg = detail
            else:
                error_msg = error_data.get("message", error_msg)
    except (ValueError, KeyError):
        error_msg = resp.text or error_msg
    return error_msg


def _process_custom_gpt_documents(uploaded_files, username):
    """Process documents for custom GPT uploads. Returns (file_ids, file_names, session_id)."""
    if not uploaded_files:
        return [], [], None

    file_ids, file_names, session_id = [], [], None

    # Use batch upload for better performance and consistency with regular uploads
    with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):
        try:
            # Prepare files for batch upload (same format as process_multiple_files)
            files_for_request = []
            for file_obj in uploaded_files:
                # Reset file pointer to beginning in case it was read before
                file_obj.seek(0)
                files_for_request.append(
                    ("files", (file_obj.name, file_obj.getvalue(), file_obj.type))
                )

            logging.info(
                f"[CUSTOM_GPT_DEBUG] Uploading {len(uploaded_files)} file(s) with custom_gpt=true"
            )

            # Prepare form data
            form_data = {
                "username": username,
                "is_image": "false",
                "custom_gpt": "true",  # FastAPI Form() will convert string to bool
            }

            # Send batch upload request
            resp = requests.post(
                f"{API_URL}/file/upload",
                files=files_for_request,
                data=form_data,
            )

            if resp.status_code == 200:
                data = resp.json()
                logging.info(f"[CUSTOM_GPT_DEBUG] Upload response: {data}")

                file_ids, file_names, session_id = _parse_upload_response(
                    data, uploaded_files
                )

                if not file_ids:
                    st.error("Upload succeeded but no file IDs were returned.")
            else:
                error_msg = _extract_error_message(resp)
                logging.error(f"[CUSTOM_GPT_DEBUG] Upload error: {error_msg}")
                st.error(f"❌ {error_msg}")

        except requests.exceptions.RequestException as e:
            error_msg = f"Network error: {str(e)}"
            logging.error(f"[CUSTOM_GPT_DEBUG] {error_msg}")
            st.error(f"❌ {error_msg}")
        except Exception as e:
            error_msg = f"Error processing documents: {str(e)}"
            logging.error(f"[CUSTOM_GPT_DEBUG] {error_msg}")
            st.error(f"❌ {error_msg}")

    return file_ids, file_names, session_id


def _initialize_settings_defaults():
    """Initialize default values for settings page."""
    defaults = {
        "custom_gpt_name": "My Custom GPT",
        "custom_gpt_model": "gpt_4o_mini",
        "custom_gpt_temperature": 0.7,
        "custom_gpt_uploaded_docs": [],
        "custom_gpt_document_ids": [],
        "custom_gpt_document_names": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_gpt_configuration():
    """Render GPT configuration inputs."""
    st.markdown("### GPT Configuration")

    gpt_name = st.text_input(
        "GPT Name",
        value=st.session_state.custom_gpt_name,
        help="Give your custom GPT a memorable name",
    )
    st.session_state.custom_gpt_name = gpt_name

    _render_model_selection()
    _render_temperature_slider()
    _render_system_prompt_display()


def _render_system_prompt_display():
    """Render system prompt display with show/hide toggle."""
    if not (
        hasattr(st.session_state, "generated_system_prompt")
        and st.session_state.generated_system_prompt
    ):
        return

    st.markdown("---")
    st.markdown("### 📋 System Prompt")

    # Simple Show/Hide toggle
    show_prompt = st.checkbox(
        "Show System Prompt",
        value=st.session_state.get("system_prompt_visible", False),
        key="system_prompt_toggle",
        help="Toggle to show or hide your generated system prompt",
    )
    st.session_state.system_prompt_visible = show_prompt

    if show_prompt:
        edited_prompt = st.text_area(
            "Customize your system prompt:",
            value=st.session_state.generated_system_prompt,
            height=500,
            key="edit_system_prompt_textarea",
            help="Edit your system prompt as needed",
        )

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button(
                "💾 Save Changes", key="save_system_prompt_btn", type="primary"
            ):
                st.session_state.generated_system_prompt = edited_prompt
                st.success("✅ System prompt updated!")

        with col2:
            if st.button(
                "🔄 Regenerate",
                key="regenerate_system_prompt_btn",
                help="Generate a new system prompt",
            ):
                if hasattr(st.session_state, "system_prompt_generator_response"):
                    delattr(st.session_state, "system_prompt_generator_response")
                st.session_state.current_step = 7
                st.rerun()


def _render_model_selection():
    """Render model selection dropdown."""
    available_models = st.session_state.get(
        "available_models", ["gpt_4o_mini", "gemini-2.5-flash", "gemini-2.5-pro"]
    )
    model_types = st.session_state.get(
        "model_types", {"text": available_models, "image": []}
    )
    text_models = model_types.get("text", available_models)

    if not text_models:
        return

    current_model_index = 0
    if st.session_state.custom_gpt_model in text_models:
        current_model_index = text_models.index(st.session_state.custom_gpt_model)

    model_choice = st.selectbox(
        "Select Model",
        options=text_models,
        index=current_model_index,
        help="Choose the AI model for your custom GPT",
    )
    st.session_state.custom_gpt_model = model_choice


def _render_temperature_slider():
    """Render temperature slider."""
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.custom_gpt_temperature,
        step=0.1,
        help="Controls randomness: 0.0 = focused, 1.0 = balanced, 2.0 = creative",
    )
    st.session_state.custom_gpt_temperature = temperature


def _render_document_section():
    """Render document upload and management section."""
    st.markdown("---")
    st.markdown("### Knowledge Documents (Optional)")
    st.info(
        "📄 Upload documents for your GPT to reference. "
        "This is completely optional - your GPT will work fine without documents."
    )

    _display_attached_documents()
    _render_document_uploader()
    _display_document_summary()


def _display_attached_documents():
    """Display currently attached documents with remove buttons."""
    if not st.session_state.custom_gpt_document_names:
        return

    st.markdown("**Currently Attached Documents:**")
    for idx, doc_name in enumerate(st.session_state.custom_gpt_document_names):
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.markdown(f"📄 {doc_name}")
        with col2:
            if st.button("Remove", key=f"remove_doc_{idx}"):
                st.session_state.custom_gpt_document_names.pop(idx)
                st.session_state.custom_gpt_document_ids.pop(idx)
                st.rerun()


def _render_document_uploader():
    """Render document uploader and process button."""
    uploaded_docs = st.file_uploader(
        "Upload reference documents",
        accept_multiple_files=True,
        type=["pdf", "txt", "doc", "docx", "csv", "xls", "xlsx"],
        help="Upload PDF, text, Word, or spreadsheet files",
        key="custom_gpt_doc_uploader",
    )

    if not uploaded_docs:
        return

    if st.button("Process Documents", key="process_custom_gpt_docs"):
        _handle_document_processing(uploaded_docs)


def _handle_document_processing(uploaded_docs):
    """Handle document processing logic."""
    if not st.session_state.username:
        st.error("Please enter a username in the sidebar before processing documents.")
        return

    logging.info(
        f"[CUSTOM_GPT_DEBUG] Starting document processing for {len(uploaded_docs)} files"
    )
    file_ids, file_names, session_id = _process_custom_gpt_documents(
        uploaded_docs, st.session_state.username
    )
    logging.info(
        f"[CUSTOM_GPT_DEBUG] Processed documents - file_ids: {file_ids}, custom_gpt flag should be True"
    )

    if file_ids:
        st.session_state.custom_gpt_document_ids.extend(file_ids)
        st.session_state.custom_gpt_document_names.extend(file_names)
        # Store session_id if documents were processed
        if session_id:
            st.session_state.custom_gpt_session_id = session_id

        st.success(f"✅ Successfully processed {len(file_ids)} document(s)")
        st.rerun()


def _display_document_summary():
    """Display summary of attached documents."""
    doc_count = len(st.session_state.custom_gpt_document_ids)
    if doc_count > 0:
        st.info(f"📊 Total documents attached: {doc_count}")
    else:
        st.info("ℹ️ No documents attached. Your GPT will use its general knowledge.")


def _configure_chat_session():
    """Configure session state for chat mode."""
    st.session_state.custom_gpt_mode = True
    st.session_state.custom_gpt_system_prompt = st.session_state.generated_system_prompt
    st.session_state.model_choice = st.session_state.custom_gpt_model
    st.session_state.temperature = st.session_state.custom_gpt_temperature

    if st.session_state.custom_gpt_document_ids:
        _configure_with_documents()
    else:
        _configure_without_documents()

    st.session_state.messages = []
    st.session_state.nav_option = "Chat"


def _configure_with_documents():
    """Configure session state with documents."""
    st.session_state.file_ids = st.session_state.custom_gpt_document_ids.copy()
    st.session_state.multi_file_mode = len(st.session_state.custom_gpt_document_ids) > 1
    st.session_state.file_id = st.session_state.custom_gpt_document_ids[0]
    st.session_state.file_uploaded = True

    # Set session_id from document upload response
    if (
        hasattr(st.session_state, "custom_gpt_session_id")
        and st.session_state.custom_gpt_session_id
    ):
        st.session_state.current_session_id = st.session_state.custom_gpt_session_id
    else:
        # Fallback: generate new session_id if none from upload
        st.session_state.current_session_id = str(uuid.uuid4())

    if "file_names" not in st.session_state:
        st.session_state.file_names = {}
    for file_id, file_name in zip(
        st.session_state.custom_gpt_document_ids,
        st.session_state.custom_gpt_document_names,
    ):
        st.session_state.file_names[file_id] = file_name


def _configure_without_documents():
    """Configure session state without documents."""
    st.session_state.file_ids = []
    st.session_state.multi_file_mode = False
    st.session_state.file_uploaded = False

    # Generate new session_id for custom GPT without documents
    st.session_state.current_session_id = str(uuid.uuid4())


def _display_settings_page():
    """Display Step 8: Settings Configuration."""
    if st.session_state.current_step < 8:
        return

    st.subheader("Configure Your Custom GPT")

    # Show success message once after auto-generating system prompt
    if st.session_state.get("system_prompt_generated_success"):
        st.success("✅ System prompt generated successfully.")
        st.session_state.system_prompt_generated_success = False

    _initialize_settings_defaults()
    _render_gpt_configuration()
    _render_document_section()

    st.markdown("----")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button(
            "Start Chat →",
            use_container_width=True,
            key="start_custom_gpt_chat",
            type="primary",
        ):
            _configure_chat_session()
            st.rerun()


def display_custom_gpt_creator():
    """Display the Custom GPT Creator interface."""
    # Initialize session state
    _initialize_session_state()

    # Display header
    _display_header_and_icon()

    # Step 0: Initial Idea
    _display_initial_idea_step()

    # Phase 2: Deep Dive - Steps 1-6
    if st.session_state.current_step >= 1:
        st.header("Deep Dive")
        user_initial_response = st.session_state.last_initial_response or ""

        # Step 1: Purpose Clarification
        _display_purpose_clarification_step(user_initial_response)

        # Step 2: Audience Understanding
        _display_audience_understanding_step()

        # Step 3: Tone & Style
        _display_tone_style_step(user_initial_response)

        # Step 4: Capabilities Definition
        _display_capabilities_definition_step()

        # Step 5: Knowledge & Context
        _display_knowledge_context_step()

        # Step 6: Examples Collection
        _display_examples_collection_step()

        # Step 7: System Prompt
        _display_system_prompt_step()

        # Step 8: Settings Page
        _display_settings_page()
