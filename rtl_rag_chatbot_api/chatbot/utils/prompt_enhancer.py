"""
Prompt enhancement utility for improving user prompts before image generation.
"""
from rtl_rag_chatbot_api.chatbot.utils.language_detector import detect_lang
from rtl_rag_chatbot_api.common.prompts_storage import ENHANCE_PROMPT


def enhance_prompt(user_prompt: str, azure_client_callback) -> dict:
    """
    Enhance a user prompt for better image generation results.

    Args:
        user_prompt: The original user prompt to enhance
        azure_client_callback: Callback function to call Azure GPT-4o-mini
                              (e.g., get_azure_non_rag_response)

    Returns:
        dict: Dictionary containing:
            - original_prompt: The original user prompt
            - enhanced_prompt: The enhanced version
            - language: Detected language of the prompt
    """
    # Detect language of the user prompt
    detected_language = detect_lang(user_prompt)

    # Construct the enhancement prompt with detected language
    enhancement_prompt = ENHANCE_PROMPT.replace("{LANGUAGE}", detected_language)
    enhancement_prompt = enhancement_prompt.replace("{USER_PROMPT}", user_prompt)

    # Call GPT-4o-mini to enhance the prompt
    enhanced_text = azure_client_callback(enhancement_prompt)

    return {
        "original_prompt": user_prompt,
        "enhanced_prompt": enhanced_text.strip(),
        "language": detected_language,
    }
