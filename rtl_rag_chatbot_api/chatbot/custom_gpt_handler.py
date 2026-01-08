"""
Custom GPT handler for Azure API calls.
Handles Custom GPT mode requests and logging.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from rtl_rag_chatbot_api.chatbot.chatbot_creator import get_azure_non_rag_response


def _log_custom_gpt_api_call(
    step_name: str,
    question: str,
    user_input: str,
    resolved_answer: str,
    api_response: str,
):
    """
    Log Custom GPT API call details.

    Args:
        step_name (str): Name of the step
        question (str): Question that was asked
        user_input (str): User's raw input
        resolved_answer (str): Resolved answer after processing
        api_response (str): API response
    """
    logging.info(f"[Custom GPT - {step_name}] Question: {question}")
    logging.info(f"[Custom GPT - {step_name}] User Input: {user_input}")
    logging.info(f"[Custom GPT - {step_name}] Resolved Answer: {resolved_answer}")
    logging.info(
        f"[Custom GPT - {step_name}] API Response: {api_response[:500]}..."
        if len(api_response) > 500
        else f"[Custom GPT - {step_name}] API Response: {api_response}"
    )


def _log_answer_resolution(
    step_name: str,
    question: str,
    examples: List[str],
    user_input: str,
    resolved_answer: str,
):
    """
    Log answer resolution API call details.

    Args:
        step_name (str): Name of the step
        question (str): Question that was asked
        examples (list): Examples provided to user
        user_input (str): User's raw input
        resolved_answer (str): Resolved answer
    """
    examples_str = ", ".join(examples) if examples else "No examples"
    logging.info(f"[Answer Resolution - {step_name}] Question: {question}")
    logging.info(f"[Answer Resolution - {step_name}] Examples: {examples_str}")
    logging.info(f"[Answer Resolution - {step_name}] User Input: {user_input}")
    logging.info(
        f"[Answer Resolution - {step_name}] Resolved Answer: {resolved_answer}"
    )


def _extract_answer_resolution_info(message: str) -> Dict[str, Any]:
    """
    Extract question, examples, and user input from answer resolution prompt.

    Args:
        message (str): The prompt message containing answer resolution format

    Returns:
        dict: Dictionary with 'question', 'user_input', and 'examples' keys
    """
    question = "Unknown"
    user_input = "Unknown"
    examples = []

    if not message:
        return {"question": question, "user_input": user_input, "examples": examples}

    # Extract question
    if "Question asked:" in message:
        try:
            question = (
                message.split("Question asked:")[1].split("\n")[0].strip().strip('"')
            )
        except Exception:
            pass

    # Extract examples
    if "Examples provided to the user:" in message:
        try:
            examples_section = (
                message.split("Examples provided to the user:")[1]
                .split("User's Answer:")[0]
                .strip()
            )
            examples = [
                ex.strip("- ").strip()
                for ex in examples_section.split("\n")
                if ex.strip().startswith("-")
            ]
        except Exception:
            pass

    # Extract user input
    if "User's Answer:" in message:
        try:
            user_input = message.split("User's Answer:")[1].strip().strip('"')
        except Exception:
            pass

    return {"question": question, "user_input": user_input, "examples": examples}


def _extract_question_from_json_response(answer: str) -> str:
    """
    Extract question from JSON response in Custom GPT step.

    Args:
        answer (str): The API response that may contain JSON

    Returns:
        str: Extracted question or "Unknown"
    """
    question = "Unknown"
    try:
        # Clean response if it has markdown code blocks
        cleaned_response = answer.strip()
        if "```json" in cleaned_response:
            cleaned_response = (
                cleaned_response.split("```json")[1].split("```")[0].strip()
            )
        elif "```" in cleaned_response:
            cleaned_response = cleaned_response.split("```")[1].split("```")[0].strip()

        response_data = json.loads(cleaned_response)
        question = response_data.get("question", "Unknown")
    except Exception:
        pass

    return question


def _build_conversation_prompt(
    text: List[str], system_prompt: Optional[str] = None
) -> str:
    """
    Build conversation prompt from text array and optional system prompt.

    Args:
        text (List[str]): Conversation history
        system_prompt (str, optional): Custom system prompt

    Returns:
        str: Formatted prompt
    """
    current_question = text[-1]
    if len(text) > 1:
        previous_messages = "\n".join([f"Previous message: {msg}" for msg in text[:-1]])
        prompt = f"{previous_messages}\nCurrent question: {current_question}"
    else:
        prompt = current_question

    # Add system prompt if provided
    if system_prompt:
        prompt = f"System Instructions: {system_prompt}\n\n{prompt}"

    return prompt


def handle_custom_gpt_azure_request(request, configs) -> JSONResponse:
    """
    Handle Custom GPT mode Azure API request.

    Args:
        request: ChatRequest with Custom GPT fields
        configs: Configuration object

    Returns:
        JSONResponse: Response with answer and session_id
    """
    available_azure_models = list(configs.azure_llm.models.keys())
    if request.model_choice not in available_azure_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model choice. Available Azure models: {available_azure_models}",
        )

    # Build conversation context from text array
    prompt = _build_conversation_prompt(request.text, request.system_prompt)

    answer = get_azure_non_rag_response(
        configs=configs,
        query=prompt,
        model_choice=request.model_choice,
        max_tokens=None,
        temperature=request.temperature,
    )

    # Return JSON response with session info for Custom GPT
    return JSONResponse(
        content={
            "response": answer,
            "session_id": request.session_id or str(uuid.uuid4()),
        }
    )


def handle_custom_gpt_step_request(request, configs) -> PlainTextResponse:
    """
    Handle Custom GPT creation step request with logging.

    Args:
        request: ChatRequest with step_name
        configs: Configuration object

    Returns:
        PlainTextResponse: Response with answer
    """
    available_azure_models = list(configs.azure_llm.models.keys())
    if request.model not in available_azure_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model choice. Available Azure models: {available_azure_models}",
        )

    # Check if this is an answer resolution call
    is_answer_resolution = (
        request.step_name and "answer_resolution" in request.step_name
    )

    answer = get_azure_non_rag_response(
        configs=configs,
        query=request.message,
        model_choice=request.model,
        max_tokens=None,
        temperature=request.temperature,
    )

    # Log based on step type
    if is_answer_resolution:
        # Extract information from ANSWER_RESOLUTION_PROMPT format
        info = _extract_answer_resolution_info(request.message)
        _log_answer_resolution(
            step_name=request.step_name,
            question=info["question"],
            examples=info["examples"],
            user_input=info["user_input"],
            resolved_answer=answer,
        )
    else:
        # For regular Custom GPT steps, extract question from JSON response
        question = _extract_question_from_json_response(answer)
        _log_custom_gpt_api_call(
            step_name=request.step_name,
            question=question,
            user_input="N/A",
            resolved_answer="N/A",
            api_response=answer,
        )

    return PlainTextResponse(answer)


def get_tone_options(language: str = "en") -> dict:
    """
    Get predefined tone and style options for custom GPT creation.

    Args:
        language: Language code for internationalization (default: "en")

    Returns:
        dict: Question and examples in the format:
            {
                "question": "How should your GPT communicate?",
                "examples": [
                    "Professional & Concise - Clear, direct, and to-the-point. No fluff.",
                    "Friendly & Encouraging - Warm, supportive, and patient like a helpful mentor.",
                    "Casual & Conversational - Relaxed, easy-going, like chatting with a colleague.",
                    "Technical & Detailed - Precise, thorough, with technical depth and accuracy.",
                    "Witty & Engaging - Smart humor, personality-driven, keeps things interesting."
                ]
            }
    """
    return {
        "question": "How should your GPT communicate?",
        "examples": [
            "Professional & Concise - Clear, direct, and to-the-point. No fluff.",
            "Friendly & Encouraging - Warm, supportive, and patient like a helpful mentor.",
            "Casual & Conversational - Relaxed, easy-going, like chatting with a colleague.",
            "Technical & Detailed - Precise, thorough, with technical depth and accuracy.",
            "Witty & Engaging - Smart humor, personality-driven, keeps things interesting.",
        ],
    }
