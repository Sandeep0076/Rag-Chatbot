#!/usr/bin/env python3
"""
A minimal, non-streaming test for any configured Azure model to get a single answer.
"""
import argparse
import os
import sys

from openai import AzureOpenAI

from configs.app_config import Config

# Add the project root to the path to allow running from the command line
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# def run_simple_test(model_key: str):
#     """Initializes config, asks one question to the specified model, and prints the answer."""
#     try:
#         config = Config()
#         print(f"✅ Configuration loaded successfully. Targeting model key: '{model_key}'")

#         # Find the specified model configuration
#         model_config = config.azure_llm.models.get(model_key)

#         if not model_config:
#             print(f"❌ Could not find model key '{model_key}' in your configuration.")
#             print(f"Available models are: {list(config.azure_llm.models.keys())}")
#             return False

#         print(f"✅ Found config for deployment: '{model_config.deployment}'")

#         client = AzureOpenAI(
#             api_key=model_config.api_key,
#             azure_endpoint=model_config.endpoint,
#             api_version=model_config.api_version,
#         )

#         question = "tell me all the Prime Minister of India till date by year"
#         print(f"❓ Asking question: \"{question}\"")

#         # Use max_completion_tokens for newer models as seen in your codebase
#         use_max_completion_tokens = any(term in model_key.lower() for term in ["o3", "o4", "gpt-5", "gpt_5"])

#         request_params = {
#             "model": model_config.deployment,
#             "messages": [
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": question},
#             ],
#         }

#         if use_max_completion_tokens:
#             request_params["max_completion_tokens"] = 2048
#             print("🔧 Using 'max_completion_tokens' parameter.")
#         else:
#             request_params["max_tokens"] = 2048
#             request_params["temperature"] = 0.3
#             print("🔧 Using 'max_tokens' parameter.")

#         response = client.chat.completions.create(**request_params)

#         print("\n" + "="*20 + " RAW API RESPONSE " + "="*20)
#         try:
#             print(f"Raw Response Object:\n{response}\n")
#             if hasattr(response, "to_dict"):
#                 print(f"Response as Dict:\n{response.to_dict()}\n")
#         except Exception as e:
#             print(f"Could not print full response object: {e}")
#         print("="*58 + "\n")

#         # Safely extract the answer
#         answer = None
#         if response.choices and len(response.choices) > 0:
#             message = response.choices[0].message
#             if message and message.content:
#                 answer = message.content.strip()

#         if answer:
#             print("ANSWER:")
#             print(answer)
#             return True
#         else:
#             print("❌ The API call succeeded, but the response content is empty.")
#             return False

#     except Exception as e:
#         print(f"❌ An error occurred: {e}")
#         return False
# #


def minimal_test(model_key: str, question: str) -> int:
    """
    Loads configuration and runs a simple test against a specified model
    using Chat Completions with reasoning_effort.
    """
    # Load configuration to get credentials
    try:
        config = Config()
        model_config = config.azure_llm.models.get(model_key)
        if not model_config:
            print(f"❌ Could not find model key '{model_key}' in your configuration.")
            print(f"Available models are: {list(config.azure_llm.models.keys())}")
            return 1

        api_key = model_config.api_key
        endpoint = model_config.endpoint
        deployment = model_config.deployment
        api_version = model_config.api_version
        print(f"✅ Config loaded for model '{model_key}'.")

    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return 1

    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )

    print(f'❓ Asking question: "{question}"')
    try:
        # Chat Completions with reasoning effort control, like the curl in the screenshot
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            reasoning_effort="minimal",
            # reasoning_effort="low",
            verbosity="low",
            max_completion_tokens=1024,
        )
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return 2

    # Extract answer
    answer = None
    try:
        if response.choices and len(response.choices) > 0:
            message = response.choices[0].message
            if message and message.content:
                answer = message.content.strip()
    except Exception:
        answer = None

    if answer:
        print("\nANSWER:")
        print(answer)
        return 0

    print("\n❌ The API call succeeded, but the response content is empty.")
    print("\n" + "=" * 20 + " RAW API RESPONSE " + "=" * 20)
    try:
        if hasattr(response, "to_dict"):
            print(response.to_dict())
        else:
            print(response)
    except Exception:
        print("(Could not display raw response)")
    return 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test an Azure OpenAI model using the Responses API."
    )
    parser.add_argument(
        "model_key",
        nargs="?",
        default="gpt_5_mini",
        help="The model key from your configuration (e.g., 'gpt_4o_mini', 'gpt_5_mini').",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="Explain German constitution  first five articles in detail in English. ",
        # default=" List me the names of the Prime Minister of India till date according to years?",
        help="The question to ask the model.",
    )
    args = parser.parse_args()

    print(f"🚀 Starting Minimal Test for '{args.model_key}'...")
    print("========================================\n")
    exit_code = minimal_test(args.model_key, args.question)
    print("\n========================================")
    if exit_code == 0:
        print("🎉 Test finished successfully.")
    else:
        print("⚠️  Test finished with issues.")
    sys.exit(exit_code)
