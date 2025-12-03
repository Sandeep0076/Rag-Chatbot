import os
import sys
from pathlib import Path

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env file
# Get the project root directory (3 levels up from this test file)
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)

# Get environment variables for GPT 5.1
# Following the pattern: AZURE_LLM__MODELS__GPT_5_1__*
endpoint = os.getenv("AZURE_LLM__MODELS__GPT_5_1__ENDPOINT") or os.getenv(
    "ENDPOINT_URL"
)
deployment = os.getenv("AZURE_LLM__MODELS__GPT_5_1__DEPLOYMENT") or os.getenv(
    "DEPLOYMENT_NAME"
)
api_version = os.getenv("AZURE_LLM__MODELS__GPT_5_1__API_VERSION", "2025-01-01-preview")
api_key = os.getenv("AZURE_LLM__MODELS__GPT_5_1__API_KEY") or os.getenv("API_KEY")

# Validate required environment variables
if not endpoint:
    print(
        "❌ Error: ENDPOINT_URL or AZURE_LLM__MODELS__GPT_5_1__ENDPOINT not found in environment variables"
    )
    sys.exit(1)

if not deployment:
    print(
        "❌ Error: DEPLOYMENT_NAME or AZURE_LLM__MODELS__GPT_5_1__DEPLOYMENT not found in environment variables"
    )
    sys.exit(1)

print(f"✅ Using endpoint: {endpoint}")
print(f"✅ Using deployment: {deployment}")
print(f"✅ Using API version: {api_version}")

# Initialize Azure OpenAI client
# Prefer API key authentication if available, otherwise use Entra ID
if api_key:
    print("🔑 Using API key authentication")
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )
else:
    print("🔐 Using Entra ID authentication (DefaultAzureCredential)")
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
    )


# IMAGE_PATH = "YOUR_IMAGE_PATH"
# encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')
chat_prompt = [
    {
        "role": "developer",
        "content": [
            {
                "type": "text",
                "text": "You are an AI assistant that helps people find information.",
            }
        ],
    }
]

# Include speech result if speech is enabled
messages = chat_prompt

print("\n🚀 Sending request to GPT 5.1...")
print("=" * 50)

try:
    completion = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_completion_tokens=16384,
        stop=None,
        stream=False,
    )

    print("\n✅ Request successful!")
    print("=" * 50)
    print("\n📄 Response JSON:")
    print(completion.to_json())

    # Also print the content if available
    if completion.choices and len(completion.choices) > 0:
        message = completion.choices[0].message
        if message and message.content:
            print("\n" + "=" * 50)
            print("💬 Response Content:")
            print("=" * 50)
            print(message.content)

except Exception as e:
    print(f"\n❌ Error occurred: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
