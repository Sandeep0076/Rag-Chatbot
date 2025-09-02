import os

from openai import AzureOpenAI

MODEL_NAME = "chatbotui-openai-gpt-5"

endpoint = os.getenv(
    "ENDPOINT_URL", "https://swedencentral.api.cognitive.microsoft.com/"
)
deployment = os.getenv("DEPLOYMENT_NAME", MODEL_NAME)


client = AzureOpenAI(
    azure_endpoint=endpoint,
    # azure_deployment=deployment,
    api_version="2025-04-01-preview",
    api_key=os.getenv("AZURE_OPENAI_API_KEY", "..."),
    default_headers={
        "x-ms-model-mesh-model-name": MODEL_NAME,
        # "x-ms-model-mesh-model-name": f"{MODEL_NAME}-mini"
    },
)


chat_prompt = [
    {"role": "system", "content": "You are a lawyer in Germany"},
    {
        "role": "user",
        "content": "Explain German constitution first five articles in detail in English",
    },
]

# Include speech result if speech is enabled
messages = chat_prompt

completion = client.chat.completions.create(
    model=deployment,
    messages=messages,
    max_completion_tokens=100000,
    presence_penalty=0,
    stop=None,
    #
    reasoning_effort="minimal",
    # verbosity="low",
    #
    stream=True,
)

for chunk in completion:
    if len(chunk.choices) > 0:
        print(chunk.choices[0].delta.content, end="")

# print(completion.to_json())
