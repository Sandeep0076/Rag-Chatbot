from openai import AzureOpenAI

from configs.app_config import Config

# Load configuration
configs = Config()

# Use configuration instead of hardcoded values
# Try to use gpt_5 first, then fall back to any available model
MODEL_NAME = "gpt_5"  # This should match a key in configs.azure_llm.models
model_config = configs.azure_llm.models.get(MODEL_NAME)


print(f"Using model: {MODEL_NAME}")

client = AzureOpenAI(
    azure_endpoint=model_config.endpoint,
    api_version=model_config.api_version,
    api_key=model_config.api_key,
    # default_headers={
    #     "x-ms-model-mesh-model-name": model_config.model_name,
    # },
)

chat_prompt = [
    {"role": "system", "content": "You are a lawyer in Germany"},
    {
        "role": "user",
        "content": "Which all countries USA was involved in Rejime change",
    },
]

# Include speech result if speech is enabled
messages = chat_prompt

# Use configuration for hyperparameters
# Prepare parameters based on model type
completion_params = {
    "model": model_config.deployment,
    "messages": messages,
    "max_completion_tokens": configs.llm_hyperparams.max_tokens,
    "presence_penalty": configs.llm_hyperparams.presence_penalty,
    "reasoning_effort": "minimal",
    "verbosity": "medium",  # gpt_4o_mini only supports "medium", not "low"
    "stream": False,
}

# Only add stop parameter for models that support it (not GPT-5)
if MODEL_NAME not in ["gpt_5", "gpt_5_mini"]:
    completion_params["stop"] = configs.llm_hyperparams.stop

completion = client.chat.completions.create(**completion_params)

# Handle both streaming and non-streaming responses
if completion_params.get("stream", False):
    # Streaming response
    for chunk in completion:
        if len(chunk.choices) > 0:
            print(chunk.choices[0].delta.content, end="")
else:
    # Non-streaming response
    if len(completion.choices) > 0:
        print(completion.choices[0].message.content)

# print(completion.to_json())
