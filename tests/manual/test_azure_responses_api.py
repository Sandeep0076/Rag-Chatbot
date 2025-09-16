# import os

# from openai import OpenAI

# # Your environment variables (which you already have)
# api_key = os.getenv("AZURE_LLM__MODELS__GPT_4_1__API_KEY")
# endpoint = os.getenv(
#     "AZURE_LLM__MODELS__GPT_4_1__ENDPOINT", "https://chatbotui-openai.openai.azure.com/"
# )
# deployment = os.getenv(
#     "AZURE_LLM__MODELS__GPT_4_1__DEPLOYMENT", "chatbotui-openai-gpt-4.1"
# )
# api_version = os.getenv("AZURE_LLM__MODELS__GPT_4_1__API_VERSION", "2025-01-01-preview")

# # # Ensure the endpoint ends with a slash and add the v1 path
# # if not endpoint.endswith('/'):
# #     endpoint += '/'
# # base_url = f"{endpoint}openai/v1/"

# # # Initialize the client
# # client = AzureOpenAI(
# #     base_url=base_url,
# #     api_key=api_key,
# #     api_version="preview"  # Use "preview" for the v1 API
# # )

# # # Test the Responses API
# # try:
# #     response = client.responses.create(
# #         model=deployment,
# #         input=[{
# #             "role": "user",
# #             "content": "What is the difference between the Responses API and Chat Completions API?"
# #         }],
# #         max_output_tokens=512,
# #         temperature=0.7
# #     )

# #     print(response.to_json())

# # except Exception as e:
# #     print(f"Error: {e}")

# client = OpenAI(
#     api_key=api_key, base_url="https://chatbotui-openai.openai.azure.com/openai/v1/"
# )

# response = client.responses.create(
#     model=deployment,
#     input="This is a test.",
# )

# print(response.model_dump_json(indent=2))
