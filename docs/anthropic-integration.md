# Anthropic (Claude via Vertex AI) Integration

This document summarizes how Anthropic Claude models are integrated into the RAG PDF API via Google Vertex AI.

## Overview

- **Model access**: Anthropic Claude on Vertex AI using the `anthropic[vertex]` SDK.

- **Handlers**:

  - `rtl_rag_chatbot_api/chatbot/anthropic_handler.py`: Anthropic-specific handler built on shared Vertex AI RAG base.

  - `rtl_rag_chatbot_api/chatbot/utils/vertexai_common.py`: Shared RAG logic for Vertex-backed models.

- **Endpoints**:

  - `POST /file/chat` with `model_choice = "Claude Sonnet 4"` (RAG chat over your uploaded files)

  - `POST /chat/anthropic` for a non-RAG plain LLM response using Anthropic on Vertex AI

  - `GET /available-models` lists available models including Anthropic

## Dependencies

- Declared in `pyproject.toml`:

  - `anthropic = {extras = ["vertex"], version = "^0.66.0"}`

  - `google-cloud-aiplatform` and `langchain-google-vertexai` are also present for broader Vertex AI usage.

## Configuration

- Defined in `configs/app_config.py` under `AnthropicConfig`:

  - `model_sonnet`: default `claude-sonnet-4@20250514`

  - `project`: GCP project (e.g., `dat-itowe-dev`)

  - `location`: GCP region (e.g., `europe-west1`)

- The application references these via `configs.anthropic`.

## Handler: Anthropic RAG

File: `rtl_rag_chatbot_api/chatbot/anthropic_handler.py`

- Inherits `VertexAIRAGHandler` to reuse indexing, retrieval, and prompt construction logic.

- Initializes an `AnthropicVertex` client using `configs.anthropic.project` and `configs.anthropic.location`.

- Maps `model_choice` → concrete model ID via config (`"Claude Sonnet 4"` → `configs.anthropic.model_sonnet`).

- Private `_call_model(prompt: str)` issues `client.messages.create(...)` with `max_tokens` and `temperature`.

## Non-RAG Helper

`get_anthropic_non_rag_response(config, prompt, model_choice, temperature=0.6, max_tokens=4096)`

- Initializes Vertex AI (`vertexai.init`) and `AnthropicVertex`.

- Builds a simple system+user prompt and returns the first text block from the response.

## CSV/Tabular Support

`rtl_rag_chatbot_api/chatbot/csv_handler.py`

- Uses `ChatAnthropicVertex` when `model_choice == "Claude Sonnet 4"`, leveraging the same config values.

## API Usage

### RAG chat over uploaded files

Endpoint: `POST /file/chat`

Request JSON example:

```json
{
  "text": ["How many mangoes are there in Garden"],
  "file_id": "<your-file-id>",
  "session_id": "<your-session-id>",
  "model_choice": "Claude Sonnet 4",
  "user_id": "<your-username>"
}
```

Notes:

- `model_choice` must be exactly `"Claude Sonnet 4"` for Anthropic usage.

- The system will route to `AnthropicHandler` and use RAG context built from your uploaded document(s).

### Plain (non-RAG) Anthropic response

Endpoint: `POST /chat/anthropic`

Request JSON example:

```json
{
  "message": "Explain RAG in simple terms",
  "model": "Claude Sonnet 4",
  "temperature": 0.8
}
```

Notes:

- This endpoint does not use embeddings/context, it is a direct model call.

- The server validates `model == "Claude Sonnet 4"`.

### Discover available models

Endpoint: `GET /available-models`

Response includes `"Claude Sonnet 4"` in the `models` array and under `model_types.text`.

## Environment and Auth

- Ensure your environment has permission to call Vertex AI Model Garden Anthropic models in the specified `project` and `location`.

- Typical auth uses Application Default Credentials (ADC). When running locally, authenticate with `gcloud auth application-default login`.

## Operational Parameters

- Default `max_tokens` used by the handler: `4096`.

- `temperature` is configurable per request; defaults are set in handler usage.

## Troubleshooting

- 400 error on `/chat/anthropic`: verify `model` is exactly `"Claude Sonnet 4"`.

- Permission errors: confirm Vertex AI access to Anthropic models and correct `project`/`location`.

- Empty outputs: handler falls back to `model_dump_json`; check response for details.

## Related Files

- `rtl_rag_chatbot_api/chatbot/anthropic_handler.py`

- `rtl_rag_chatbot_api/chatbot/utils/vertexai_common.py`

- `rtl_rag_chatbot_api/chatbot/model_handler.py`

- `rtl_rag_chatbot_api/chatbot/csv_handler.py`

- `rtl_rag_chatbot_api/app.py` (endpoints `/file/chat`, `/chat/anthropic`, `/available-models`)

- `configs/app_config.py` (`AnthropicConfig`)
