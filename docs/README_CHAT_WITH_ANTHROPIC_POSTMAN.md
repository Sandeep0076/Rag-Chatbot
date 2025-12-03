# Postman Instructions for `/chat/anthropic` Endpoint

## Endpoint Overview

**Endpoint**: `POST /chat/anthropic`

**Description**: Get a non-RAG response from an Anthropic (Vertex) model using Claude Sonnet 4.

**Response Type**: Plain text (not JSON)

---

## Step-by-Step Postman Setup

### 1. Create a New Request

1. Open Postman
2. Click **New** → **HTTP Request**
3. Set the request method to **POST**

### 2. Set the Request URL

**Base URL**: `http://localhost:8080` (for local development)
**Full Endpoint**: `http://localhost:8080/chat/anthropic`

**Note**: Replace `localhost:8080` with your actual server URL if testing against a deployed instance.

### 3. Configure Headers

Go to the **Headers** tab and add:

| Key | Value |
|-----|-------|
| `Content-Type` | `application/json` |
| `Authorization` | `Bearer YOUR_TOKEN_HERE` |

**Important**: Replace `YOUR_TOKEN_HERE` with your actual OAuth Bearer token.

### 4. Configure Request Body

1. Go to the **Body** tab
2. Select **raw**
3. Select **JSON** from the dropdown
4. Enter the following JSON structure:

```json
{
  "model": "claude-sonnet-4@20250514",
  "message": "Explain the concept of machine learning in simple terms."
}
```

**Alternative Model Option**:
```json
{
  "model": "claude-sonnet-4-5",
  "message": "Explain the concept of machine learning in simple terms."
}
```

### 5. Request Parameters

| Parameter | Type | Required | Description | Valid Values |
|-----------|------|----------|-------------|--------------|
| `model` | string | ✅ Yes | The Anthropic model to use | `"claude-sonnet-4@20250514"` or `"claude-sonnet-4-5"` |
| `message` | string | ✅ Yes | The prompt/question to send to the model | Any text string |

**Note**: The `temperature` parameter is mentioned in the endpoint documentation but is not currently part of the `ChatRequest` model. If you need temperature control, you may need to update the model definition. The endpoint defaults to `0.8` if temperature is not provided.

### 6. Example Request Body

**Basic Example**:
```json
{
  "model": "claude-sonnet-4@20250514",
  "message": "What is artificial intelligence?"
}
```

**Complex Question Example**:
```json
{
  "model": "claude-sonnet-4@20250514",
  "message": "Explain the differences between supervised and unsupervised learning. Provide examples of each."
}
```

**Code Explanation Example**:
```json
{
  "model": "claude-sonnet-4-5",
  "message": "Explain how gradient descent works in neural networks. Include a simple example."
}
```

### 7. Send the Request

1. Click the **Send** button
2. Wait for the response (typically 5-15 seconds depending on complexity)

### 8. Expected Response

**Success Response** (Status: 200 OK):
- **Response Type**: Plain text
- **Content**: The model's generated response as a plain text string

**Example Response**:
```
Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data, make predictions, and improve their performance over time.

There are three main types of machine learning:
1. Supervised learning: Uses labeled data to train models
2. Unsupervised learning: Finds patterns in unlabeled data
3. Reinforcement learning: Learns through trial and error with rewards
```

**Error Responses**:

**400 Bad Request** - Invalid model:
```json
{
  "detail": "Invalid model choice. Use 'claude-sonnet-4@20250514' or 'claude-sonnet-4-5'."
}
```

**401 Unauthorized** - Missing or invalid token:
```json
{
  "detail": "Not authenticated"
}
```

**500 Internal Server Error**:
```json
{
  "detail": "An error occurred: [error message]"
}
```

---

## Complete Postman Collection JSON

You can import this into Postman:

```json
{
  "info": {
    "name": "Anthropic Chat",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Chat with Anthropic - Claude Sonnet 4",
      "request": {
        "method": "POST",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          },
          {
            "key": "Authorization",
            "value": "Bearer YOUR_TOKEN_HERE",
            "type": "text"
          }
        ],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"model\": \"claude-sonnet-4@20250514\",\n  \"message\": \"Explain the concept of machine learning in simple terms.\"\n}"
        },
        "url": {
          "raw": "http://localhost:8080/chat/anthropic",
          "protocol": "http",
          "host": ["localhost"],
          "port": "8080",
          "path": ["chat", "anthropic"]
        }
      }
    }
  ]
}
```

---

## Troubleshooting

### Common Issues

1. **401 Unauthorized Error**
   - Verify your Bearer token is valid and correctly formatted
   - Ensure the token hasn't expired
   - Check that the `Authorization` header is set correctly: `Bearer <token>`

2. **400 Bad Request - Invalid Model**
   - Ensure you're using exactly: `"claude-sonnet-4@20250514"` or `"claude-sonnet-4-5"`
   - Check for typos in the model name

3. **Connection Refused**
   - Verify the API server is running
   - Check the URL is correct (default: `http://localhost:8080`)
   - Ensure firewall/network settings allow the connection

4. **500 Internal Server Error**
   - Check server logs for detailed error messages
   - Verify Anthropic/Vertex AI credentials are configured correctly
   - Ensure the model is available in your region/project

### Testing Tips

1. **Start Simple**: Begin with a basic question to verify the endpoint works
2. **Check Response Time**: Non-RAG responses are typically faster than RAG responses
3. **Compare Models**: Try both model options to see differences in responses
4. **Monitor Rate Limits**: Be aware of API rate limits for Anthropic models

---

## Additional Notes

- This endpoint provides **non-RAG** responses (no document context)
- The response is returned as **plain text**, not JSON
- Maximum tokens: 4096 (hardcoded in the endpoint)
- Default temperature: 0.8 (if temperature support is added to the model)
