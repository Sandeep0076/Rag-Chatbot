a# Chat with Multiple URLs - Postman Step-by-Step Guide

## Overview

This guide provides detailed step-by-step instructions for using Postman to chat with multiple URLs using the RTL RAG Chatbot API. The system can process multiple URLs simultaneously, extract content from each URL, create embeddings, and allow you to chat with the combined content from all URLs.

## Prerequisites

- Postman installed on your machine
- Valid authentication token for the API
- API base URL (e.g., `http://your-api-domain` or `http://localhost:8000`)
- One or more valid URLs to process

## Step-by-Step Process

### Step 1: Process Multiple URLs

#### 1.1 Create New Request in Postman
1. Open Postman
2. Click "New" → "Request"
3. Name your request: "Upload Multiple URLs"
4. Select or create a collection to save it

#### 1.2 Configure the Upload Request
1. **Set HTTP Method**: Change from GET to **POST**
2. **Enter URL**: `{{base_url}}/file/upload`
   - Replace `{{base_url}}` with your actual API URL
   - Example: `http://localhost:8000/file/upload`

#### 1.3 Set Headers
1. Go to the **Headers** tab
2. Add header:
   - **Key**: `Authorization`
   - **Value**: `Bearer your-auth-token-here`
   - Replace `your-auth-token-here` with your actual token

#### 1.4 Configure Request Body
1. Go to the **Body** tab
2. Select **form-data**
3. Add the following key-value pairs:

| Key | Type | Value | Required | Description |
|-----|------|-------|----------|-------------|
| `username` | Text | `your-username` | ✅ | Username for tracking |
| `urls` | Text | Multiple URLs (see format below) | ✅ | URLs to process |
| `is_image` | Text | `false` | ❌ | Set to false for URL content |

#### 1.5 URL Format Options
You can provide multiple URLs in two formats:

**Option A: Comma-separated**
```
https://example1.com, https://example2.com, https://example3.com
```

**Option B: Newline-separated**
```
https://example1.com
https://example2.com
https://example3.com
```

**Example URLs for testing:**
```
https://en.wikipedia.org/wiki/Artificial_intelligence
https://en.wikipedia.org/wiki/Machine_learning
https://en.wikipedia.org/wiki/Natural_language_processing
```

#### 1.6 Send the Request
1. Click **Send** button
2. Wait for the response (may take 30-60 seconds for multiple URLs)

#### 1.7 Expected Response
```json
{
    "file_id": "uuid-string-1",
    "file_ids": [
        "uuid-string-1",
        "uuid-string-2",
        "uuid-string-3"
    ],
    "multi_file_mode": true,
    "message": "Processed all 3 URLs successfully",
    "status": "success",
    "original_filename": "url_content.txt",
    "is_image": false,
    "is_tabular": false,
    "session_id": "uuid-session-string"
}
```

**Important**: Save the `file_ids` array - you'll need it for the chat step!

### Step 2: Check Embedding Status (Optional)

Before chatting, you can verify that embeddings are ready:

#### 2.1 Create Status Check Request
1. Create new request: "Check Embedding Status"
2. **Method**: GET
3. **URL**: `{{base_url}}/embeddings/status/{{file_id}}`
   - Replace `{{file_id}}` with one of the file IDs from step 1

#### 2.2 Send Request
Expected response when ready:
```json
{
    "status": "ready_for_chat",
    "can_chat": true,
    "file_id": "uuid-string",
    "message": "Embeddings are ready for chat"
}
```

### Step 3: Chat with Multiple URLs

#### 3.1 Create Chat Request
1. Create new request: "Chat with Multiple URLs"
2. **Method**: POST
3. **URL**: `{{base_url}}/file/chat`

#### 3.2 Set Headers
1. **Authorization**: `Bearer your-auth-token-here`
2. **Content-Type**: `application/json`

#### 3.3 Configure Request Body
1. Go to **Body** tab
2. Select **raw** and **JSON**
3. Enter the following JSON structure:

```json
{
    "text": ["What are the main differences between artificial intelligence, machine learning, and natural language processing?"],
    "file_ids": [
        "uuid-string-1",
        "uuid-string-2",
        "uuid-string-3"
    ],
    "model_choice": "gpt_4o_mini",
    "user_id": "your-user-id",
    "session_id": "uuid-session-string",
    "temperature": 0.7
}
```

#### 3.4 Request Parameters Explained

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `text` | ✅ | Array with your question as the last element | `["What is AI?"]` |
| `file_ids` | ✅ | Array of file IDs from Step 1 | `["uuid1", "uuid2"]` |
| `model_choice` | ✅ | AI model to use | `"gpt_4o_mini"` |
| `user_id` | ✅ | Unique user identifier | `"user123"` |
| `temperature` | ❌ | Response creativity (0.0-2.0) | `0.7` |
| `session_id` | ✅ | Session tracking | `"session-uuid"` |

#### 3.5 Available Models
- `gpt_4o_mini` (recommended for most cases)
- `gpt_4o`
- `gemini-2.5-flash`
- `gemini-2.5-pro`

#### 3.6 Send Chat Request
1. Click **Send**
2. Wait for response (typically 5-15 seconds)

#### 3.7 Expected Chat Response
```json
{
    "response": "Based on the content from the provided URLs, here are the main differences between artificial intelligence, machine learning, and natural language processing:\n\n**Artificial Intelligence (AI)**:\n- Broad field focused on creating machines that can perform tasks requiring human intelligence...",
    "is_table": false,
    "sources": [
        "uuid-string-1",
        "uuid-string-2",
        "uuid-string-3"
    ]
}
```

## Advanced Usage

### Multi-turn Conversation
For follow-up questions, include conversation history in the `text` array with alternating user messages and AI answers:

```json
{
    "text": [
        "What are the main differences between AI, ML, and NLP?",
        "AI is the broad field of creating intelligent machines, ML is a subset that learns from data, and NLP focuses on language understanding.",
        "Can you provide examples of real-world applications for each?"
    ],
    "file_ids": ["uuid1", "uuid2", "uuid3"],
    "model_choice": "gpt_4o_mini",
    "user_id": "user123",
    "session_id": "uuid-session-string"
}
```

### Temperature Settings Guide
- **0.0-0.3**: Highly focused, deterministic responses
- **0.4-0.7**: Balanced creativity and accuracy (recommended)
- **0.8-2.0**: More creative but potentially less accurate

### Error Handling

#### Common Error Responses

**Invalid URLs:**
```json
{
    "status": "error",
    "message": "No valid URLs provided"
}
```

**Processing Failed:**
```json
{
    "status": "error",
    "message": "Failed to process all URLs"
}
```

**Chat Error:**
```json
{
    "detail": "No file context available for chat"
}
```

## Complete Postman Collection Example

You can create a Postman collection with these three requests:

### Collection Structure:
```
📁 RAG Multi-URL Chat
├── 📝 1. Upload Multiple URLs
├── 📝 2. Check Embedding Status  
└── 📝 3. Chat with URLs
```

### Environment Variables (Optional):
Create a Postman environment with:
- `base_url`: `http://localhost:8000`
- `auth_token`: `your-actual-token`
- `file_ids`: `["uuid1","uuid2","uuid3"]`

## Tips for Success

1. **URL Selection**: Choose URLs with substantial text content
2. **Wait for Processing**: Allow 30-60 seconds for URL processing
3. **Check Status**: Verify embeddings are ready before chatting
4. **Save File IDs**: Always save the `file_ids` from the upload response
5. **Clear Questions**: Ask specific questions that can be answered from the URL content
6. **Model Choice**: Use `gpt_4o_mini` for most use cases (faster and cost-effective)

## Troubleshooting

### Issue: "No valid URLs provided"
- **Solution**: Check URL format and ensure URLs are accessible

### Issue: "Embeddings not ready"
- **Solution**: Wait longer or check embedding status endpoint

### Issue: "No file context available"
- **Solution**: Verify you're using the correct `file_ids` from the upload response

### Issue: Timeout during upload
- **Solution**: Try fewer URLs or check URL accessibility

## Security Notes

- Keep your authentication token secure
- Don't share file IDs publicly
- URLs should be publicly accessible (no authentication required)
- The system will attempt to extract text content from the provided URLs

---

This guide enables you to effectively use the multi-URL chat feature via Postman for comprehensive document analysis across multiple web sources.
