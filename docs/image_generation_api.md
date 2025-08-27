# Image Generation API Documentation

## API Endpoints

The RTL-Deutschland RAG Chatbot API provides two main endpoints for image generation:

1. `/image/generate` - Generate images using a single model (DALL-E 3 or Imagen)
2. `/image/generate-combined` - Generate images using both models simultaneously

## Single Model Image Generation

### Endpoint: `/image/generate`

**Method**: POST

**Request Body Schema**:

```json
{
  "prompt": "string",
  "size": "string",
  "n": "integer",
  "model_choice": "string",
  "username": "string"
}
```

**Parameters**:

| Parameter | Type | Required | Description | Constraints |
|-----------|------|----------|-------------|------------|
| `prompt` | string | Yes | Detailed description of the image to generate | Max 1000 characters |
| `size` | string | Yes | Image dimensions | One of: "1024x1024", "1024x1792", "1792x1024" |
| `n` | integer | No | Number of images to generate | 1-4 for Imagen, always 1 for DALL-E 3 |
| `model_choice` | string | Yes | Model to use for generation | One of: "dall-e-3", "imagen-3.0-generate-002", etc. |
| `username` | string | No | Username for tracking/logging | Optional |

**Response Schema**:

```json
{
  "success": "boolean",
  "is_base64": "boolean",
  "image_urls": ["string"],
  "prompt": "string",
  "model": "string",
  "size": "string",
  "error": "string"
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether the request was successful |
| `is_base64` | boolean | Whether the image data is base64-encoded |
| `image_urls` | array | Array of all base64-encoded image data (optimized format) |
| `prompt` | string | The original prompt used |
| `model` | string | The model used for generation |
| `size` | string | The size of the generated image(s) |
| `error` | string | Error message if `success` is false |

**Memory Optimization Note**: The response now uses only the `image_urls` array instead of duplicating URLs in both `image_url` and `image_urls` fields, reducing memory usage and data transmission overhead.

**Example Request**:

```bash
curl -X POST "http://localhost:8080/image/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A futuristic city with flying cars and neon lights",
    "size": "1024x1024",
    "n": 2,
    "model_choice": "imagen-3.0-generate-002",
    "username": "user123"
  }'
```

**Example Response**:

```json
{
  "success": true,
  "is_base64": true,
  "image_urls": [
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
  ],
  "prompt": "A futuristic city with flying cars and neon lights",
  "model": "imagen-3.0-generate-002",
  "size": "1024x1024"
}
```

**Important Notes**:

- When using DALL-E 3 (`model_choice` contains "dall-e"), the API will always generate only 1 image regardless of the `n` parameter value
- When using Imagen (`model_choice` contains "imagen"), the API will generate `n` images (1-4)
- The `image_url` field contains the first generated image for backward compatibility
- The `image_urls` field contains all generated images as an array

## Combined Model Image Generation

### Endpoint: `/image/generate-combined`

**Method**: POST

**Request Body Schema**:

```json
{
  "prompt": "string",
  "size": "string",
  "n": "integer",
  "username": "string"
}
```

**Parameters**:

| Parameter | Type | Required | Description | Constraints |
|-----------|------|----------|-------------|------------|
| `prompt` | string | Yes | Detailed description of the image to generate | Max 1000 characters |
| `size` | string | Yes | Image dimensions | One of: "1024x1024", "1024x1792", "1792x1024" |
| `n` | integer | No | Number of images to generate for Imagen | 1-4 (DALL-E will always be 1) |
| `username` | string | No | Username for tracking/logging | Optional |

**Response Schema**:

```json
{
  "success": "boolean",
  "dalle_result": {
    "success": "boolean",
    "is_base64": "boolean",
    "image_urls": ["string"],
    "prompt": "string",
    "model": "string",
    "size": "string"
  },
  "imagen_result": {
    "success": "boolean",
    "is_base64": "boolean",
    "image_urls": ["string"],
    "prompt": "string",
    "model": "string",
    "size": "string"
  },
  "error": "string"
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether the overall request was successful |
| `dalle_result` | object | Result from DALL-E 3 generation |
| `imagen_result` | object | Result from Imagen generation |
| `error` | string | Error message if `success` is false |

Each result object contains:

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether this model's generation was successful |
| `is_base64` | boolean | Whether the image data is base64-encoded |
| `image_url` | string | Base64-encoded image data (first image) |
| `image_urls` | array | Array of all base64-encoded image data |
| `prompt` | string | The original prompt used |
| `model` | string | The model used for generation |
| `size` | string | The size of the generated image(s) |

**Example Request**:

```bash
curl -X POST "http://localhost:8080/image/generate-combined" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene lake surrounded by mountains at sunset",
    "size": "1024x1024",
    "n": 2,
    "username": "user123"
  }'
```

**Example Response**:

```json
{
  "success": true,
  "dalle_result": {
    "success": true,
    "is_base64": true,
    "image_urls": ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."],
    "prompt": "A serene lake surrounded by mountains at sunset",
    "model": "dall-e-3",
    "size": "1024x1024"
  },
  "imagen_result": {
    "success": true,
    "is_base64": true,
    "image_urls": [
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
    ],
    "prompt": "A serene lake surrounded by mountains at sunset",
    "model": "imagen-3.0-generate-002",
    "size": "1024x1024"
  }
}
```

**Important Notes**:

- The combined endpoint always uses DALL-E 3 and Imagen models
- DALL-E 3 will always generate exactly 1 image
- Imagen will generate `n` images (1-4) as specified in the request
- Both models use the same prompt and image size
- The API uses concurrent processing to generate images from both models simultaneously

## Error Handling

The API returns appropriate HTTP status codes and error messages for various failure scenarios:

- **400 Bad Request**: Invalid parameters (wrong size, invalid model choice, etc.)
- **401 Unauthorized**: Authentication failure with the model provider
- **429 Too Many Requests**: Rate limiting by the model provider
- **500 Internal Server Error**: Server-side errors

Error response example:

```json
{
  "success": false,
  "error": "Invalid image size. Supported sizes are: 1024x1024, 1024x1792, 1792x1024"
}
```

## Implementation Notes

- The API uses asynchronous processing for efficient handling of requests
- Images are returned as base64-encoded strings for easy embedding in web applications
- The API enforces the DALL-E 3 limitation of generating only 1 image per request
- For Imagen, multiple images can be generated in a single request (up to 4)
- The combined endpoint uses concurrent processing with ThreadPoolExecutor and asyncio.gather

## Security Considerations

- The API validates all input parameters to prevent injection attacks
- Sensitive credentials are stored as environment variables, not in code
- Rate limiting is implemented to prevent abuse
- Input prompts are validated to ensure they meet the model provider's content policies
