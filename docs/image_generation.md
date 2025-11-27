# Image Generation Module

## Overview

The Image Generation module in the RTL-Deutschland RAG Chatbot API provides capabilities to generate images using multiple AI models, including Azure OpenAI's DALL-E 3, Google Vertex AI's Imagen, and Google's Gemini 2.5 Flash Image (NanoBanana). The system supports both individual model generation and combined generation using both models simultaneously.

## Features

- **Multiple Model Support**: Generate images using DALL-E 3, Imagen, NanoBanana (Gemini 2.5 Flash Image), or both DALL-E and Imagen simultaneously
- **Customizable Parameters**: Control image size, number of images, and detailed prompts
- **Base64 Image Encoding**: Images are returned as base64-encoded strings for easy embedding in web applications
- **Streamlit UI Integration**: User-friendly interface for image generation with model selection
- **Asynchronous Processing**: Efficient handling of image generation requests
- **Prompt History Support**: Context-aware image generation with prompt history for all models

## Model Capabilities and Limitations

### DALL-E 3 (Azure OpenAI)
- **Strengths**: High-quality, photorealistic images with excellent prompt understanding
- **Limitations**:
  - Can only generate 1 image per API request
  - Limited to specific image sizes (1024x1024, 1024x1792, 1792x1024)
  - Higher latency compared to some other models

### Imagen (Google Vertex AI)
- **Strengths**: Fast generation, ability to create multiple images per request
- **Capabilities**:
  - Can generate up to 4 images per API request
  - Supports the same image sizes as DALL-E 3
  - Generally lower latency

### NanoBanana (Gemini 2.5 Flash Image)
- **Strengths**: Fast generation, Google's latest image generation model, ability to create multiple images per request
- **Capabilities**:
  - Can generate up to 4 images per API request
  - Supports the same image sizes as DALL-E 3 and Imagen (1024x1024, 1024x1792, 1792x1024)
  - Uses Vertex AI authentication (shares configuration with Gemini models)
  - Base64-encoded image responses for seamless integration
- **Model Name**: `gemini-2.5-flash-image` (displayed as "NanoBanana" in UI)

### Combined Mode
- Generates images from both DALL-E 3 and Imagen simultaneously for comparison
- Always generates 1 image from DALL-E 3 and the user-specified number from Imagen
- Note: NanoBanana is not included in combined mode; it's available as a standalone option

## API Usage

### Single Model Image Generation

**Endpoint**: `/image/generate`

**Method**: POST

**Request Body**:
```json
{
  "prompt": "A detailed description of the image you want to generate",
  "size": "1024x1024",
  "n": 1,
  "model_choice": "imagen-3.0-generate-002"
}
```

**Parameters**:
- `prompt` (string, required): Detailed description of the image to generate (can be an array for prompt history)
- `size` (string, required): Image size (1024x1024, 1024x1792, or 1792x1024)
- `n` (integer, optional): Number of images to generate (1-4 for Imagen and NanoBanana, always 1 for DALL-E)
- `model_choice` (string, required): Model to use (e.g., "dall-e-3", "imagen-3.0-generate-002", "NanoBanana")

**Response**:
```json
{
  "success": true,
  "is_base64": true,
  "image_urls": ["base64_encoded_image_data", "..."],
  "prompt": "Original prompt",
  "model": "imagen-3.0-generate-002",
  "size": "1024x1024"
}
```

**Notes**:
- For DALL-E 3, `n` will always be treated as 1 regardless of input
- For Imagen and NanoBanana, `n` can be 1-4
- `prompt` can be an array for context-aware generation: `["previous prompt", "current prompt"]`
- `image_urls` contains all generated images as an array of base64 data URLs
- All models return images in the format: `data:image/png;base64,{base64_data}`

**Example with NanoBanana**:
```json
{
  "prompt": ["A red car", "make it blue"],
  "size": "1024x1024",
  "n": 2,
  "model_choice": "NanoBanana"
}
```

### Combined Model Image Generation

**Endpoint**: `/image/generate-combined`

**Method**: POST

**Request Body**:
```json
{
  "prompt": "A detailed description of the image you want to generate",
  "size": "1024x1024",
  "n": 2
}
```

**Parameters**:
- `prompt` (string, required): Detailed description of the image to generate
- `size` (string, required): Image size (1024x1024, 1024x1792, or 1792x1024)
- `n` (integer, optional): Number of images to generate for Imagen (1-4, DALL-E will always be 1)

**Response**:
```json
{
  "success": true,
  "dalle_result": {
    "success": true,
    "is_base64": true,
    "image_urls": ["base64_encoded_image_data"],
    "prompt": "Original prompt",
    "model": "dall-e-3",
    "size": "1024x1024"
  },
  "imagen_result": {
    "success": true,
    "is_base64": true,
    "image_urls": ["base64_encoded_image_data_1", "base64_encoded_image_data_2"],
    "prompt": "Original prompt",
    "model": "imagen-3.0-generate-002",
    "size": "1024x1024"
  }
}
```

## Environment Configuration

The following environment variables need to be configured:

### Azure OpenAI (DALL-E 3)
```
AZURE_DALLE_3_API_KEY=your_api_key
AZURE_DALLE_3_ENDPOINT=your_endpoint
AZURE_DALLE_3_API_VERSION=your_api_version
AZURE_DALLE_3_DEPLOYMENT=your_deployment
AZURE_DALLE_3_MODEL_NAME=dall-e-3
```

### Google Vertex AI (Imagen)
```
GEMINI__PROJECT=your_gcp_project
GEMINI__LOCATION=your_location
VERTEXAI_IMAGEN_MODEL_NAME=imagen-3.0-generate-002
```

### NanoBanana (Gemini 2.5 Flash Image)
NanoBanana uses the same Vertex AI configuration as Gemini models:
```
GEMINI__PROJECT=your_gcp_project
GEMINI__LOCATION=your_location
```
No additional configuration is required. The model uses Vertex AI authentication and shares the project and location settings with other Gemini models.

## Streamlit UI Usage

The Streamlit UI provides a user-friendly interface for image generation:

1. Select the desired model from the sidebar (DALL-E 3, Imagen, NanoBanana, or Combined)
2. Enter a detailed prompt describing the image you want to generate
3. Select the image size from the dropdown (1024x1024, 1024x1792, or 1792x1024)
4. For Imagen and NanoBanana models, select the number of images to generate (1-4)
5. Click "Generate Image" to create the image(s)
6. View and download the generated images

**Notes**:
- When using DALL-E 3 or Combined mode, the UI will automatically enforce the 1 image limitation for DALL-E 3
- NanoBanana supports multiple image generation (1-4 images) like Imagen
- The UI automatically switches to the "Image generation" tab when an image model is selected
- Prompt history is maintained for context-aware follow-up image generation

## Implementation Details

The image generation functionality is implemented across several files:

- `rtl_rag_chatbot_api/chatbot/dalle_handler.py`: Handles DALL-E 3 image generation
- `rtl_rag_chatbot_api/chatbot/imagen_handler.py`: Handles Imagen image generation
- `rtl_rag_chatbot_api/chatbot/nanobanana_handler.py`: Handles NanoBanana (Gemini 2.5 Flash Image) generation
- `rtl_rag_chatbot_api/chatbot/combined_image_handler.py`: Handles combined model generation (DALL-E + Imagen)
- `rtl_rag_chatbot_api/app.py`: Contains the API endpoints (`/image/generate` and `/image/generate-combined`)
- `streamlit_image_generation.py`: Implements the Streamlit UI for image generation

### NanoBanana Implementation

The NanoBanana handler (`nanobanana_handler.py`) uses the Google `genai` library with Vertex AI authentication:
- Uses `genai.Client` with `vertexai=True` for authentication
- Model name: `gemini-2.5-flash-image`
- Extracts images from API response parts containing `inline_data`
- Converts image bytes to base64 data URLs matching the format used by other image generators
- Supports prompt history for context-aware image generation
- Returns standardized response format compatible with existing UI components

## Error Handling

The API handles various error scenarios:

- Invalid model selections
- API rate limiting
- Authentication failures
- Invalid prompt content
- Network issues

Errors are returned with appropriate HTTP status codes and descriptive messages.

## Future Enhancements

Planned enhancements for the image generation module:

- Support for additional image generation models
- Image editing capabilities
- Image variation generation
- Prompt enhancement using AI
- Image metadata storage and retrieval
