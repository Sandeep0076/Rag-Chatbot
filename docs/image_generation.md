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
- **Strengths**: Fast generation, Google's latest image generation model, ability to create multiple images per request, **supports image-to-image editing with single or multiple input images**
- **Capabilities**:
  - Can generate up to **10 images per API request** (significantly more than DALL-E 3 and Imagen)
  - Supports the same image sizes as DALL-E 3 and Imagen (1024x1024, 1024x1792, 1792x1024)
  - **Unique Feature**: Supports image-to-image editing (modify existing images based on prompts)
  - **Multi-Image Input Support**: Can accept 1-3 input images and edit them together with a text prompt
  - Uses Vertex AI authentication (shares configuration with Gemini models)
  - Base64-encoded image responses for seamless integration
  - **Enhanced Multi-Image Generation**: Explicitly requests multiple distinct images in the prompt for better variety
- **Model Name**: `gemini-2.5-flash-image` (displayed as "NanoBanana" in UI)
- **Image Editing**: Unlike DALL-E and Imagen which only support text-to-image generation, NanoBanana can accept one or more input images and modify them based on your prompt
- **Multi-Image Support**: When requesting multiple images, the system automatically enhances the prompt to ensure variety and distinctness between generated images

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
- `prompt` (string or array, required): Detailed description of the image to generate. Can be:
  - A single string: `"A red car"`
  - An array for prompt history: `["A red car", "make it blue"]`
- `size` (string, required): Image size (1024x1024, 1024x1792, or 1792x1024)
- `n` (integer, optional): Number of images to generate (1-4 for Imagen, 1-10 for NanoBanana, always 1 for DALL-E)
- `model_choice` (string, required): Model to use (e.g., "dall-e-3", "imagen-3.0-generate-002", "NanoBanana")
- `session_id` (string, optional): Session identifier for tracking conversation context across requests
- `reference_image_file_id` (string, optional): File ID of reference image (for frontend tracking purposes)
- `input_image_base64` (string or array, optional): Base64-encoded input image(s) for image-to-image editing (NanoBanana only)
  - Can be a single string (legacy) or array of strings (multi-image support)
  - Format: `data:image/png;base64,{base64_data}` or `data:image/jpeg;base64,{base64_data}`
  - Supports 1-3 images for multi-image editing
  - Maximum size: 10MB per image
  - Only supported by NanoBanana model
  - Example single image: `"data:image/png;base64,iVBORw0KG..."`
  - Example multiple images: `["data:image/jpeg;base64,...", "data:image/png;base64,..."]`

**Response**:
```json
{
  "success": true,
  "is_base64": true,
  "image_urls": ["base64_encoded_image_data", "..."],
  "prompt": "Original prompt",
  "model": "imagen-3.0-generate-002",
  "size": "1024x1024",
  "operation_type": "text_to_image",
  "reference_image_used": false,
  "session_id": "optional-session-id",
  "reference_image_file_id": "optional-file-id"
}
```

**Response Fields**:
- `success` (boolean): Whether the generation was successful
- `is_base64` (boolean): Whether images are base64-encoded (true for Imagen and NanoBanana)
- `image_urls` (array): Array of generated images (URLs for DALL-E, base64 data URLs for others)
- `prompt` (string): The original or final prompt used
- `model` (string): The model that generated the images
- `size` (string): The size of generated images
- `operation_type` (string): Either "text_to_image" or "image_to_image"
- `reference_image_used` (boolean): Whether an input image was used for editing
- `session_id` (string, optional): Session ID if provided in request
- `reference_image_file_id` (string, optional): Reference image file ID if provided

**Notes**:
- For DALL-E 3, `n` will always be treated as 1 regardless of input
- For Imagen, `n` can be 1-4
- For NanoBanana, `n` can be 1-10 (supports the highest number of images per request)
- `prompt` can be an array for context-aware generation: `["previous prompt", "current prompt"]`
- `image_urls` contains all generated images as an array of base64 data URLs
- All models return images in the format: `data:image/png;base64,{base64_data}`
- For multiple images with NanoBanana, the prompt is automatically enhanced to ensure variety and distinctness

**Example 1: Text-to-Image with NanoBanana**:
```json
{
  "prompt": "A red sports car in a desert landscape",
  "size": "1024x1024",
  "n": 2,
  "model_choice": "NanoBanana"
}
```

**Example 2: Image-to-Image Editing with NanoBanana**:
```json
{
  "prompt": "Make the car blue and add mountains in the background",
  "size": "1024x1024",
  "n": 1,
  "model_choice": "NanoBanana",
  "input_image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgA...",
  "session_id": "user-session-123"
}
```

**Example 3: Multi-Image Generation with NanoBanana**:
```json
{
  "prompt": "A cyberpunk hacker working on a computer",
  "size": "1024x1024",
  "n": 3,
  "model_choice": "NanoBanana"
}
```

**Example 4: Context-Aware Generation with Prompt History**:
```json
{
  "prompt": ["A red car", "make it blue", "add racing stripes"],
  "size": "1024x1024",
  "n": 1,
  "model_choice": "NanoBanana"
}
```

**Example 5: Multi-Image Editing with NanoBanana**:
```json
{
  "prompt": "Take the couple from the first picture and make them stand near that stone building",
  "size": "1024x1024",
  "n": 1,
  "model_choice": "NanoBanana",
  "input_image_base64": [
    "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgA..."
  ],
  "session_id": "user-session-123"
}
```
This example shows how to upload 2 images and edit them together. NanoBanana supports 1-3 input images for multi-image editing.

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

The Streamlit UI provides a user-friendly interface for image generation with full support for image editing:

### Basic Image Generation

1. Select the desired model from the sidebar (DALL-E 3, Imagen, NanoBanana, or Combined)
2. Enter a detailed prompt describing the image you want to generate
3. Select the image size from the dropdown (1024x1024, 1024x1792, or 1792x1024)
4. For Imagen and NanoBanana models, select the number of images to generate (1-4)
5. Click "Generate Image" to create the image(s)
6. View and download the generated images

### Image-to-Image Editing (NanoBanana Only)

When NanoBanana is selected, additional image editing features are available:

1. **Upload an Image**: Use the file uploader to select an image to edit (PNG, JPG, JPEG, WEBP)
   - Maximum file size: 10MB
   - The uploaded image will be displayed as a preview
2. **Enter Editing Prompt**: Describe how you want to modify the image
   - Example: "Make the sky purple and add stars"
3. **Automatic Reference**: If no image is uploaded, the system can use the last generated image as reference
4. **Session-Based History**: The UI maintains a history of generated images within your session
   - View previous images in the "View previous generated images" expander
   - The last generated image is automatically used for modifications if no new image is uploaded

### Features

- **10MB File Size Validation**: Automatic validation prevents uploading images larger than 10MB
- **Image Preview**: Uploaded images are displayed before generation
- **Session Management**: Each session has a unique ID for tracking image generation history
- **Prompt History**: Maintains context across multiple generations for better results
- **Operation Type Display**: Shows whether the operation was text-to-image or image-to-image
- **Clear History**: Button to clear all prompt and image history and start a new session

**Notes**:
- When using DALL-E 3 or Combined mode, the UI will automatically enforce the 1 image limitation for DALL-E 3
- NanoBanana supports multiple image generation (1-4 images) like Imagen
- The UI automatically switches to the "Image generation" tab when an image model is selected
- Prompt history is maintained for context-aware follow-up image generation
- Image editing is **only available with NanoBanana** - DALL-E and Imagen support text-to-image only

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
- **Image-to-Image Support**: Accepts `input_image_base64` parameter and converts it to `types.Blob` for the Gemini API
- Extracts images from API response parts containing `inline_data`
- Converts image bytes to base64 data URLs matching the format used by other image generators
- Supports prompt history for context-aware image generation
- Returns standardized response format compatible with existing UI components
- Validates that input images are only processed when using NanoBanana model

## Error Handling

The API handles various error scenarios:

- Invalid model selections
- API rate limiting
- Authentication failures
- Invalid prompt content
- Network issues

Errors are returned with appropriate HTTP status codes and descriptive messages.

## Storage Architecture

The image generation system uses different storage strategies depending on the client:

### Node.js Frontend (Production)
- **Persistent Storage**: Images are stored in a database with unique `file_id` identifiers
- **Session Tracking**: Uses `session_id` to track conversation context across requests
- **Reference Images**: Frontend manages image storage and passes `reference_image_file_id` for tracking
- **Input Images**: Sends images as base64-encoded strings via `input_image_base64` parameter

### Streamlit UI (Development/Testing)
- **Temporary Storage**: Uses `st.session_state` for session-based image history
- **Automatic Cleanup**: History is cleared when the session ends or user clicks "Clear History"
- **Last 5 Images**: Keeps only the most recent 5 generated images to avoid memory issues
- **Session ID**: Generates a unique UUID for each session to track context

### Backend Processing
- **Stateless**: Backend does not persist images; it only processes base64 data
- **10MB Limit**: Enforces maximum file size for input images
- **Model Validation**: Ensures image editing is only attempted with NanoBanana

## Image Editing Workflow

### For Node.js Frontend:
1. User uploads image → Frontend stores in database with `file_id`
2. User enters modification prompt
3. Frontend sends request with:
   - `input_image_base64`: Base64-encoded image data
   - `reference_image_file_id`: Database file ID for tracking
   - `session_id`: User session identifier
   - `prompt`: Modification instructions
4. Backend processes with NanoBanana
5. Returns modified image with metadata
6. Frontend stores new image with new `file_id`

### For Streamlit UI:
1. User uploads image OR system uses last generated image
2. Image is converted to base64 and validated (10MB limit)
3. Preview is displayed
4. User enters modification prompt
5. Request sent to backend with:
   - `input_image_base64`: Base64-encoded image
   - `session_id`: Auto-generated UUID
   - `prompt`: Modification instructions
6. Backend processes with NanoBanana
7. Generated image is displayed and stored in `st.session_state`
8. Image becomes available for future modifications in the session

## Future Enhancements

Planned enhancements for the image generation module:

- Support for additional image generation models
- Image variation generation (using existing images as style reference)
- Prompt enhancement using AI
- Batch image editing capabilities
- Image metadata storage and retrieval
- Advanced image manipulation features (inpainting, outpainting)
