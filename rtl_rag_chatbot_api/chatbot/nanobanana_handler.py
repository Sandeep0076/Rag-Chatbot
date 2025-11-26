import base64
import logging
from typing import Any, Dict, List, Optional

import vertexai
from google import genai
from google.genai import types

from rtl_rag_chatbot_api.common.errors import (
    ImageCreationError,
    ImagePromptRejectedError,
)


class NanoBananaGenerator:
    """
    Handles interactions with Google Gemini 2.5 Flash Image (NanoBanana) for image generation.
    """

    def __init__(self, configs):
        """
        Initialize the NanoBanana image generator with configurations.

        Args:
            configs: Application configuration object
        """
        self.configs = configs
        self.client = None
        self.model_name = "gemini-2.5-flash-image"
        self.initialize_client()

    def initialize_client(self):
        """Initialize Google Vertex AI client for NanoBanana (Gemini Flash Image)."""
        try:
            # Initialize Vertex AI with project and location from Gemini configs
            vertexai.init(
                project=self.configs.gemini.project,
                location=self.configs.gemini.location,
            )

            # Create genai client using Vertex AI authentication
            self.client = genai.Client(
                vertexai=True,
                project=self.configs.gemini.project,
                location=self.configs.gemini.location,
            )
            logging.info(
                "NanoBanana (Gemini Flash Image) client initialized successfully"
            )
        except Exception as e:
            logging.error(f"Error initializing NanoBanana client: {str(e)}")
            raise

    def _create_success_response(
        self, image_data: List[str], prompt: str, size: str
    ) -> Dict[str, Any]:
        """Create a success response matching ImagenGenerator format.

        Args:
            image_data: List of base64 encoded image data URLs
            prompt: Original prompt
            size: Image size

        Returns:
            Success response dictionary matching ImagenGenerator format
        """
        logging.info(f"NanoBanana: Returning {len(image_data)} URLs in response")

        # Return response exactly matching ImagenGenerator format
        return {
            "success": True,
            "is_base64": True,  # Required for Streamlit to handle base64 images
            "image_urls": image_data,  # Array of "data:image/png;base64,..." URLs
            "prompt": prompt,
            "model": self.model_name,
            "size": size,
        }

    def _create_error_response(
        self,
        error_msg: str,
        prompt: str,
        error_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create standardized error response using BaseAppError hierarchy.

        Maps safety / blocked content to ImagePromptRejectedError (3004), other issues
        to ImageCreationError (3003).
        """
        lower_msg = error_msg.lower()
        safety_indicators = [
            "safety",
            "blocked",
            "policy",
            "filter",
            "adult",
            "violation",
            "harm",
        ]
        is_prompt_rejected = any(k in lower_msg for k in safety_indicators)

        details = error_details.copy() if error_details else {}
        details.update(
            {
                "prompt": prompt,
                "model": self.model_name,
            }
        )

        if is_prompt_rejected:
            err = ImagePromptRejectedError(
                "Prompt rejected by safety filters", details=details
            )
        else:
            err = ImageCreationError(error_msg, details=details)
        return err.to_response()

    def _prepare_prompt(self, prompt: str, size: str) -> str:
        """Prepare the full prompt with size specifications.

        Args:
            prompt: Original user prompt
            size: Image size specification (e.g., "1024x1024")

        Returns:
            Full prompt with size specifications
        """
        size_parts = size.split("x")
        if len(size_parts) == 2:
            width, height = size_parts
            size_spec = f"Generate a {width}x{height} pixels image. "
        else:
            size_spec = f"Generate a {size} pixels image. "
        return size_spec + prompt

    def _create_generation_config(self) -> types.GenerateContentConfig:
        """Create the generation configuration with safety settings.

        Returns:
            GenerateContentConfig with appropriate settings
        """
        return types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=32768,
            response_modalities=["TEXT", "IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT", threshold="OFF"
                ),
            ],
        )

    def _extract_image_from_part(self, part) -> Optional[str]:
        """Extract base64 image data URL from a response part.

        Args:
            part: Response part that may contain inline image data

        Returns:
            Base64 data URL string if image found, None otherwise
        """
        if not (hasattr(part, "inline_data") and part.inline_data):
            return None

        try:
            image_bytes = part.inline_data.data
            if not image_bytes:
                logging.warning("NanoBanana: inline_data exists but data is empty")
                return None

            # Convert to base64 data URL (matching ImagenGenerator format)
            base64_data = base64.b64encode(image_bytes).decode("utf-8")
            image_data_url = f"data:image/png;base64,{base64_data}"

            logging.info(f"NanoBanana: Image received (size: {len(image_bytes)} bytes)")
            return image_data_url
        except Exception as e:
            logging.error(
                f"NanoBanana: Error extracting image from inline_data: {str(e)}"
            )
            return None

    def _extract_text_from_part(self, part) -> str:
        """Extract text from a response part.

        Args:
            part: Response part that may contain text

        Returns:
            Text content or empty string
        """
        if hasattr(part, "text") and part.text:
            logging.debug(f"NanoBanana: Received text part: {part.text[:100]}...")
            return part.text
        return ""

    def _process_response_parts(self, response) -> tuple[List[str], str]:
        """Process response to extract images and text.

        Args:
            response: API response object

        Returns:
            Tuple of (image_urls list, response_text string)
        """
        image_urls = []
        response_text = ""

        if not (hasattr(response, "candidates") and response.candidates):
            return image_urls, response_text

        for candidate in response.candidates:
            if not (hasattr(candidate, "content") and candidate.content):
                continue

            if not (hasattr(candidate.content, "parts") and candidate.content.parts):
                continue

            for part in candidate.content.parts:
                # Try to extract image
                image_url = self._extract_image_from_part(part)
                if image_url:
                    image_urls.append(image_url)
                    continue

                # Try to extract text
                text = self._extract_text_from_part(part)
                if text:
                    response_text += text

        # Also check response.text if available (safely)
        try:
            if hasattr(response, "text") and response.text:
                response_text += response.text
        except Exception:
            # response.text throws error when response contains inline_data, which is expected
            pass

        return image_urls, response_text

    def _validate_and_create_response(
        self, image_urls: List[str], response_text: str, prompt: str, size: str
    ) -> Dict[str, Any]:
        """Validate image generation results and create appropriate response.

        Args:
            image_urls: List of generated image URLs
            response_text: Any text response from the API
            prompt: Original prompt
            size: Image size

        Returns:
            Success or error response dictionary
        """
        if len(image_urls) == 0:
            error_msg = (
                "No images were generated by NanoBanana. "
                "Possible causes: 1) Content safety filters, "
                "2) Invalid generation parameters, "
                "3) API service issue."
            )
            if response_text:
                error_msg += f" Response text: '{response_text[:200]}...'"

            logging.error(
                f"{error_msg} Processed {len(image_urls)} images from response."
            )
            return self._create_error_response(error_msg, prompt)

        logging.info(f"NanoBanana: Successfully generated {len(image_urls)} image(s)")
        return self._create_success_response(image_urls, prompt, size)

    def generate_image(
        self, prompt: str, size: str = "1024x1024", n: int = 1
    ) -> Dict[str, Any]:
        """
        Generate an image based on the provided prompt using NanoBanana (Gemini Flash Image).

        Args:
            prompt (str): The text prompt to generate an image from
            size (str): Image size (default: "1024x1024")
            n (int): Number of images to generate (default: 1)

        Returns:
            Dict[str, Any]: Response containing the generated image URLs and other metadata
        """
        try:
            logging.info(
                f"NanoBanana: Generating {n} image(s) with prompt: '{prompt[:100]}...'"
            )

            # Prepare prompt with size specifications
            full_prompt = self._prepare_prompt(prompt, size)

            # Create content for the request
            contents = [
                types.Content(role="user", parts=[types.Part(text=full_prompt)])
            ]

            # Configure generation settings
            generate_content_config = self._create_generation_config()

            # Generate content using API
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=generate_content_config,
                )

                # Extract images and text from response
                image_urls, response_text = self._process_response_parts(response)

            except Exception as e:
                logging.error(
                    f"NanoBanana: Error in non-streaming generation: {str(e)}"
                )
                raise

            # Validate results and create response
            return self._validate_and_create_response(
                image_urls, response_text, prompt, size
            )

        except Exception as e:
            logging.error(
                f"Error generating image with NanoBanana: {str(e)}", exc_info=True
            )
            return self._create_error_response(
                f"Image generation failed: {str(e)}",
                prompt,
                error_details={"error_type": type(e).__name__},
            )
