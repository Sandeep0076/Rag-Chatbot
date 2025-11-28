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

    def _prepare_prompt(self, prompt: str, size: str, n: int = 1) -> str:
        """Prepare the full prompt with size specifications and number of images.

        Args:
            prompt: Original user prompt
            size: Image size specification (e.g., "1024x1024")
            n: Number of images to generate

        Returns:
            Full prompt with size specifications and image count
        """
        size_parts = size.split("x")
        if len(size_parts) == 2:
            width, height = size_parts
            size_spec = f"Generate {n} distinct {width}x{height} pixels image{'s' if n > 1 else ''}. "
        else:
            size_spec = (
                f"Generate {n} distinct {size} pixels image{'s' if n > 1 else ''}. "
            )

        # For multiple images, add explicit instruction for variety and clear enumeration
        if n > 1:
            # Use the pattern from research: "Generate three distinct images: A, B, and C"
            variety_instruction = (
                f"Create exactly {n} different images that are distinct and unique variations. "
                f"Each of the {n} images should be a separate interpretation of the following theme: "
            )
            return size_spec + variety_instruction + prompt
        else:
            return size_spec + prompt

    def _create_generation_config(self, n: int = 1) -> types.GenerateContentConfig:
        """Create the generation configuration with safety settings.

        Args:
            n: Number of images to generate (for potential future use)

        Returns:
            GenerateContentConfig with appropriate settings
        """
        config = types.GenerateContentConfig(
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

        logging.info(f"NanoBanana: Created generation config for {n} image(s)")
        return config

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
        total_candidates = 0
        total_parts = 0

        if not (hasattr(response, "candidates") and response.candidates):
            logging.warning("NanoBanana: No candidates found in response")
            return image_urls, response_text

        total_candidates = len(response.candidates)
        logging.info(f"NanoBanana: Processing {total_candidates} candidate(s)")

        for candidate_idx, candidate in enumerate(response.candidates):
            if not (hasattr(candidate, "content") and candidate.content):
                logging.warning(f"NanoBanana: Candidate {candidate_idx} has no content")
                continue

            if not (hasattr(candidate.content, "parts") and candidate.content.parts):
                logging.warning(f"NanoBanana: Candidate {candidate_idx} has no parts")
                continue

            candidate_parts = len(candidate.content.parts)
            total_parts += candidate_parts
            logging.info(
                f"NanoBanana: Candidate {candidate_idx} has {candidate_parts} part(s)"
            )

            for part_idx, part in enumerate(candidate.content.parts):
                # Try to extract image
                image_url = self._extract_image_from_part(part)
                if image_url:
                    image_urls.append(image_url)
                    logging.info(
                        f"NanoBanana: Found image in candidate {candidate_idx}, part {part_idx}"
                    )
                    continue

                # Try to extract text
                text = self._extract_text_from_part(part)
                if text:
                    response_text += text
                    logging.debug(
                        f"NanoBanana: Found text in candidate {candidate_idx}, part {part_idx}"
                    )

        # Also check response.text if available (safely)
        try:
            if hasattr(response, "text") and response.text:
                response_text += response.text
                logging.debug("NanoBanana: Added response.text to response_text")
        except Exception:
            # response.text throws error when response contains inline_data, which is expected
            logging.debug(
                "NanoBanana: response.text not accessible (expected for image responses)"
            )
            pass

        logging.info(
            f"NanoBanana: Processed {total_candidates} candidates, "
            f"{total_parts} total parts, found {len(image_urls)} images"
        )
        return image_urls, response_text

    def _validate_and_create_response(
        self,
        image_urls: List[str],
        response_text: str,
        prompt: str,
        size: str,
        expected_count: int = 1,
    ) -> Dict[str, Any]:
        """Validate image generation results and create appropriate response.

        Args:
            image_urls: List of generated image URLs
            response_text: Any text response from the API
            prompt: Original prompt
            size: Image size
            expected_count: Expected number of images

        Returns:
            Success or error response dictionary
        """
        received_count = len(image_urls)

        if received_count == 0:
            error_msg = (
                "No images were generated by NanoBanana. "
                "Possible causes: 1) Content safety filters, "
                "2) Invalid generation parameters, "
                "3) API service issue."
            )
            if response_text:
                error_msg += f" Response text: '{response_text[:200]}...'"

            logging.error(
                f"{error_msg} Expected {expected_count}, received {received_count} images from response."
            )
            return self._create_error_response(error_msg, prompt)

        # Log success with count comparison
        if received_count != expected_count:
            logging.warning(
                f"NanoBanana: Expected {expected_count} images but received {received_count} images"
            )

        logging.info(
            f"NanoBanana: Successfully generated {received_count} image(s) (expected: {expected_count})"
        )
        return self._create_success_response(image_urls, prompt, size)

    def _prepare_image_input(self, input_image_base64: str) -> Optional[bytes]:
        """
        Prepare base64 image input for Gemini API.

        Args:
            input_image_base64: Base64-encoded image data (with or without data URI prefix)

        Returns:
            Image bytes or None if invalid
        """
        try:
            # Remove data URI prefix if present
            if input_image_base64.startswith("data:image/"):
                # Format: data:image/png;base64,<data>
                base64_data = input_image_base64.split(",", 1)[1]
            else:
                base64_data = input_image_base64

            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_data)
            logging.info(f"NanoBanana: Prepared input image ({len(image_bytes)} bytes)")
            return image_bytes
        except Exception as e:
            logging.error(f"NanoBanana: Error preparing image input: {str(e)}")
            return None

    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        n: int = 1,
        input_image_base64: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate an image based on the provided prompt using NanoBanana (Gemini Flash Image).
        Supports both text-to-image and image-to-image generation.

        Args:
            prompt (str): The text prompt to generate an image from
            size (str): Image size (default: "1024x1024")
            n (int): Number of images to generate (default: 1, max: 4)
            input_image_base64 (Optional[str]): Base64-encoded input image for image-to-image editing

        Returns:
            Dict[str, Any]: Response containing the generated image URLs and other metadata
        """
        try:
            # Validate number of images (limited to 4 images per request)
            if n < 1:
                return self._create_error_response(
                    "Number of images must be at least 1", prompt
                )
            if n > 4:
                logging.warning(
                    f"NanoBanana: Requested {n} images, limiting to 4 (maximum supported)"
                )
                n = 4

            operation_type = "edit_existing" if input_image_base64 else "new_generation"
            logging.info(
                f"NanoBanana: {operation_type} - Generating {n} image(s) with prompt: '{prompt[:100]}...'"
            )

            # Prepare prompt with size specifications and number of images
            full_prompt = self._prepare_prompt(prompt, size, n)

            logging.info(
                f"NanoBanana: Prepared prompt for {n} image(s): '{full_prompt[:150]}...'"
            )

            # Create content parts for the request
            content_parts = []

            # Add input image if provided (for image-to-image editing)
            if input_image_base64:
                image_bytes = self._prepare_image_input(input_image_base64)
                if image_bytes:
                    # Add image as inline data
                    content_parts.append(
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="image/png", data=image_bytes
                            )
                        )
                    )
                    logging.info("NanoBanana: Added input image for editing")
                else:
                    logging.warning(
                        "NanoBanana: Failed to prepare input image, proceeding with text-to-image"
                    )

            # Add text prompt
            content_parts.append(types.Part(text=full_prompt))

            # Create content for the request
            contents = [types.Content(role="user", parts=content_parts)]

            # Configure generation settings
            generate_content_config = self._create_generation_config(n)

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
                logging.error(f"NanoBanana: Error in generation: {str(e)}")
                raise

            # Validate results and create response
            return self._validate_and_create_response(
                image_urls, response_text, prompt, size, n
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
