import base64
import logging
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Union

import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

from rtl_rag_chatbot_api.common.errors import (
    ImageCreationError,
    ImagePromptRejectedError,
)


class ImagenGenerator:
    """
    Handles interactions with Google Vertex AI Imagen for image generation.
    """

    def __init__(self, configs):
        """
        Initialize the Imagen image generator with configurations.

        Args:
            configs: Application configuration object
        """
        self.configs = configs
        self.client = None
        self.initialize_client()
        # Cache of initialized Imagen model clients keyed by model name
        self._model_clients: Dict[str, Any] = {
            self.configs.vertexai_imagen.model_name: self.client
        }

    def initialize_client(self):
        """Initialize Google Vertex AI client for Imagen."""
        try:
            # Initialize Vertex AI with project and location from configs
            vertexai.init(
                project=self.configs.vertexai_imagen.project,
                location=self.configs.vertexai_imagen.location,
            )

            # Initialize the image generation model from preview module
            self.client = ImageGenerationModel.from_pretrained(
                self.configs.vertexai_imagen.model_name
            )
            logging.info("Imagen client initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing Imagen client: {str(e)}")
            raise

    def _get_model_client(self, model_name: str):
        """Get or create a cached Imagen client for the specified model name."""
        try:
            if model_name in self._model_clients:
                return self._model_clients[model_name]

            # Ensure Vertex AI is initialized (idempotent)
            vertexai.init(
                project=self.configs.vertexai_imagen.project,
                location=self.configs.vertexai_imagen.location,
            )

            client = ImageGenerationModel.from_pretrained(model_name)
            self._model_clients[model_name] = client
            logging.info(f"Initialized Imagen client for model: {model_name}")
            return client
        except Exception as e:
            logging.error(
                f"Failed to initialize Imagen client for {model_name}: {str(e)}"
            )
            raise

    def _convert_size_to_aspect_ratio(self, size: str) -> str:
        """Convert size string to aspect ratio format for Imagen.

        Args:
            size: Size string in format like "1024x1024"

        Returns:
            Aspect ratio string like "1:1"
        """
        if size == "1024x1024":
            return "1:1"  # Square
        elif size == "1024x1792":
            return "9:16"  # Portrait
        elif size == "1792x1024":
            return "16:9"  # Landscape
        else:
            # Default to square if unrecognized
            return "1:1"

    def _extract_base64_string(self, generated_image: Any) -> Optional[str]:
        """Try to extract base64 string from generated image using _as_base64_string.

        Args:
            generated_image: GeneratedImage object from Imagen API

        Returns:
            Base64 string or None if extraction failed
        """
        if not hasattr(generated_image, "_as_base64_string"):
            return None

        try:
            # Based on logs, this is the primary successful method
            base64_data = generated_image._as_base64_string()
            logging.info("Successfully extracted base64 data")
            return base64_data
        except Exception as e:
            # Fall back to attribute access if method call fails
            try:
                if hasattr(generated_image, "_as_base64_string"):
                    base64_data = generated_image._as_base64_string
                    if base64_data:
                        return base64_data
            except Exception:
                pass

            logging.error(f"Error accessing _as_base64_string: {str(e)}")
            return None

    def _extract_image_bytes(self, generated_image: Any) -> Optional[bytes]:
        """Try to extract image bytes from generated image using _image_bytes.

        Args:
            generated_image: GeneratedImage object from Imagen API

        Returns:
            Image bytes or None if extraction failed
        """
        if (
            not hasattr(generated_image, "_image_bytes")
            or not generated_image._image_bytes
        ):
            return None

        try:
            image_bytes = generated_image._image_bytes
            logging.info("Successfully extracted _image_bytes")
            return image_bytes
        except Exception as e:
            logging.error(f"Error accessing _image_bytes: {str(e)}")

        return None

    def _extract_using_save_method(self, generated_image: Any) -> Optional[bytes]:
        """Try to extract image by saving it to a temporary file.

        Args:
            generated_image: GeneratedImage object from Imagen API

        Returns:
            Image bytes or None if extraction failed
        """
        if not hasattr(generated_image, "save"):
            return None

        try:
            # Create a temporary file to save the image
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"imagen_{uuid.uuid4()}.png")

            # Save the image to the temporary file
            generated_image.save(temp_file)
            logging.info(f"Saved image to temporary file: {temp_file}")

            # Read the file and return the bytes
            with open(temp_file, "rb") as f:
                image_bytes = f.read()

            # Clean up the temporary file
            try:
                os.remove(temp_file)
            except Exception:
                pass

            return image_bytes
        except Exception as e:
            logging.error(f"Error using save method: {str(e)}")

        return None

    def _extract_loaded_bytes(self, generated_image: Any) -> Optional[bytes]:
        """Try to extract image bytes from _loaded_bytes attribute.

        Args:
            generated_image: GeneratedImage object from Imagen API

        Returns:
            Image bytes or None if extraction failed
        """
        if (
            not hasattr(generated_image, "_loaded_bytes")
            or not generated_image._loaded_bytes
        ):
            return None

        try:
            image_bytes = generated_image._loaded_bytes
            logging.info("Successfully extracted _loaded_bytes")
            return image_bytes
        except Exception as e:
            logging.error(f"Error accessing _loaded_bytes: {str(e)}")

        return None

    def _create_success_response(
        self, image_data: Union[str, List[str]], prompt: str, size: str
    ) -> Dict[str, Any]:
        """Create a success response with optimized URL structure.

        Args:
            image_data: Base64 encoded image data URL or list of URLs for multiple images
            prompt: Original prompt
            size: Image size

        Returns:
            Success response dictionary with single URL array to reduce memory usage
        """
        # Convert single image to list for consistent response structure
        if isinstance(image_data, str):
            image_urls = [image_data]
        else:
            image_urls = image_data

        # Log memory optimization info
        logging.info(
            f"Optimized response: returning {len(image_urls)} URLs in single array"
        )

        # Return optimized response with only image_urls array (no duplicate image_url field)
        return {
            "success": True,
            "is_base64": True,
            "image_urls": image_urls,  # Single source of truth for image URLs
            "prompt": prompt,
            "model": self.configs.vertexai_imagen.model_name,
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
        ]
        is_prompt_rejected = any(k in lower_msg for k in safety_indicators)

        details = error_details.copy() if error_details else {}
        details.update(
            {
                "prompt": prompt,
                "model": self.configs.vertexai_imagen.model_name,
            }
        )

        if is_prompt_rejected:
            err = ImagePromptRejectedError(
                "Prompt rejected by safety filters", details=details
            )
        else:
            err = ImageCreationError(error_msg, details=details)
        return err.to_response()

    def _extract_image_data(self, generated_image: Any) -> Optional[str]:
        """Try all methods to extract image data from the generated image.

        Args:
            generated_image: GeneratedImage object from Imagen API

        Returns:
            Base64 encoded image data URL or None if all extraction methods failed
        """
        # Try different extraction methods in order of preference
        # 1. Try base64 string extraction
        base64_data = self._extract_base64_string(generated_image)
        if base64_data:
            return f"data:image/png;base64,{base64_data}"

        # 2. Try image bytes extraction
        image_bytes = self._extract_image_bytes(generated_image)
        if image_bytes:
            base64_data = base64.b64encode(image_bytes).decode("utf-8")
            return f"data:image/png;base64,{base64_data}"

        # 3. Try save method extraction
        image_bytes = self._extract_using_save_method(generated_image)
        if image_bytes:
            base64_data = base64.b64encode(image_bytes).decode("utf-8")
            return f"data:image/png;base64,{base64_data}"

        # 4. Try loaded bytes extraction
        image_bytes = self._extract_loaded_bytes(generated_image)
        if image_bytes:
            base64_data = base64.b64encode(image_bytes).decode("utf-8")
            return f"data:image/png;base64,{base64_data}"

        return None

    def _process_response(self, response: Any, prompt: str) -> Dict[str, Any]:
        """Process the API response to extract images.

        Args:
            response: Response from the Imagen API
            prompt: Original prompt

        Returns:
            Response dictionary with image data or error
        """
        if not hasattr(response, "images"):
            logging.error("Response doesn't have 'images' attribute")
            response_info = str(response)
            logging.info("Response info: {}".format(response_info[:200] + "..."))
            return self._create_error_response(
                "Unexpected response format from Imagen API: {}".format(type(response)),
                prompt,
            )

        images = response.images
        logging.info(f"Found images attribute with {len(images)} images")

        if len(images) == 0:
            logging.error("No images were generated by Imagen API")

            # Extract detailed error information from response
            error_details = {}
            error_reasons = []

            # Check generation parameters
            if hasattr(response, "generation_parameters"):
                gen_params = response.generation_parameters
                logging.info(f"Generation parameters: {gen_params}")
                error_details["generation_parameters"] = str(gen_params)

            # Check for safety attributes/filters
            if hasattr(response, "safety_attributes"):
                safety_attrs = response.safety_attributes
                logging.info(f"Safety attributes: {safety_attrs}")
                error_details["safety_attributes"] = str(safety_attrs)
                error_reasons.append("Content may have been blocked by safety filters")

            # Check raw response for additional error info
            if hasattr(response, "_raw_response"):
                raw_resp = response._raw_response
                logging.info(f"Raw response type: {type(raw_resp)}")

                # Try to extract error from raw response
                if hasattr(raw_resp, "error"):
                    error_details["api_error"] = str(raw_resp.error)
                    error_reasons.append(f"API Error: {raw_resp.error}")
                    logging.error(f"API returned error: {raw_resp.error}")

                # Check for blocked reasons
                if hasattr(raw_resp, "blocked_reason"):
                    blocked_reason = raw_resp.blocked_reason
                    error_details["blocked_reason"] = str(blocked_reason)
                    error_reasons.append(f"Blocked: {blocked_reason}")
                    logging.error(f"Content blocked: {blocked_reason}")

            # Check for finish_reason on response
            if hasattr(response, "finish_reason"):
                finish_reason = response.finish_reason
                error_details["finish_reason"] = str(finish_reason)
                if finish_reason and finish_reason != "SUCCESS":
                    error_reasons.append(f"Generation stopped: {finish_reason}")
                    logging.error(f"Finish reason: {finish_reason}")

            # Build comprehensive error message
            if error_reasons:
                primary_error = "; ".join(error_reasons)
            else:
                primary_error = "No images were generated by the API"

            error_msg = (
                f"{primary_error}. "
                f"Possible causes: 1) Content safety filters, "
                f"2) Invalid generation parameters, "
                f"3) API service issue. "
                f"Prompt: '{prompt[:100]}...'"
            )

            # Log all collected error details
            if error_details:
                logging.error(f"Imagen error details: {error_details}")

            return self._create_error_response(error_msg, prompt, error_details)

        # Process all generated images
        all_image_data = []

        for i, generated_image in enumerate(images):
            logging.info(f"Processing image {i + 1}/{len(images)}")

            # Try to extract image data using all available methods
            image_data = self._extract_image_data(generated_image)
            if image_data:
                all_image_data.append(image_data)
            else:
                logging.error(f"Failed to extract data for image {i + 1}")

        # Check if we extracted at least one image
        if all_image_data:
            logging.info(f"Successfully processed {len(all_image_data)} images")
            return self._create_success_response(all_image_data, prompt, size="")
        else:
            return self._create_error_response(
                "Could not extract image data from generated image", prompt
            )

    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        n: int = 1,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate an image based on the provided prompt using Imagen.

        Args:
            prompt (str): The text prompt to generate an image from
            size (str): Image size (default: "1024x1024")
            n (int): Number of images to generate (default: 1)
            model_name (Optional[str]): Override Imagen model name for this request

        Returns:
            Dict[str, Any]: Response containing the generated image URL and other metadata
        """
        try:
            # Determine which Imagen model to use
            used_model_name = model_name or self.configs.vertexai_imagen.model_name
            client_to_use = (
                self.client
                if used_model_name == self.configs.vertexai_imagen.model_name
                else self._get_model_client(used_model_name)
            )

            # Convert size to aspect ratio
            aspect_ratio = self._convert_size_to_aspect_ratio(size)

            try:
                # Check if this is Imagen 3.0 which requires person_generation parameter
                is_imagen_3 = (
                    "3.0" in used_model_name or "imagen-3" in used_model_name.lower()
                )

                # Build parameters dict
                params = {
                    "prompt": prompt,
                    "number_of_images": int(n),
                    "aspect_ratio": aspect_ratio,
                }

                # For Imagen 3.0, add person_generation to avoid safety filter issues
                if is_imagen_3:
                    params["person_generation"] = "allow_adult"
                response = client_to_use.generate_images(**params)

                # Process the response to extract image data
                result = self._process_response(response, prompt)

                # Add size and model to the result if it's a success
                if result.get("success"):
                    if "size" in result and not result["size"]:
                        result["size"] = size
                    # Ensure the model field reflects the actual model used
                    result["model"] = used_model_name

                return result

            except Exception as e:
                logging.error("Error processing Imagen response: {}".format(str(e)))
                return self._create_error_response(
                    "Error processing response: {}".format(str(e)), prompt
                )

        except Exception as e:
            logging.error(f"Error generating image with Imagen: {str(e)}")
            return self._create_error_response(str(e), prompt)
