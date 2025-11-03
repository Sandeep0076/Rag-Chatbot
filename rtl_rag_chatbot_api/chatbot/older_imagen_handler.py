import base64
import logging
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Union

import vertexai
from vertexai.preview.vision_models import ImageGenerationModel


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

    def _create_error_response(self, error_msg: str, prompt: str) -> Dict[str, Any]:
        """Create an error response.

        Args:
            error_msg: Error message
            prompt: Original prompt

        Returns:
            Error response dictionary
        """
        return {
            "success": False,
            "error": error_msg,
            "prompt": prompt,
        }

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
            logging.error("No images were generated")
            return self._create_error_response("No images were generated", prompt)

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
            logging.info(f"Generating image with prompt: {prompt}")

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
                # Generate images according to the API
                response = client_to_use.generate_images(
                    prompt=prompt,
                    number_of_images=int(n),
                    aspect_ratio=aspect_ratio
                    # person_generation="allow_all",
                    # safety_filter_level="block_none",
                )

                # Log the response type for debugging
                logging.info(f"Imagen response type: {type(response)}")

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
