import json
import logging
from typing import Any, Dict

from openai import AzureOpenAI


class DalleImageGenerator:
    """
    Handles interactions with Azure OpenAI DALL-E 3 model for image generation.
    """

    def __init__(self, configs):
        """
        Initialize the DALL-E 3 image generator with configurations.

        Args:
            configs: Application configuration object
        """
        self.configs = configs
        self.client = None
        self.initialize_client()

    def initialize_client(self):
        """Initialize Azure OpenAI client for DALL-E 3."""
        try:
            # Use config object instead of direct environment variables
            self.client = AzureOpenAI(
                api_key=self.configs.azure_dalle_3.api_key,
                api_version=self.configs.azure_dalle_3.api_version,
                azure_endpoint=self.configs.azure_dalle_3.endpoint,
            )
            logging.info("DALL-E 3 client initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing DALL-E 3 client: {str(e)}")
            raise

    def generate_image(
        self, prompt: str, size: str = "1024x1024", n: int = 1
    ) -> Dict[str, Any]:
        """
        Generate an image based on the provided prompt using DALL-E 3.

        Args:
            prompt (str): The text prompt to generate an image from
            size (str): Image size (default: "1024x1024")
            n (int): Number of images to generate (default: 1)

        Returns:
            Dict[str, Any]: Response containing the generated image URLs and other metadata
            Uses optimized response format with single image_urls array to reduce memory usage

        Raises:
            Exception: If there's an error in generating the image
        """
        try:
            logging.info(f"Generating image with prompt: {prompt}")

            result = self.client.images.generate(
                model=self.configs.azure_dalle_3.model_name,
                prompt=prompt,
                n=n,
                size=size,
            )

            # Extract response data
            response_data = json.loads(result.model_dump_json())

            # Extract image URLs - DALL-E typically returns one image
            image_urls = [item["url"] for item in response_data["data"]]

            # Log memory optimization info
            logging.info(
                f"Optimized DALL-E response: returning {len(image_urls)} URLs in single array"
            )

            return {
                "success": True,
                "is_base64": False,  # DALL-E returns HTTP URLs, not base64 data
                "image_urls": image_urls,  # Single source of truth for image URLs
                "prompt": prompt,
                "model": self.configs.azure_dalle_3.model_name,
                "size": size,
            }

        except Exception as e:
            logging.error(f"Error generating image with DALL-E 3: {str(e)}")
            return {"success": False, "error": str(e), "prompt": prompt}
