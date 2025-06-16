import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from rtl_rag_chatbot_api.chatbot.dalle_handler import DalleImageGenerator
from rtl_rag_chatbot_api.chatbot.imagen_handler import ImagenGenerator


class CombinedImageGenerator:
    """
    Handles simultaneous image generation from multiple models (DALL-E and Imagen).
    Simply uses instances of the individual generators without duplicating their logic.
    """

    def __init__(self, configs, dalle_generator=None, imagen_generator=None):
        """
        Initialize combined image generator with configurations.

        Args:
            configs: Application configuration object
            dalle_generator: Optional existing DalleImageGenerator instance
            imagen_generator: Optional existing ImagenGenerator instance
        """
        self.configs = configs
        # Use existing generators if provided, otherwise create new ones
        self.dalle_generator = (
            dalle_generator if dalle_generator else DalleImageGenerator(configs)
        )
        self.imagen_generator = (
            imagen_generator if imagen_generator else ImagenGenerator(configs)
        )

    async def generate_images(
        self, prompt: str, size: str = "1024x1024", n: int = 1, **kwargs
    ) -> Dict[str, Any]:
        """
        Generate images from both DALL-E and Imagen models concurrently.
        Directly uses the existing generator classes without duplicating their code.

        Args:
            prompt: Text prompt to generate images from
            size: Image size
            n: Number of images to generate per model
            **kwargs: Additional parameters that will be passed to both generators
                      allowing for future extensibility

        Returns:
            Dict containing results from both models
        """
        try:
            logging.info(f"Generating images with both models for prompt: {prompt}")

            # Run both generators in separate threads since they are not async methods
            # This prevents blocking the event loop while still allowing concurrent execution
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Create async tasks for both model executions
                loop = asyncio.get_event_loop()

                # DALL-E 3 can only generate 1 image per API call, always use n=1
                dalle_future = loop.run_in_executor(
                    executor,
                    lambda: self.dalle_generator.generate_image(
                        prompt=prompt, size=size, n=1, **kwargs  # Force n=1 for DALL-E
                    ),
                )

                # Imagen can generate multiple images as requested
                imagen_future = loop.run_in_executor(
                    executor,
                    lambda: self.imagen_generator.generate_image(
                        prompt=prompt,
                        size=size,
                        n=n,
                        **kwargs,  # Use requested n for Imagen
                    ),
                )

                # Wait for both futures to complete
                dalle_result, imagen_result = await asyncio.gather(
                    dalle_future, imagen_future
                )

            # Combine results
            combined_result = {
                "success": dalle_result.get("success", False)
                or imagen_result.get("success", False),
                "dalle_result": dalle_result,
                "imagen_result": imagen_result,
                "prompt": prompt,
                "models": ["dall-e-3", self.configs.vertexai_imagen.model_name],
                "size": size,
            }

            # Include any additional parameters that were passed
            for key, value in kwargs.items():
                if key not in combined_result:
                    combined_result[key] = value

            return combined_result

        except Exception as e:
            logging.error(f"Error in combined image generation: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "prompt": prompt,
                "models": ["dall-e-3", self.configs.vertexai_imagen.model_name],
            }
