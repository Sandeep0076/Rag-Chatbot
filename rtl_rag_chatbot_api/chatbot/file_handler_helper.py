"""
Helper functions for processing URLs individually in the FileHandler.
This module contains utility functions to support processing URLs individually
rather than concatenating them into a single document.
"""
import logging
import os
import uuid
from typing import List

import aiofiles

# chat with URL is deprecated, but keeping it for backward compatibility


async def process_single_url(
    self, url: str, username: str, background_tasks, embedding_handler
) -> dict:
    """Process a single URL and create embeddings.

    Args:
        url: The URL to process
        username: The username associated with this request
        background_tasks: FastAPI BackgroundTasks to schedule async tasks
        embedding_handler: Handler for creating embeddings

    Returns:
        Dict containing result information
    """
    try:
        logging.info(f"Processing URL: {url}")

        # Generate unique file_id for this URL
        url_file_id = str(uuid.uuid4())
        file_name = f"{url_file_id}_url_content.txt"
        temp_file_path = f"local_data/{file_name}"
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

        # Initialize website handler
        from rtl_rag_chatbot_api.chatbot.website_handler import WebsiteHandler

        website_handler = WebsiteHandler()

        # Extract content from the URL
        content, title, is_successful = website_handler.extract_content_from_single_url(
            url
        )
        if not is_successful or not content:
            website_handler.cleanup()
            return {
                "file_id": url_file_id,
                "status": "error",
                "message": f"Could not extract content from URL: {url}",
                "is_image": False,
                "is_tabular": False,
                "original_filename": title or "url_content.txt",
                "url": url,
                "temp_file_path": None,
            }

        # Check content quality
        is_substantive, word_count = website_handler.check_content_quality(content)
        if not is_substantive:
            website_handler.cleanup()
            return {
                "file_id": url_file_id,
                "status": "error",
                "message": (
                    f"Die Verarbeitung von {url} ist fehlgeschlagen. "
                    "Bitte versuchen Sie es erneut mit einer anderen Domain/einer anderen URL."
                ),
                "is_image": False,
                "is_tabular": False,
                "original_filename": title or "url_content.txt",
                "url": url,
                "temp_file_path": None,
            }

        # Write content to file
        async with aiofiles.open(temp_file_path, "w", encoding="utf-8") as f:
            # Add header with URL and title if available
            await f.write(f"Source URL: {url}\n")
            if title:
                await f.write(f"Page Title: {title}\n\n")
            else:
                await f.write("\n")

            # Write the main content
            await f.write(content)
            await f.write(f"\n\nWord count: {word_count} words")

        # Calculate hash based on content
        content_hash = self.calculate_file_hash(content.encode("utf-8"))

        # Check if we've already processed this content
        existing_file_id, _ = await self.find_existing_file_by_hash_async(content_hash)

        # If we found existing content with the same hash, reuse it
        if existing_file_id:
            logging.info(
                f"Found existing content with ID {existing_file_id}, reusing it"
            )
            # Update the file info with the new username
            self.gcs_handler.update_file_info(existing_file_id, {"username": username})
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return {
                "file_id": existing_file_id,
                "status": "existing",
                "message": "Content already processed, reusing existing file",
                "is_image": False,
                "is_tabular": False,
                "is_url": True,
                "url": url,
                "original_filename": title or "url_content.txt",
                "temp_file_path": None,
            }

        # Create metadata for the file
        friendly_name = title or f"Content from {url.split('//')[1].split('/')[0]}"
        metadata = {
            "file_hash": content_hash,
            "original_filename": f"{friendly_name}.txt",
            "username": [username],
            "is_url": True,
            "url": url,
            "title": title,
            "file_id": url_file_id,
            "word_count": word_count,
        }

        # Save metadata
        self.gcs_handler.temp_metadata = metadata

        # Import here to avoid circular imports
        from rtl_rag_chatbot_api.app import SessionLocal, create_embeddings_background

        # Schedule background task to create embeddings
        background_tasks.add_task(
            create_embeddings_background,
            url_file_id,
            temp_file_path,
            embedding_handler,
            self.configs,
            SessionLocal,
            [username],
        )

        # Clean up the website handler
        website_handler.cleanup()

        return {
            "file_id": url_file_id,
            "status": "success",
            "message": f"URL processed successfully: {url}",
            "is_image": False,
            "is_tabular": False,
            "is_url": True,
            "url": url,
            "original_filename": friendly_name,
            "temp_file_path": temp_file_path,
        }

    except Exception as e:
        logging.error(f"Error processing URL {url}: {str(e)}")
        return {
            "file_id": None,
            "status": "error",
            "message": f"Error processing URL {url}: {str(e)}",
            "is_image": False,
            "is_tabular": False,
            "url": url,
            "temp_file_path": None,
        }


async def process_urls_individually(
    self,
    urls_text: str,
    username: str,
    temp_file_id: str,
    background_tasks,
    embedding_handler,
) -> dict:
    """Process multiple URLs and extract content from each individually.

    Args:
        urls_text: Text containing URLs separated by commas or newlines
        username: The username for this request
        temp_file_id: A temporary ID (not used in this implementation)
        background_tasks: FastAPI BackgroundTasks for async operations
        embedding_handler: Handler for creating embeddings

    Returns:
        Dict containing status and results for all URLs
    """
    try:
        # Parse the URL list
        if "\n" in urls_text:
            # Split by newlines if they exist
            url_list = [url.strip() for url in urls_text.split("\n") if url.strip()]
        else:
            # Otherwise split by commas
            url_list = [url.strip() for url in urls_text.split(",") if url.strip()]

        if not url_list:
            return {"status": "error", "message": "No valid URLs provided"}

        logging.info(f"Processing {len(url_list)} URLs individually: {url_list}")

        # Process each URL individually
        results = []
        for url in url_list:
            result = await process_single_url(
                self, url, username, background_tasks, embedding_handler
            )
            results.append(result)

        # Get successful URLs
        successful_urls = [
            r for r in results if r["status"] == "success" or r["status"] == "existing"
        ]

        # Return a summary response
        if not successful_urls:
            return {
                "status": "error",
                "message": "Die angegebenen URLs konnten nicht verarbeitet werden.\n"
                "Bitte versuchen Sie es erneut mit anderen.",
                "url_results": results,
            }

        return {
            "status": "success",
            "message": f"Processed {len(successful_urls)} of {len(url_list)} URLs successfully",
            "file_ids": [r["file_id"] for r in successful_urls],
            "url_results": results,
        }

    except Exception as e:
        logging.error(f"Error in process_urls_individually: {str(e)}")
        return {"status": "error", "message": f"Error processing URLs: {str(e)}"}


def parse_url_list(urls_text: str) -> List[str]:
    """Parse a string containing URLs into a list of URLs.

    Args:
        urls_text: Text containing URLs separated by commas or newlines

    Returns:
        List of cleaned URLs
    """
    if "\n" in urls_text:
        # Split by newlines if they exist
        return [url.strip() for url in urls_text.split("\n") if url.strip()]
    else:
        # Otherwise split by commas
        return [url.strip() for url in urls_text.split(",") if url.strip()]
