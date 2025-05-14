import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import html2text
import httpx

from rtl_rag_chatbot_api.common.base_handler import BaseRAGHandler
from rtl_rag_chatbot_api.common.chroma_manager import ChromaDBManager
from rtl_rag_chatbot_api.common.models import DocumentChunk

logger = logging.getLogger(__name__)


class ConfluenceHandler(BaseRAGHandler):
    """Handles interaction with Confluence via the Atlassian MCP server."""

    def __init__(self, configs, gcs_handler=None):
        """Initializes the ConfluenceHandler.

        Args:
            configs: Configuration object containing Confluence MCP settings
            gcs_handler: Optional GCS handler for storage operations
        """
        # Call parent class constructor
        super().__init__(configs, gcs_handler)

        # Store Confluence-specific config
        self.mcp_config = configs.confluence_mcp

        # Check if Confluence integration is configured
        if not self.mcp_config.url or not self.mcp_config.api_key:
            logger.warning(
                "Confluence MCP URL or API Key not configured. Confluence integration will be disabled."
            )
            self.client = None
            self.is_enabled = False
            return

        # Set up HTTP client with authentication
        headers = {
            "Accept": "application/json",
            # Use 'x-api-key' or 'Authorization: Bearer' depending on MCP server config
            "x-api-key": self.mcp_config.api_key,
        }

        self.client = httpx.AsyncClient(
            base_url=self.mcp_config.url, headers=headers, timeout=60.0
        )

        # Set up HTML to text converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        self.html_converter.ignore_images = True
        self.is_enabled = True

        # Initialize a ChromaDB collection for Confluence pages
        self.chroma_manager = ChromaDBManager()
        self.collection_name = "confluence_collection"

        logger.info(
            f"ConfluenceHandler initialized. MCP URL: {self.mcp_config.url}, "
            f"Spaces: {self.mcp_config.space_keys or 'All'}"
        )

    async def close(self):
        """Closes the httpx client."""
        if self.client:
            await self.client.aclose()
            logger.info("ConfluenceHandler client closed.")

    def _convert_html_to_text(self, html_content: str) -> str:
        """Converts HTML content to plain text."""
        try:
            return self.html_converter.handle(html_content)
        except Exception as e:
            logger.error(f"Error converting HTML to text: {e}")
            return ""  # Return empty string on conversion error

    # --- Placeholder methods for core functionality ---

    async def get_page_content(self, page_id: str) -> Optional[Dict[str, Any]]:
        """Fetches the content and metadata of a specific Confluence page.

        Args:
            page_id: The ID of the Confluence page.

        Returns:
            A dictionary containing page details (id, title, body, version, etc.) or None if fetch fails.
        """
        if not self.is_enabled:
            logger.warning("Confluence integration is not enabled.")
            return None

        logger.debug(f"Fetching content for page ID: {page_id}")
        try:
            # Call the MCP server endpoint for content
            endpoint = f"/confluence/rest/api/content/{page_id}?expand=body.storage,version,space"
            response = await self.client.get(endpoint)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error fetching page {page_id}: {e.response.status_code} - {e.response.text}"
            )
            return None
        except Exception as e:
            logger.error(f"Error fetching page {page_id}: {str(e)}")
            return None

    async def search_pages(
        self, query: str, space_keys: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Searches for Confluence pages using a query string (CQL).

        Args:
            query: The search query (e.g., title ~ "report" OR text ~ "dashboard").
            space_keys: Optional list of space keys to limit the search.

        Returns:
            A list of page results.
        """
        if not self.is_enabled:
            logger.warning("Confluence integration is not enabled.")
            return []

        try:
            # Build CQL query based on input query and space_keys
            search_terms = " OR ".join([f'text ~ "{term}"' for term in query.split()])
            # Add title search for better results
            title_terms = " OR ".join([f'title ~ "{term}"' for term in query.split()])
            cql = f"({search_terms}) OR ({title_terms})"

            # Add space filter if specified
            space_filter = ""
            if space_keys and len(space_keys) > 0:
                space_list = ",".join([f'"{key}"' for key in space_keys])
                space_filter = f" AND space IN ({space_list})"
                cql += space_filter

            # Default to spaces from config if no spaces specified
            elif self.mcp_config.space_keys and len(self.mcp_config.space_keys) > 0:
                space_list = ",".join(
                    [f'"{key}"' for key in self.mcp_config.space_keys]
                )
                space_filter = f" AND space IN ({space_list})"
                cql += space_filter

            # URL encode the CQL query
            cql_param = httpx.QueryParams(
                {"cql": cql, "limit": "25", "expand": "space"}
            )
            logger.debug(f"Searching with CQL: {cql}")

            # Make the request to the MCP server
            response = await self.client.get(
                "/confluence/rest/api/content/search", params=cql_param
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            logger.info(f"Found {len(results)} pages matching query: '{query}'")
            return results
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error searching pages: {e.response.status_code} - {e.response.text}"
            )
            return []
        except Exception as e:
            logger.error(f"Error searching pages: {str(e)}")
            return []

    async def process_page_to_chunks(
        self, page_data: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Processes a Confluence page dictionary into text chunks for embedding.

        Args:
            page_data: The dictionary representing a Confluence page (from get_page_content or search_pages).

        Returns:
            A list of DocumentChunk objects ready for embedding.
        """
        if (
            not page_data
            or "body" not in page_data
            or "storage" not in page_data["body"]
        ):
            return []

        page_id = page_data.get("id")
        title = page_data.get("title", "Untitled")
        html_content = page_data["body"]["storage"].get("value", "")
        space_key = page_data.get("space", {}).get("key", "UNKNOWN")
        version = page_data.get("version", {}).get("number", 0)
        last_modified = page_data.get("version", {}).get("when", "")
        web_ui_link = page_data.get("_links", {}).get("webui", "")

        plain_text = self._convert_html_to_text(html_content)

        if not plain_text or len(plain_text.split()) < 10:  # Basic quality check
            logger.warning(
                f"Skipping page {page_id} ('{title}') due to insufficient content."
            )
            return []

        # Simple chunking by paragraphs (can be enhanced with more sophisticated splitting)
        paragraphs = [p for p in plain_text.split("\n\n") if p.strip()]

        # Combine smaller paragraphs to avoid too-small chunks
        chunks = []
        current_chunk = ""
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < 2000:  # Aim for ~2000 char chunks
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"

        # Add the last chunk if it contains content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # If no chunks were created (e.g., very short page), use the whole text as one chunk
        if not chunks and plain_text.strip():
            chunks = [plain_text.strip()]

        doc_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"confluence_{page_id}_v{version}_chunk{i}"
            metadata = {
                "source": "confluence",
                "page_id": page_id,
                "title": title,
                "space_key": space_key,
                "version": version,
                "last_modified": last_modified,
                "url": f"{self.mcp_config.url.rstrip('/')}{web_ui_link}"
                if web_ui_link
                else None,
                "chunk_index": i,
            }
            doc_chunks.append(
                DocumentChunk(id=chunk_id, text=chunk_text, metadata=metadata)
            )

        return doc_chunks

    async def search_and_retrieve_chunks(
        self, query: str, space_keys: Optional[List[str]] = None, n_results: int = 5
    ) -> List[DocumentChunk]:
        """Search for relevant Confluence pages and return processed chunks.

        This method can be used in two ways:
        1. If embeddings are precomputed: Use vector search in ChromaDB
        2. If embeddings are not precomputed: Search Confluence directly and process on-the-fly

        Returns:
            List of document chunks relevant to the query
        """
        # Check if we have a populated Confluence collection in ChromaDB
        if await self._has_embeddings():
            logger.info("Using pre-computed embeddings to retrieve chunks")
            # Get embeddings from Azure OpenAI for the query
            # This requires integration with embedding_handler
            # TODO: Implement integration with embedding model
            # Return most relevant chunks
            return []  # Placeholder
        else:
            logger.info(
                "No pre-computed embeddings found, searching Confluence directly"
            )
            # Search for pages matching the query
            search_results = await self.search_pages(query, space_keys)

            # For each result, get the full page content and process into chunks
            all_chunks = []
            for page_summary in search_results[:5]:  # Limit to top 5 for performance
                page_id = page_summary.get("id")
                if not page_id:
                    continue

                # Get full page content
                page_data = await self.get_page_content(page_id)
                if not page_data:
                    continue

                # Process page into chunks
                page_chunks = await self.process_page_to_chunks(page_data)
                all_chunks.extend(page_chunks)

            # Return the chunks (up to n_results)
            return all_chunks[:n_results]

    async def _has_embeddings(self) -> bool:
        """Check if we have pre-computed Confluence embeddings in ChromaDB."""
        # TODO: Implement check for existing collection with data
        return False  # For now, assume we're always searching directly

    async def sync_incremental(self):
        """Fetches pages modified since the last sync and updates the vector store."""
        if not self.is_enabled:
            logger.warning("Cannot sync, Confluence integration is disabled.")
            return

        logger.info(f"Starting incremental Confluence sync at {datetime.now()}")
        try:
            # Get timestamp of last sync from state file
            last_sync_time = await self._get_last_sync_time()

            # Determine which spaces to sync (from config or all)
            space_keys = self.mcp_config.space_keys

            for space_key in space_keys or []:
                logger.info(f"Syncing space: {space_key}")

                # Build CQL query for pages modified since last sync
                date_str = last_sync_time.strftime("%Y-%m-%d %H:%M")
                cql = f'space = "{space_key}" AND lastModified > "{date_str}"'

                # Get modified pages
                modified_pages = await self._search_with_cql(cql)
                logger.info(
                    f"Found {len(modified_pages)} modified pages in space {space_key}"
                )

                # Process each page
                for page_summary in modified_pages:
                    page_id = page_summary.get("id")
                    if not page_id:
                        continue

                    # Get full page content
                    page_data = await self.get_page_content(page_id)
                    if not page_data:
                        continue

                    # Process page into chunks and store in ChromaDB (future implementation)
                    _ = await self.process_page_to_chunks(page_data)
                    # TODO: Implement storage in ChromaDB

            # Update last sync time
            await self._update_last_sync_time()

            logger.info("Incremental Confluence sync completed successfully")
        except Exception as e:
            logger.error(f"Error during incremental sync: {str(e)}")

    async def _search_with_cql(self, cql: str) -> List[Dict[str, Any]]:
        """Helper method to search Confluence with raw CQL."""
        try:
            cql_param = httpx.QueryParams({"cql": cql, "limit": "100"})
            response = await self.client.get(
                "/confluence/rest/api/content/search", params=cql_param
            )
            response.raise_for_status()
            return response.json().get("results", [])
        except Exception as e:
            logger.error(f"Error searching with CQL: {str(e)}")
            return []

    async def _get_last_sync_time(self) -> datetime:
        """Get the timestamp of the last sync from state file."""
        # Create state directory if it doesn't exist
        state_dir = os.path.join(os.path.dirname(__file__), "..", "..", "state")
        os.makedirs(state_dir, exist_ok=True)

        state_file = os.path.join(state_dir, "confluence_sync_state.json")

        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                    last_sync = state.get("last_sync")
                    if last_sync:
                        return datetime.fromisoformat(last_sync)
            except Exception as e:
                logger.error(f"Error reading sync state: {str(e)}")

        # Default to 7 days ago if no state file or error
        return datetime.now() - timedelta(days=7)

    async def _update_last_sync_time(self):
        """Update the timestamp of the last sync in the state file."""
        state_dir = os.path.join(os.path.dirname(__file__), "..", "..", "state")
        os.makedirs(state_dir, exist_ok=True)

        state_file = os.path.join(state_dir, "confluence_sync_state.json")

        try:
            state = {"last_sync": datetime.now().isoformat()}
            with open(state_file, "w") as f:
                json.dump(state, f)
        except Exception as e:
            logger.error(f"Error updating sync state: {str(e)}")


# Example usage (for testing/development)
async def main():
    from dotenv import load_dotenv

    load_dotenv()
    # Make sure CONFLUENCE_MCP_URL and CONFLUENCE_MCP_API_KEY are in your .env
    # This is just example code - in practice, a Config instance would be created from env vars
    from configs.app_config import Config

    sample_config = Config()  # This would load from env vars
    handler = ConfluenceHandler(sample_config)
    if handler.is_enabled:
        # Example: Search pages (replace with actual CQL query)
        # search_results = await handler.search_pages(query='title ~ "Project Plan"', space_keys=['DP'])
        # print(f"Found {len(search_results)} pages.")

        # Example: Get content for a specific page ID (replace with a real ID)
        # page_content = await handler.get_page_content("12345678")
        # if page_content:
        #     chunks = handler.process_page_to_chunks(page_content)
        #     print(f"Processed page into {len(chunks)} chunks.")
        #     if chunks:
        #         print(f"First chunk metadata: {chunks[0].metadata}")
        pass  # Add test calls here

    await handler.close()


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
