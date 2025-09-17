"""
SharePoint Handler for RTL RAG Chatbot API.

This module provides functionality to connect to SharePoint,
retrieve documents from the RTL Wissens-Hub, and process them
for use with the RAG system.
"""

import hashlib
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import aiofiles
import requests
from fastapi import HTTPException

# SharePoint/Microsoft Graph API libraries
# You'll need to install these with: poetry add office365-rest-python-client msal
try:
    import msal
    from office365.runtime.auth.authentication_context import AuthenticationContext
    from office365.sharepoint.client_context import ClientContext
    from office365.sharepoint.files.file import File
except ImportError:
    logging.warning(
        "SharePoint libraries not installed. Run: poetry add office365-rest-python-client msal"
    )

from rtl_rag_chatbot_api.chatbot.embedding_handler import EmbeddingHandler
from rtl_rag_chatbot_api.common.chroma_manager import ChromaDBManager


class SharePointHandler:
    """
    Handles SharePoint document retrieval and processing for the RAG system.

    This class manages authentication with SharePoint, document retrieval from
    the RTL Wissens-Hub, content extraction, and integration with the RAG system.

    Attributes:
        configs: Configuration object containing necessary settings.
        gcs_handler: Google Cloud Storage handler for cloud operations.
        embedding_handler: Handler for creating and managing embeddings.
    """

    def __init__(self, configs, gcs_handler, embedding_handler=None):
        """
        Initialize the SharePoint handler with necessary configurations and handlers.

        Args:
            configs: Configuration object containing necessary settings.
            gcs_handler: Google Cloud Storage handler for cloud operations.
            embedding_handler: Optional handler for creating embeddings.
        """
        self.configs = configs
        self.gcs_handler = gcs_handler
        self.embedding_handler = embedding_handler or EmbeddingHandler(
            configs, gcs_handler
        )
        self.chroma_manager = ChromaDBManager()

        # SharePoint connection settings - these should be added to your environment variables
        self.tenant_id = os.getenv(
            "SHAREPOINT_TENANT_ID", "efce8346-592b-4b6e-b1c2-0fd07bd5e442"
        )
        self.client_id = os.getenv("SHAREPOINT_CLIENT_ID")
        self.client_secret = os.getenv("SHAREPOINT_CLIENT_SECRET")
        self.site_url = os.getenv(
            "SHAREPOINT_SITE_URL", "https://rtlde.sharepoint.com/sites/Wissens-Hub"
        )

        # Initialize auth context
        self.auth_context = None
        self.client_context = None

        # Document tracking
        self.processed_docs = {}

    def authenticate(self) -> bool:
        """
        Authenticate with SharePoint using app-only authentication.

        Returns:
            bool: True if authentication was successful, False otherwise.
        """
        try:
            if not self.client_id or not self.client_secret:
                logging.error("SharePoint client ID or secret not configured")
                return False

            # Using MSAL for modern authentication
            app = msal.ConfidentialClientApplication(
                self.client_id,
                authority=f"https://login.microsoftonline.com/{self.tenant_id}",
                client_credential=self.client_secret,
            )

            # Acquire token for SharePoint
            scopes = [f"{self.site_url}/.default"]
            result = app.acquire_token_for_client(scopes=scopes)

            if "access_token" not in result:
                logging.error(
                    f"Failed to acquire token: {result.get('error_description', 'Unknown error')}"
                )
                return False

            # Initialize SharePoint client context
            self.auth_context = AuthenticationContext(self.site_url)
            self.auth_context.acquire_token_for_app(self.client_id, self.client_secret)
            self.client_context = ClientContext(self.site_url, self.auth_context)

            logging.info("Successfully authenticated with SharePoint")
            return True

        except Exception as e:
            logging.error(f"SharePoint authentication error: {str(e)}")
            return False

    def calculate_document_hash(self, content: bytes) -> str:
        """
        Calculate a unique hash for a document based on its content.

        Args:
            content: The binary content of the document.

        Returns:
            str: The hexadecimal digest of the MD5 hash.
        """
        return hashlib.md5(content).hexdigest()

    async def get_document_list(
        self, folder_path: str = "/Shared Documents"
    ) -> List[Dict]:
        """
        Get a list of documents from the specified SharePoint folder.

        Args:
            folder_path: The relative path to the folder in SharePoint.

        Returns:
            List[Dict]: List of document metadata dictionaries.
        """
        try:
            if not self.client_context:
                if not self.authenticate():
                    raise HTTPException(
                        status_code=401, detail="Failed to authenticate with SharePoint"
                    )

            # Get the web and then the folder
            web = self.client_context.web
            folder = web.get_folder_by_server_relative_url(folder_path)
            files = folder.files
            self.client_context.load(files)
            self.client_context.execute_query()

            documents = []
            for file in files:
                # Skip system files and non-document files
                if file.name.startswith("~$") or file.name.startswith("."):
                    continue

                # Get file properties
                doc_info = {
                    "name": file.name,
                    "url": file.serverRelativeUrl,
                    "size": file.length,
                    "modified": file.timeLastModified.strftime("%Y-%m-%d %H:%M:%S"),
                    "created": file.timeCreated.strftime("%Y-%m-%d %H:%M:%S"),
                }
                documents.append(doc_info)

            return documents

        except Exception as e:
            logging.error(f"Error getting document list: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to get document list: {str(e)}"
            )

    async def download_document(self, document_url: str) -> Tuple[bytes, str]:
        """
        Download a document from SharePoint.

        Args:
            document_url: The server-relative URL of the document.

        Returns:
            Tuple[bytes, str]: The document content and filename.
        """
        try:
            if not self.client_context:
                if not self.authenticate():
                    raise HTTPException(
                        status_code=401, detail="Failed to authenticate with SharePoint"
                    )

            # Get the file by server relative URL
            file_obj = self.client_context.web.get_file_by_server_relative_url(
                document_url
            )
            self.client_context.load(file_obj)
            self.client_context.execute_query()

            # Download file content
            response = File.open_binary(self.client_context, document_url)

            return response.content, file_obj.name

        except Exception as e:
            logging.error(f"Error downloading document: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to download document: {str(e)}"
            )

    async def process_document(self, document_url: str, username: str) -> Dict:
        """
        Process a SharePoint document for use with the RAG system.

        Args:
            document_url: The server-relative URL of the document.
            username: The username of the user processing the document.

        Returns:
            Dict: Dictionary containing processing results.
        """
        try:
            # Download the document
            content, filename = await self.download_document(document_url)

            # Calculate hash and generate file_id
            file_hash = self.calculate_document_hash(content)
            file_id = f"sp_{file_hash}"

            # Check if document has already been processed
            existing_file_id = await self.find_existing_document_by_hash(file_hash)
            if existing_file_id:
                logging.info(
                    f"Document already processed: {filename} with ID {existing_file_id}"
                )

                # Update file_info.json with the new username
                self.gcs_handler.update_file_info(
                    existing_file_id, {"username": username}
                )

                # Check if local embeddings exist
                azure_path = f"./chroma_db/{existing_file_id}/azure"
                local_exists = os.path.exists(azure_path) and os.path.exists(
                    os.path.join(azure_path, "chroma.sqlite3")
                )

                if not local_exists:
                    self.gcs_handler.download_files_from_folder_by_id(existing_file_id)

                return {
                    "file_id": existing_file_id,
                    "is_sharepoint": True,
                    "message": "Document already exists and has embeddings.",
                    "status": "existing",
                    "original_filename": filename,
                }

            # Save document to temp file
            os.makedirs("local_data", exist_ok=True)
            temp_file_path = f"local_data/{file_id}_{filename}"
            async with aiofiles.open(temp_file_path, "wb") as buffer:
                await buffer.write(content)

            # Create metadata
            metadata = {
                "file_id": file_id,
                "file_hash": file_hash,
                "original_filename": filename,
                "is_sharepoint": True,
                "sharepoint_url": document_url,
                "username": [username],
                "embeddings_status": "pending",
            }

            # Store metadata
            self.gcs_handler.temp_metadata = metadata

            # Return processing result
            return {
                "file_id": file_id,
                "is_sharepoint": True,
                "message": "Document processed and ready for embedding creation.",
                "status": "success",
                "temp_file_path": temp_file_path,
                "original_filename": filename,
            }

        except Exception as e:
            logging.error(f"Error processing document: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to process document: {str(e)}"
            )

    async def find_existing_document_by_hash(self, file_hash: str) -> Optional[str]:
        """
        Find an existing document by its hash.

        Args:
            file_hash: The hash of the document to find.

        Returns:
            Optional[str]: The file_id if found, None otherwise.
        """
        try:
            # List all blobs with the file_info.json pattern
            blobs = list(
                self.gcs_handler.bucket.list_blobs(
                    prefix=f"{self.configs.gcp_resource.gcp_embeddings_folder}/"
                )
            )
            file_info_blobs = [b for b in blobs if b.name.endswith("file_info.json")]

            for blob in file_info_blobs:
                # Download and parse the file_info.json
                content = blob.download_as_string()
                file_info = json.loads(content)

                # Check if hash matches
                if file_info.get("file_hash") == file_hash:
                    # Extract file_id from the blob name
                    # Pattern: file-embeddings/{file_id}/file_info.json
                    parts = blob.name.split("/")
                    if len(parts) >= 2:
                        return parts[1]

            return None

        except Exception as e:
            logging.error(f"Error finding existing document: {str(e)}")
            return None

    async def create_embeddings(self, file_id: str, temp_file_path: str) -> Dict:
        """
        Create embeddings for a processed SharePoint document.

        Args:
            file_id: The ID of the document.
            temp_file_path: The path to the temporary file.

        Returns:
            Dict: Dictionary containing embedding creation results.
        """
        try:
            # Ensure embedding handler is initialized
            if not self.embedding_handler:
                self.embedding_handler = EmbeddingHandler(
                    self.configs, self.gcs_handler
                )

            # Create embeddings
            result = await self.embedding_handler.create_and_upload_embeddings(
                file_id, temp_file_path, is_image=False
            )

            # Update metadata
            self.gcs_handler.update_file_info(
                file_id, {"embeddings_status": "completed"}
            )

            return result

        except Exception as e:
            logging.error(f"Error creating embeddings: {str(e)}")
            # Update metadata with error
            self.gcs_handler.update_file_info(
                file_id, {"embeddings_status": "failed", "error": str(e)}
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to create embeddings: {str(e)}"
            )

    async def search_sharepoint(self, query: str) -> List[Dict]:
        """
        Search for documents in SharePoint using the search API.

        Args:
            query: The search query.

        Returns:
            List[Dict]: List of search results.
        """
        try:
            if not self.client_context:
                if not self.authenticate():
                    raise HTTPException(
                        status_code=401, detail="Failed to authenticate with SharePoint"
                    )

            # Use SharePoint search REST API
            search_url = (
                f"{self.site_url}/_api/search/query?querytext='{query}'&rowlimit=10"
            )

            # Get the access token
            access_token = self.auth_context.acquire_token_for_app(
                self.client_id, self.client_secret
            ).access_token

            # Make the request
            headers = {
                "Accept": "application/json;odata=verbose",
                "Authorization": f"Bearer {access_token}",
            }

            response = requests.get(search_url, headers=headers)
            response.raise_for_status()

            # Parse results
            results = response.json()
            items = (
                results.get("d", {})
                .get("query", {})
                .get("PrimaryQueryResult", {})
                .get("RelevantResults", {})
                .get("Table", {})
                .get("Rows", {})
                .get("results", [])
            )

            search_results = []
            for item in items:
                cells = item.get("Cells", {}).get("results", [])
                result = {}

                for cell in cells:
                    key = cell.get("Key")
                    value = cell.get("Value")
                    if key in [
                        "Title",
                        "Path",
                        "Author",
                        "Size",
                        "Created",
                        "FileExtension",
                    ]:
                        result[key] = value

                if "Path" in result:
                    search_results.append(result)

            return search_results

        except Exception as e:
            logging.error(f"Error searching SharePoint: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to search SharePoint: {str(e)}"
            )

    async def process_sharepoint_folder(self, folder_path: str, username: str) -> Dict:
        """
        Process all documents in a SharePoint folder.

        Args:
            folder_path: The relative path to the folder in SharePoint.
            username: The username of the user processing the documents.

        Returns:
            Dict: Dictionary containing processing results.
        """
        try:
            # Get document list
            documents = await self.get_document_list(folder_path)

            results = {
                "total": len(documents),
                "processed": 0,
                "skipped": 0,
                "failed": 0,
                "documents": [],
            }

            # Process each document
            for doc in documents:
                try:
                    result = await self.process_document(doc["url"], username)

                    # Create embeddings if new document
                    if result["status"] == "success":
                        embedding_result = await self.create_embeddings(
                            result["file_id"], result["temp_file_path"]
                        )
                        result["embedding_status"] = embedding_result.get(
                            "status", "unknown"
                        )

                    results["documents"].append(
                        {
                            "name": doc["name"],
                            "file_id": result["file_id"],
                            "status": result["status"],
                        }
                    )

                    if result["status"] == "success":
                        results["processed"] += 1
                    elif result["status"] == "existing":
                        results["skipped"] += 1

                except Exception as e:
                    logging.error(f"Error processing document {doc['name']}: {str(e)}")
                    results["failed"] += 1
                    results["documents"].append(
                        {"name": doc["name"], "status": "failed", "error": str(e)}
                    )

            return results

        except Exception as e:
            logging.error(f"Error processing SharePoint folder: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to process SharePoint folder: {str(e)}"
            )


# Example usage function
async def example_usage(configs, gcs_handler):
    """
    Example of how to use the SharePoint handler.

    Args:
        configs: Configuration object.
        gcs_handler: GCS handler instance.
    """
    # Initialize SharePoint handler
    sp_handler = SharePointHandler(configs, gcs_handler)

    # Authenticate
    if not sp_handler.authenticate():
        print("Failed to authenticate with SharePoint")
        return

    # List documents in the root folder
    documents = await sp_handler.get_document_list("/Shared Documents")
    print(f"Found {len(documents)} documents")

    # Process a document
    if documents:
        result = await sp_handler.process_document(documents[0]["url"], "example_user")
        print(f"Processed document: {result}")

        # Create embeddings
        if result["status"] == "success":
            embedding_result = await sp_handler.create_embeddings(
                result["file_id"], result["temp_file_path"]
            )
            print(f"Created embeddings: {embedding_result}")

    # Search SharePoint
    search_results = await sp_handler.search_sharepoint("project documentation")
    print(f"Search results: {search_results}")
