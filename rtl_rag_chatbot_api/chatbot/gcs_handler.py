import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

import google.auth
import google.oauth2.credentials
from google.cloud import storage

from rtl_rag_chatbot_api.chatbot.utils.encryption import decrypt_file


class GCSHandler:
    """
    Handles interactions with Google Cloud Storage (GCS) for file operations.

    This class provides methods for downloading, uploading, and managing files in Google Cloud Storage.
    It includes functionality  cleaning up local storage, and finding
    existing files based on name or hash.

    Attributes:
        configs (Config): Configuration object containing GCP resource settings.
        credentials (google.oauth2.credentials.Credentials): Google OAuth credentials.
        _storage_client (storage.Client): Google Cloud Storage client.
        bucket (storage.Bucket): GCS bucket object.
        bucket_name (str): Name of the GCS bucket.
    """

    def __init__(self, _configs):
        """
        Initializes the GCSHandler with the given configurations.

        Args:
            _configs (Config): Configuration object containing GCP resource settings.
        """
        self.configs = _configs
        self.credentials = None
        self.temp_metadata = None

        if os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN", None) is not None:
            logging.info("Found GOOGLE_OAUTH_ACCESS_TOKEN token, now using it.")
            self.credentials = google.oauth2.credentials.Credentials(
                os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN"), refresh_token=""
            )

        try:
            self._storage_client = storage.Client(
                self.configs.gcp_resource.gcp_project, credentials=self.credentials
            )
        except Exception:
            # Used for production
            self._storage_client = storage.Client(self.configs.gcp_resource.gcp_project)

        self.bucket = self._storage_client.get_bucket(
            self.configs.gcp_resource.bucket_name
        )
        self.bucket_name = self.configs.gcp_resource.bucket_name

    def download_files_from_gcs(
        self, bucket_name: str, source_blob_name: str, destination_file_path: str
    ):
        """
        A generic method to download files from GCS.
        """
        bucket = self._storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        # ensure the directory exists
        os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)

        # and download the blob to a file
        blob.download_to_filename(destination_file_path)

    def download_files_from_folder_by_id(self, file_id):
        """Download files maintaining the original structure."""
        prefix = f"file-embeddings/{file_id}/"
        blobs = list(self.bucket.list_blobs(prefix=prefix))

        if not blobs:
            logging.warning(f"No embeddings found for file ID: {file_id}")
            return

        for blob in blobs:
            if blob.name.endswith("/"):
                continue

            # Maintain the exact same structure as in GCS
            relative_path = blob.name[len(prefix) :]
            local_path = os.path.join("chroma_db", file_id, relative_path)

            # Handle encrypted files
            if relative_path == "tabular_data.db.encrypted":
                # Download to temporary encrypted file
                encrypted_path = local_path
                local_path = os.path.join("chroma_db", file_id, "tabular_data.db")

                # Ensure directory exists
                os.makedirs(os.path.dirname(encrypted_path), exist_ok=True)

                # Download and decrypt
                blob.download_to_filename(encrypted_path)
                try:
                    decrypt_file(encrypted_path)
                    logging.info(f"Decrypted {blob.name} to {local_path}")
                finally:
                    # Clean up encrypted file
                    if os.path.exists(encrypted_path):
                        os.remove(encrypted_path)
                        logging.info(f"Cleaned up encrypted file {encrypted_path}")
            else:
                # Handle regular files
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                blob.download_to_filename(local_path)
                logging.info(f"Downloaded {blob.name} to {local_path}")

        logging.info(f"Finished downloading all files for file ID: {file_id}")

    def cleanup_local_files(self, exclude=[]):
        folders_to_clean = ["chroma_db", "local_data", "processed_data"]
        for folder in folders_to_clean:
            if folder in exclude:
                continue
            folder_path = os.path.join(os.getcwd(), folder)
            if os.path.exists(folder_path):
                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                logging.info(f"Cleaned up contents of {folder}")
            else:
                logging.info(f"{folder} does not exist, skipping cleanup")

    def upload_db_files_to_gcs(self, file_id: str, embedding_type: str):
        """Upload files maintaining consistent folder structure."""
        try:
            base_path = f"./chroma_db/{file_id}/{embedding_type}"
            gcs_base_path = f"file-embeddings/{file_id}/{embedding_type}"

            # Upload embeddings
            files_to_upload = {}
            for file in Path(base_path).rglob("*"):
                if file.is_file():
                    relative_path = file.relative_to(base_path)
                    gcs_object_name = f"{gcs_base_path}/{relative_path}"
                    files_to_upload[str(relative_path)] = (str(file), gcs_object_name)

            self.upload_to_gcs(self.bucket_name, files_to_upload)
            logging.info(f"Uploaded embeddings to {gcs_base_path}")
        except Exception as e:
            logging.error(f"Error uploading to GCS: {str(e)}")
        raise

    def check_file_exists(self, blob_path: str) -> bool:
        """
        Check if a file exists in Google Cloud Storage.

        Args:
            blob_path (str): The path to the blob in the bucket.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        try:
            blob = self.bucket.blob(blob_path)
            return blob.exists()
        except Exception as e:
            logging.error(f"Error checking if file exists in GCS: {str(e)}")
            return False

    def _store_file_info_json_locally(self, blob_path: str, content: dict) -> None:
        """
        Store file_info.json locally in the chroma_db directory.

        Args:
            blob_path (str): Path of the blob in GCS
            content (dict): Content to store in the file
        """
        local_path_parts = blob_path.split("/")
        if len(local_path_parts) >= 3:
            file_id = local_path_parts[1]
            local_dir = f"./chroma_db/{file_id}"
            local_path = f"{local_dir}/file_info.json"
            os.makedirs(local_dir, exist_ok=True)
            logging.info(f"Writing file_info.json locally to {local_path}")
            try:
                with open(local_path, "w") as f:
                    json.dump(content, f, indent=2)
            except Exception as e:
                logging.error(f"Error writing file_info.json locally: {str(e)}")

    def _store_encrypted_file_locally(self, source_path: str, blob_path: str) -> None:
        """
        Keep a local copy of an encrypted file in chroma_db directory.

        Args:
            source_path (str): Path to the source file
            blob_path (str): Path of the blob in GCS
        """
        local_path_parts = blob_path.split("/")
        if len(local_path_parts) >= 3 and "file-embeddings" in blob_path:
            file_id = local_path_parts[1]
            filename = local_path_parts[-1]
            local_dir = f"./chroma_db/{file_id}"
            local_path = f"{local_dir}/{filename}"
            os.makedirs(local_dir, exist_ok=True)
            try:
                shutil.copy2(source_path, local_path)
                logging.info(f"Copied encrypted file to local storage: {local_path}")
            except Exception as e:
                logging.error(f"Error copying encrypted file locally: {str(e)}")

    def _upload_single_item(
        self, bucket, source: Union[str, dict], destination_blob_name: str
    ) -> None:
        """
        Upload a single item to GCS bucket and handle local storage if needed.

        Args:
            bucket: GCS bucket object
            source: Source content (file path or dictionary)
            destination_blob_name: Target blob path in GCS
        """
        blob = bucket.blob(destination_blob_name)

        # Handle file_info.json (both GCS upload and local storage)
        if (
            isinstance(source, dict)
            and "file-embeddings" in destination_blob_name
            and destination_blob_name.endswith("/file_info.json")
        ):
            self._store_file_info_json_locally(destination_blob_name, source)

        # Upload to GCS based on source type
        if isinstance(source, dict):
            blob.upload_from_string(
                data=json.dumps(source), content_type="application/json"
            )
        elif isinstance(source, str):
            # Handle encrypted files (keep local copy)
            if destination_blob_name.endswith(".encrypted"):
                self._store_encrypted_file_locally(source, destination_blob_name)

            # Upload file to GCS
            blob.upload_from_filename(source)

        logging.info(f"Uploaded to {destination_blob_name}")

    def upload_to_gcs(
        self,
        bucket_name: str,
        source: Union[str, dict, Dict[str, Union[str, dict, tuple]]],
        destination_blob_name: Optional[str] = None,
    ):
        """
        Upload content to Google Cloud Storage.

        Args:
            bucket_name (str): Name of the GCS bucket
            source: Content to upload, can be:
                - A string (filepath)
                - A dictionary (JSON content)
                - A dictionary of (source, destination) tuples for batch upload
            destination_blob_name (Optional[str]): Target path in GCS, not required for batch uploads
        """
        bucket = self._storage_client.bucket(bucket_name)

        if isinstance(source, dict) and destination_blob_name is None:
            # Multiple upload case - process each item
            for _, (item_source, item_destination) in source.items():
                self._upload_single_item(bucket, item_source, item_destination)
        else:
            # Single upload case
            self._upload_single_item(bucket, source, destination_blob_name)

    def find_existing_file_by_hash(self, file_hash):
        try:
            logging.info(f"Searching for existing file with hash: {file_hash}")
            blobs = self._storage_client.list_blobs(
                self.bucket_name, prefix="file-embeddings/"
            )

            for blob in blobs:
                if blob.name.endswith("/file_info.json"):
                    file_info = json.loads(blob.download_as_bytes().decode("utf-8"))
                    if file_info.get("file_hash") == file_hash:
                        return file_info.get("file_id")

            logging.info(f"No file found with hash: {file_hash}")
            return None
        except Exception as e:
            logging.error(f"Error in find_existing_file_by_hash: {str(e)}")
            return None

    def get_file_info(self, file_id: str):
        blob = self.bucket.blob(f"file-embeddings/{file_id}/file_info.json")
        if blob.exists():
            return json.loads(blob.download_as_bytes().decode("utf-8"))
        return {}

    def update_file_info(self, file_id: str, new_info: dict):
        blob = self.bucket.blob(f"file-embeddings/{file_id}/file_info.json")
        if blob.exists():
            current_info = json.loads(blob.download_as_bytes().decode("utf-8"))

            # Handle username conversion to array if it exists in new_info
            if "username" in new_info:
                current_username = new_info["username"]

                # If current_username is already a list (e.g., from URL metadata),
                # we need to handle it differently than individual usernames
                if isinstance(current_username, list):
                    # If it's already a list, use it as-is for new files or merge for existing files
                    if "username" in current_info:
                        existing_username = current_info["username"]
                        if not isinstance(existing_username, list):
                            existing_username = [existing_username]
                        # Merge the lists, preserving duplicates (tracks frequency)
                        existing_username.extend(current_username)
                        new_info["username"] = existing_username
                    else:
                        # No existing username, use the provided list as-is
                        new_info["username"] = current_username
                else:
                    # Handle single username (traditional approach for PDFs)
                    if "username" in current_info:
                        existing_username = current_info["username"]

                        if not isinstance(existing_username, list):
                            # If existing username is not already an array, convert to list
                            existing_username = [existing_username]

                        # append user to list, no matter if it's already there
                        # the number of times the user is listed in the array is equivalent to the
                        # number of times they've uploaded the file
                        existing_username.append(current_username)
                        new_info["username"] = existing_username
                    else:
                        # No existing username, set as a single-item array
                        new_info["username"] = [current_username]

            current_info.update(new_info)
            blob.upload_from_string(
                json.dumps(current_info), content_type="application/json"
            )

            # Also update the local copy of file_info.json
            local_dir = f"./chroma_db/{file_id}"
            local_path = f"{local_dir}/file_info.json"
            if os.path.exists(local_path):
                try:
                    os.makedirs(local_dir, exist_ok=True)
                    with open(local_path, "w") as f:
                        json.dump(current_info, f, indent=2)
                    logging.info(f"Updated local file_info.json at {local_path}")
                except Exception as e:
                    logging.error(f"Error updating local file_info.json: {str(e)}")
        else:
            # For new files, convert username to array if present
            if "username" in new_info and not isinstance(new_info["username"], list):
                new_info["username"] = [new_info["username"]]

            blob.upload_from_string(
                json.dumps(new_info), content_type="application/json"
            )

            # Also create the local copy of file_info.json for new files
            local_dir = f"./chroma_db/{file_id}"
            local_path = f"{local_dir}/file_info.json"
            try:
                os.makedirs(local_dir, exist_ok=True)
                with open(local_path, "w") as f:
                    json.dump(new_info, f, indent=2)
                logging.info(f"Created local file_info.json at {local_path}")
            except Exception as e:
                logging.error(f"Error creating local file_info.json: {str(e)}")

    def update_username_list(self, file_id: str, username_list: list):
        """Update the username list directly.

        Args:
            file_id (str): The ID of the file to update
            username_list (list): The new list of usernames
        """
        blob = self.bucket.blob(f"file-embeddings/{file_id}/file_info.json")
        if blob.exists():
            # Get the current file info from GCS
            current_info = json.loads(blob.download_as_bytes().decode("utf-8"))

            # If we have a local copy of the file_info.json, use that for the username list
            # to ensure we don't lose any recently added usernames
            local_path = f"./chroma_db/{file_id}/file_info.json"
            if os.path.exists(local_path):
                try:
                    with open(local_path, "r") as f:
                        local_info = json.load(f)
                        if "username" in local_info and isinstance(
                            local_info["username"], list
                        ):
                            # Use the local username list if it exists as it may be more up-to-date
                            current_username_list = local_info["username"]
                        else:
                            current_username_list = current_info.get("username", [])
                except Exception as e:
                    logging.error(f"Error reading local file_info.json: {str(e)}")
                    # Fallback to the GCS version
                    current_username_list = current_info.get("username", [])
            else:
                # No local file, use the GCS version
                current_username_list = current_info.get("username", [])

            # Ensure current_username_list is a list
            if not isinstance(current_username_list, list):
                current_username_list = (
                    [current_username_list] if current_username_list else []
                )

            # Ensure username_list is a list
            if not isinstance(username_list, list):
                username_list = [username_list] if username_list else []

            # Create a set for deduplication during comparison but preserve order and duplicates for storage
            current_set = set(current_username_list)

            # Add any new usernames that aren't in the current list
            updated_list = current_username_list.copy()
            for username in username_list:
                if username not in current_set:
                    updated_list.append(username)
                    current_set.add(username)

            # Update the file info with the merged username list
            current_info["username"] = updated_list

            # Upload the updated file info to GCS
            blob.upload_from_string(
                json.dumps(current_info), content_type="application/json"
            )
            logging.info(f"Updated username list for file_id {file_id}: {updated_list}")

            # Also update the local copy of file_info.json
            local_dir = f"./chroma_db/{file_id}"
            if os.path.exists(local_path):
                try:
                    with open(local_path, "w") as f:
                        json.dump(current_info, f, indent=2)
                    logging.info(f"Updated local username list in {local_path}")
                except Exception as e:
                    logging.error(f"Error updating local username list: {str(e)}")
            else:
                try:
                    os.makedirs(local_dir, exist_ok=True)
                    with open(local_path, "w") as f:
                        json.dump(current_info, f, indent=2)
                    logging.info(
                        f"Created local file_info.json with updated username list at {local_path}"
                    )
                except Exception as e:
                    logging.error(
                        f"Error creating local file_info.json with username list: {str(e)}"
                    )

    def delete_embeddings(self, file_id: str):
        """
        Deletes all embeddings associated with a file_id from GCS.

        Args:
            file_id (str): The ID of the file whose embeddings should be deleted
        """
        try:
            logging.info(f"Deleting embeddings for file_id: {file_id}")

            # Delete embeddings folder from GCS
            prefix = f"file-embeddings/{file_id}/"
            blobs = self.bucket.list_blobs(prefix=prefix)

            for blob in blobs:
                try:
                    blob.delete()
                    logging.info(f"Deleted blob: {blob.name}")
                except Exception as e:
                    logging.error(f"Error deleting blob {blob.name}: {str(e)}")

            logging.info(f"Successfully deleted all embeddings for file_id: {file_id}")

        except Exception as e:
            logging.error(f"Error in delete_embeddings: {str(e)}", exc_info=True)
            raise

    # Removed delete_google_embeddings method as part of unified Azure embeddings approach

    def find_file_by_original_name(self, filename: str) -> Optional[str]:
        """
        Search through all folders in file-embeddings to find a file_id by its original filename.

        Args:
            filename (str): The original filename to search for

        Returns:
            Optional[str]: The file_id if found, None otherwise
        """
        prefix = "file-embeddings/"
        for blob in self.bucket.list_blobs(prefix=prefix):
            if blob.name.endswith("file_info.json"):
                try:
                    content = blob.download_as_string()
                    file_info = json.loads(content)
                    if file_info.get("original_filename") == filename:
                        # Extract file_id from path (format: file-embeddings/file_id/file_info.json)
                        return blob.name.split("/")[1]
                except (json.JSONDecodeError, IndexError) as e:
                    logging.error(f"Error processing {blob.name}: {str(e)}")
                    continue
        return None

    def check_embeddings_status(self, file_id: str) -> str:
        """
        Check the status of embeddings for a given file ID.

        This method checks if there's a file_info.json for the given file ID and
        returns the embeddings_status field. If the file doesn't exist or doesn't
        have an embeddings_status field, it returns "unknown".

        Args:
            file_id (str): The ID of the file to check

        Returns:
            str: The status of the embeddings ("completed", "in_progress", "failed", or "unknown")
        """
        try:
            # Check if there's a temporary metadata with this file_id
            if self.temp_metadata and self.temp_metadata.get("file_id") == file_id:
                # Return the status from temp_metadata if available
                return self.temp_metadata.get("embeddings_status", "in_progress")

            # Check if there's a file_info.json for this file_id
            file_info = self.get_file_info(file_id)
            if file_info:
                # Return the status from file_info if available
                return file_info.get("embeddings_status", "in_progress")

            # Check if there are any blobs in the file-embeddings directory
            prefix = f"file-embeddings/{file_id}/"
            blobs = list(self.bucket.list_blobs(prefix=prefix, max_results=1))
            if blobs:
                # If there are blobs but no file_info.json, assume embeddings are in progress
                return "in_progress"

            # If no blobs are found, return unknown
            return "unknown"

        except Exception as e:
            logging.error(f"Error checking embeddings status for {file_id}: {str(e)}")
            return "unknown"
