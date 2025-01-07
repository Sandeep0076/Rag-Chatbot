import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

import google.auth
import google.oauth2.credentials
from google.cloud import storage


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

            # Ensure directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Download file
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

    def upload_to_gcs(
        self,
        bucket_name: str,
        source: Union[str, dict, Dict[str, Union[str, dict, tuple]]],
        destination_blob_name: Optional[str] = None,
    ):
        bucket = self._storage_client.bucket(bucket_name)

        if isinstance(source, dict) and destination_blob_name is None:
            # Multiple upload case
            for _, (item_source, item_destination) in source.items():
                blob = bucket.blob(item_destination)
                if isinstance(item_source, dict):
                    blob.upload_from_string(
                        data=json.dumps(item_source), content_type="application/json"
                    )
                elif isinstance(item_source, str):
                    blob.upload_from_filename(item_source)
                print(f"Uploaded to {item_destination}")
        else:
            # Single upload case
            blob = bucket.blob(destination_blob_name)
            if isinstance(source, dict):
                blob.upload_from_string(
                    data=json.dumps(source), content_type="application/json"
                )
            elif isinstance(source, str):
                blob.upload_from_filename(source)
            print(f"Uploaded to {destination_blob_name}")

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
            current_info.update(new_info)
            blob.upload_from_string(
                json.dumps(current_info), content_type="application/json"
            )
        else:
            blob.upload_from_string(
                json.dumps(new_info), content_type="application/json"
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
