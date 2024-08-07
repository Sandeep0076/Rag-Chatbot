import json
import logging
import os
import shutil
from typing import Dict, List, Optional, Tuple, Union

import google.auth
import google.oauth2.credentials
from google.cloud import storage

from rtl_rag_chatbot_api.common.encryption_utils import decrypt_file


class GCSHandler:
    """
    A class to handle interactions with Google Cloud Storage (GCS) for downloading and managing data files.

    Attributes:
    configs (Config): Configuration object containing GCP resource settings.
    credentials (google.oauth2.credentials.Credentials): Google OAuth credentials.
    _storage_client (storage.Client): Google Cloud Storage client.
    bucket (storage.Bucket): GCS bucket object.
    """

    def __init__(self, _configs):
        """
        Initialize the GCSHandler class.

        Parameters:
        _configs (object): Configuration object containing GCP resource settings.
        """
        self.configs = _configs
        self.credentials = None

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

    def check_and_download_folder(
        self,
        bucket_name: str,
        folder_path: str,
        folder_name: str,
        destination_path: str,
    ) -> Tuple[bool, Optional[List[str]]]:
        """
        Check for the existence of a folder in a Google Cloud Storage bucket,
        download its contents, and decrypt any encrypted files.

        This method checks for the existence of a specified folder within a GCS bucket.
        If the folder exists, it downloads all files within that folder to a local
        destination. Any files with a '.encrypted' extension are automatically decrypted
        after download, and the encrypted versions are deleted locally.

        Args:
            bucket_name (str): The name of the GCS bucket.
            folder_path (str): The path within the bucket where the folder is located.
            folder_name (str): The name of the folder to check and download.
            destination_path (str): The local path where files should be downloaded.

        Returns:
            Tuple[bool, Optional[List[str]]]: A tuple containing:
                - A boolean indicating whether any files were downloaded (True) or not (False).
                - A list of paths to the downloaded (and potentially decrypted) files,
                or None if no files were downloaded.
        """
        bucket = self._storage_client.bucket(bucket_name)
        prefix = f"{folder_path}/{folder_name}/"
        blobs = list(bucket.list_blobs(prefix=prefix))

        if not blobs:
            logging.info(f"Folder {folder_name} not found in {folder_path}")
            return False, None

        downloaded_files = []
        for blob in blobs:
            if blob.name.endswith("/"):  # Skip directory markers
                continue
            file_name = blob.name.split("/")[-1]
            local_file_path = os.path.join(destination_path, file_name)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            blob.download_to_filename(local_file_path)
            logging.info(f"Downloaded {blob.name} to {local_file_path}")

            if file_name.endswith(".encrypted"):
                try:
                    decrypted_file_path = decrypt_file(local_file_path)
                    downloaded_files.append(decrypted_file_path)
                    os.remove(local_file_path)  # Remove the encrypted file
                    logging.info(
                        f"Decrypted {local_file_path} to {decrypted_file_path}"
                    )
                except Exception as e:
                    logging.error(f"Failed to decrypt {local_file_path}: {str(e)}")
                    # If decryption fails, don't add the file to downloaded_files
            else:
                downloaded_files.append(local_file_path)

        return len(downloaded_files) > 0, downloaded_files

    def download_files_from_folder_by_id(self, file_id):
        """
        Download all files from a specific folder in GCS based on the folder ID.
        If no folder is found, log a message and raise an exception.

        Parameters:
        file_id (str): The ID of the folder to download files from.

        Raises:
        FileNotFoundError: If no embeddings are found for the given file_id.
        """
        prefix = f"file-embeddings/{file_id}/"
        blobs = list(self.bucket.list_blobs(prefix=prefix))

        if not blobs:
            error_message = f"No embeddings found for file ID: {file_id}"
            logging.error(error_message)
            raise FileNotFoundError(error_message)

        for blob in blobs:
            if blob.name.endswith("/"):  # Skip directory markers
                continue

            # Construct the local file path
            relative_path = blob.name[len(prefix) :]
            local_file_path = os.path.join("chroma_db", file_id, relative_path)

            # Ensure the directory exists
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the file
            blob.download_to_filename(local_file_path)
            logging.info(f"Downloaded {blob.name} to {local_file_path}")

        logging.info(f"Finished downloading all files for folder ID: {file_id}")

    def cleanup_local_files(self):
        """
        Clean up files inside chroma_db and local_data folders,
        as well as __pycache__ directories.
        """
        folders_to_clean = ["chroma_db", "local_data", "processed_data"]
        for folder in folders_to_clean:
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

    def upload_file_to_gcs(
        self, bucket_name: str, source_file_path: str, destination_blob_name: str
    ):
        """Upload a file to the bucket."""
        bucket = self._storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_path)

        print(f"File {source_file_path} uploaded to {destination_blob_name}.")

    def upload_to_gcs(
        self,
        bucket_name: str,
        source: Union[str, dict, Dict[str, Tuple[Union[str, dict], str]]],
        destination_blob_name: Optional[str] = None,
    ):
        """
        Upload a file, JSON data, or multiple items to the bucket.

        Args:
            bucket_name (str): The name of the GCS bucket.
            source (Union[str, dict, Dict[str, Tuple[Union[str, dict], str]]]):
                - A file path (str)
                - A dictionary to be uploaded as JSON
                - A dictionary of items to upload, where each value is a tuple of (source, destination_blob_name)
            destination_blob_name (Optional[str]): The destination blob name in GCS. Not used for multiple uploads.
        """
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

    def download_and_decrypt_file(self, file_id: str, destination_path: str):
        bucket = self._storage_client.bucket(self.configs.gcp_resource.bucket_name)
        prefix = f"files-raw/{file_id}/"
        blobs = list(bucket.list_blobs(prefix=prefix))

        encrypted_blob = next(
            (blob for blob in blobs if blob.name.endswith(".encrypted")), None
        )

        if not encrypted_blob:
            raise FileNotFoundError(f"No encrypted file found for file_id: {file_id}")

        encrypted_file_path = os.path.join(
            destination_path, os.path.basename(encrypted_blob.name)
        )
        os.makedirs(os.path.dirname(encrypted_file_path), exist_ok=True)

        encrypted_blob.download_to_filename(encrypted_file_path)

        decrypted_file_path = decrypt_file(encrypted_file_path)

        # Clean up the encrypted file
        os.remove(encrypted_file_path)

        return decrypted_file_path

    def find_existing_file(self, filename):
        try:
            logging.info(f"Searching for existing file: {filename}")
            # List all blobs in the 'files-raw' folder
            blobs = self._storage_client.list_blobs(
                self.bucket_name, prefix="files-raw/"
            )

            # Iterate through all blobs
            for blob in blobs:
                # Check if the blob name ends with the filename we're looking for
                if blob.name.endswith(f"/{filename}.encrypted"):
                    # Extract the file_id from the blob name
                    file_id = blob.name.split("/")[-2]
                    logging.info(f"Found existing file: {filename} with ID: {file_id}")
                    return file_id

            logging.info(f"File {filename} not found in existing folders")
            return None
        except Exception as e:
            logging.error(f"Error in find_existing_file: {str(e)}")
            return None
