import logging
import os

import google.auth
import google.oauth2.credentials
from google.cloud import storage


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
    ):
        """
        Check if a specific folder exists in the given path and download its contents if it does.

        Parameters:
        bucket_name (str): Name of the GCS bucket
        folder_path (str): Path to the parent folder in the bucket
        folder_name (str): Name of the specific folder to check for
        destination_path (str): Local path to save downloaded files

        Returns:
        bool: True if the folder was found and files were downloaded, False otherwise
        """
        bucket = self._storage_client.bucket(bucket_name)
        prefix = f"{folder_path}/{folder_name}/"
        blobs = list(bucket.list_blobs(prefix=prefix))

        if not blobs:
            logging.info(f"Folder {folder_name} not found in {folder_path}")
            return False

        for blob in blobs:
            if blob.name.endswith("/"):  # Skip directory markers
                continue
            file_name = blob.name.split("/")[-1]
            local_file_path = os.path.join(destination_path, file_name)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            blob.download_to_filename(local_file_path)
            logging.info(f"Downloaded {blob.name} to {local_file_path}")

        return True

    def download_files_from_folder_by_id(self, file_id):
        """
        Download all files from a specific folder in GCS based on the folder ID.
        If no folder is found, log a message and raise an exception.

        Parameters:
        file_id (str): The ID of the folder to download files from.

        Raises:
        FileNotFoundError: If no embeddings are found for the given file_id.
        """
        prefix = f"pdf-embeddings/{file_id}/"
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
