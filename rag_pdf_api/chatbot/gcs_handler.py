import logging
import os

import google.auth
import google.oauth2.credentials
from google.cloud import storage


class GCSHandler:
    """
    A class to handle interactions with Google Cloud Storage (GCS) for downloading and managing data files.
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
        except:
            self._storage_client = storage.Client(self.configs.gcp_resource.gcp_project)

        self.bucket = self._storage_client.get_bucket(
            self.configs.gcp_resource.bucket_name
        )

    def find_folder_by_id(self, folder_id):
        """
        Find the folder in GCS based on the given ID.

        Parameters:
        folder_id (str): The ID of the folder to find.

        Returns:
        str: The name of the folder with the given ID, or None if no folder is found.
        """
        logging.info(f"Now retrieving the folder with ID: {folder_id}")
        blobs = self.bucket.list_blobs(
            prefix=self.configs.gcp_resource.embeddings_folder
        )

        # Extract folder names from blob names
        folder_names = {blob.name.split("/")[1] for blob in blobs}

        # Find the folder with the given ID
        matching_folder = next((name for name in folder_names if name == folder_id), None)

        if matching_folder:
            logging.info(f"Found folder with ID: {folder_id}")
            return matching_folder
        else:
            logging.info(f"No folder found with ID: {folder_id}")
            return None

    def download_chromadb_files_from_gcs(self, folder_id, local_destination="chroma_db"):
        """
        Download Chroma DB files from the GCS bucket.

        Parameters:
        folder_id (str): The ID of the folder in GCS.
        local_destination (str): The local destination folder to save downloaded files.
        """
        embeddings_folder = self.configs.gcp_resource.embeddings_folder
        blobs = self.bucket.list_blobs(prefix=f"{embeddings_folder}{folder_id}")

        for blob in blobs:
            if (
                blob.name.startswith(embeddings_folder)
                and blob.name != embeddings_folder
            ):
                file_name_on_local = blob.name.split(f"{embeddings_folder}{folder_id}/")[-1]
                local_file_path = os.path.join(local_destination, file_name_on_local)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                blob.download_to_filename(local_file_path)
                logging.info(
                    f"Downloaded gs://{self.bucket}/{blob.name} to {local_file_path} with name {file_name_on_local}"
                )

    def download_files_from_gcs(self, bucket_name: str, source_blob_name: str, destination_file_path: str):
        """
        A generic method to download files from GCS.
        """
        bucket = self._storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        # ensure the directory exists
        os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)

        # and download the blob to a file
        blob.download_to_filename(destination_file_path)

    def download_files_from_folder_by_id(self, folder_id):
        """
        Download files from a specific folder in GCS based on the folder ID.

        Parameters:
        folder_id (str): The ID of the folder to download files from.
        """
        if folder_id:
            logging.info("Now calling download method")
            self.download_chromadb_files_from_gcs(
                folder_id=folder_id,
            )
        else:
            logging.info("No folder found with the given ID.")

    def check_and_download_folder(self, bucket_name: str, folder_path: str, folder_name: str, destination_path: str):
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
            if blob.name.endswith('/'):  # Skip directory markers
                continue
            file_name = blob.name.split('/')[-1]
            local_file_path = os.path.join(destination_path, file_name)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            blob.download_to_filename(local_file_path)
            logging.info(f"Downloaded {blob.name} to {local_file_path}")

        return True