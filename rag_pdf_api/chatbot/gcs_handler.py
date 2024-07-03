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
        if os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN", None) is not None:
            logging.info("Found GOOGLE_OAUTH_ACCESS_TOKEN token, now using it.")
            self.credentials = google.oauth2.credentials.Credentials(
                os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN"), refresh_token=""
            )

        try:
            # Used for end-to-end test pipeline
            self._storage_client = storage.Client(
                self.configs.gcp_resource.gcp_project, credentials=self.credentials
            )
        except:
            # Used for production
            self._storage_client = storage.Client(self.configs.gcp_resource.gcp_project)

        self.bucket = self._storage_client.get_bucket(
            self.configs.gcp_resource.bucket_name
        )
        self.latest_timestamp_folder = self.get_latest_time_stamp_folder()

    def get_latest_time_stamp_folder(self):
        """
        Retrieve the folder with the latest timestamp name in GCS.

        Returns:
        str: The name of the folder with the latest timestamp, or None if no folders are found.
        """
        logging.info("Now retrieving the latest timestamp folder")
        blobs = self.bucket.list_blobs(
            prefix=self.configs.gcp_resource.embeddings_folder
        )

        # Extract timestamps from blob names
        timestamps = [blob.name.split("/")[1] for blob in blobs]

        # Sort timestamps in descending order
        timestamps.sort(reverse=True)

        # Logging information
        logging.info(
            f"There were a total of {len(timestamps)} blobs in the bucket. The latest ts folder is {timestamps[0]}"
        )

        # Return the latest timestamp folder
        if timestamps:
            return timestamps[0]
        else:
            return None

    def download_chromadb_files_from_gcs(
        self, latest_timestamp, local_destination="chroma_db"
    ):
        """
        Download Chroma DB files from the GCS bucket.

        Parameters:
        latest_timestamp (str): The folder name with the latest timestamp in GCS.
        local_destination (str): The local destination folder to save downloaded files.
        """
        embeddings_folder = self.configs.gcp_resource.embeddings_folder
        blobs = self.bucket.list_blobs(prefix=embeddings_folder + latest_timestamp)

        for blob in blobs:
            if (
                blob.name.startswith(embeddings_folder)
                and blob.name != embeddings_folder
            ):
                file_name_on_local = blob.name.split(
                    embeddings_folder + latest_timestamp + "/"
                )[-1]
                local_file_path = os.path.join(local_destination, file_name_on_local)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                blob.download_to_filename(local_file_path)
                logging.info(
                    f"Downloaded gs://{self.bucket}/{blob.name} to {local_file_path} with name {file_name_on_local}"
                )

    def download_files_from_gcs(
        self, bucket_name: str, source_blob_name: str, destination_file_path: str
    ):
        """
        A generic method to download files from GCS.
        """

        # Get the bucket
        bucket = self._storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        # ensure the directory exists
        os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)

        # and download the blob to a file
        blob.download_to_filename(destination_file_path)

    def download_latest_timestamp_files(self):
        """
        Download files from the newest folder in GCS.
        """
        if self.latest_timestamp_folder:
            logging.info("Now calling download method")
            self.download_chromadb_files_from_gcs(
                latest_timestamp=self.latest_timestamp_folder,
            )
        else:
            logging.info("No timestamp folders found.")
