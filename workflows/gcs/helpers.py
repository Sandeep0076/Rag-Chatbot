import logging

from google.cloud import storage

from workflows.app_config import config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s/%(funcName)s - %(message)s"
)
log = logging.getLogger(__name__)

storage_client = storage.Client(project=config.gcp.gcp_project)
bucket = storage_client.bucket(config.gcp.bucket_name)


def file_present_in_gcp(bucket_prefix: str) -> bool:
    """
    Check if a given file exists in a GCS bucket

    Args:
        bucket_prefix: Path to the file

    Returns:
        Boolean value to indicate if file exists or not
    """
    blobs = bucket.list_blobs(prefix=bucket_prefix)

    return True if len(list(blobs)) > 0 else False


def delete_embeddings(file_id: str):
    """
    Deletes all embeddings associated with a file_id from GCS.

    Args:
        file_id (str): The ID of the file whose embeddings should be deleted
    """
    try:
        log.info(f"Deleting embeddings for file_id: {file_id}")

        # Delete embeddings folder from GCS
        prefix = f"{config.gcp.embeddings_root_folder}/{file_id}/"
        blobs = bucket.list_blobs(prefix=prefix)

        for blob in blobs:
            try:
                blob.delete()
                log.info(f"Deleted blob: {blob.name}")
            except Exception as e:
                log.error(f"Error deleting blob {blob.name}: {str(e)}")

        log.info(f"Successfully deleted all embeddings for file_id: {file_id}")

    except Exception as e:
        log.error(f"Error in delete_embeddings: {str(e)}", exc_info=True)
        raise
