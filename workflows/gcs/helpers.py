import json
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


def delete_embeddings(file_id: str, user_email: str) -> None:
    """
    Deletes all embeddings associated with a file_id associated to the given user_email from GCS.

    Args:
        file_id (str): The ID of the file whose embeddings should be deleted
        user_email (str): The email of the user that is associated with the embedding
    """
    try:
        log.info(
            f"Removing association of {user_email} with embeddings for file_id: {file_id}"
        )

        # download file_info.json
        embeddings_file_folder = f"{config.gcp.embeddings_root_folder}/{file_id}/"
        file_info_blob_name = f"{embeddings_file_folder}/file_info.json"
        file_info_blob = bucket.blob(file_info_blob_name)

        if not file_info_blob.exists():
            log.error(
                f"Expected file_info.json at {file_info_blob_name} but couldn't find :/"
            )
            log.error(
                f"Error removing association of {user_email} with file_id {file_id}. Continuing..."
            )
            return

        file_info_content = file_info_blob.download_as_text()
        file_info = json.loads(file_info_content)

        # check user exists ..
        username_list = file_info.get("username", [])
        if user_email in username_list:
            username_list.remove(user_email)
            log.info(f"Removed {user_email} from file_info.json for file_id: {file_id}")

        # .. and update the username list
        file_info["username"] = username_list

        # if the list is not empty, update the file_info.json back to the bucket
        if username_list:
            updated_content = json.dumps(file_info)
            file_info_blob.upload_from_string(
                updated_content, content_type="application/json"
            )
            log.info(
                f"Updated file_info.json and removed {user_email} from list for file_id: {file_id}"
            )
        else:
            # delete the whole embeddings folder if username list is empty
            try:
                blobs = bucket.list_blobs(prefix=embeddings_file_folder)
                bucket.delete_blobs(list(blobs))
                log.info(
                    f"Successfully deleted whole embeddings folder for file_id: {file_id}"
                )
            except Exception as e:
                log.error(
                    f"Error deleting blobs in folder {embeddings_file_folder}: {str(e)}"
                )
                raise

    except Exception as e:
        log.error(f"Error in delete_embeddings: {str(e)}", exc_info=True)
        raise
