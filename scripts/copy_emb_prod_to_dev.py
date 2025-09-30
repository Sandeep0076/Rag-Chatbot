import json

from google.api_core.exceptions import NotFound
from google.cloud import storage

# --- Configuration ---
# Your source bucket
SOURCE_BUCKET_NAME = "chatbot-storage-prod-gcs-eu"
# Your destination bucket
DESTINATION_BUCKET_NAME = "chatbot-storage-dev-gcs-eu"
# The common directory *within* the bucket where the folders are located
BASE_PREFIX = "file-embeddings/"
# Target user to filter by in file_info.json
TARGET_USERNAME = "sandeep.pathania@rtl.de"
"""
Copy only those folders under BASE_PREFIX from SOURCE_BUCKET_NAME to DESTINATION_BUCKET_NAME
where file_info.json contains TARGET_USERNAME. Preserves the full path/key (server-side copy).
"""

# --- GCS Client Initialization ---
try:
    # IMPORTANT: Ensure your environment is authenticated (e.g., gcloud auth application-default login)
    storage_client = storage.Client()
    source_bucket = storage_client.bucket(SOURCE_BUCKET_NAME)
    destination_bucket = storage_client.bucket(DESTINATION_BUCKET_NAME)
except Exception as e:
    print(f"🛑 Error initializing GCS client or accessing buckets: {e}")
    exit()

# --------------------------------------------------------------------------------

FILE_INFO_JSON = "file_info.json"


def copy_folder_if_user_matches(folder_prefix: str) -> bool:
    """
    Check the file_info.json under the provided folder prefix and copy the entire
    folder's contents to the destination bucket if TARGET_USERNAME is present.

    Returns True if the folder was copied, otherwise False.
    """
    json_blob_path = f"{folder_prefix}{FILE_INFO_JSON}"
    try:
        json_blob = source_bucket.blob(json_blob_path)
        json_content = json_blob.download_as_text()
        file_info = json.loads(json_content)
        usernames = file_info.get("username", [])
        if isinstance(usernames, str):
            usernames = [usernames]

        if TARGET_USERNAME in usernames:
            print(f"✅ Match in {folder_prefix} -> copying...")
            copied = 0
            for blob in source_bucket.list_blobs(prefix=folder_prefix):
                source_bucket.copy_blob(blob, destination_bucket, blob.name)
                copied += 1
                if copied % 100 == 0:
                    print(
                        f"   -> Copied {copied} objects from {folder_prefix} so far..."
                    )
            print(f"   -> Completed copy of {copied} objects from {folder_prefix}")
            return True
        else:
            return False
    except NotFound:
        return False
    except Exception as e:
        print(f"🛑 Error processing {folder_prefix}: {e}")
        return False


# --------------------------------------------------------------------------------

# Main Execution Block

print(
    f"Scanning {SOURCE_BUCKET_NAME}/{BASE_PREFIX} for folders containing '{TARGET_USERNAME}'"
)

folders_seen = 0
folders_copied = 0
iterator = source_bucket.list_blobs(prefix=BASE_PREFIX, delimiter="/")
for page in iterator.pages:
    prefixes = getattr(page, "prefixes", set()) or set()
    for folder_prefix in sorted(prefixes):
        folders_seen += 1
        if copy_folder_if_user_matches(folder_prefix):
            folders_copied += 1

print(
    f"\n--- Script execution complete ---\n"
    f"Folders scanned: {folders_seen}\n"
    f"Folders copied: {folders_copied}\n"
)
