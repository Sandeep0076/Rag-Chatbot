import json
import os

os.environ["gcp"] = json.dumps(
    {
        "gcp_project": os.getenv("GCP_PROJECT"),
        "bucket_name": os.getenv("BUCKET_NAME"),
        "embeddings_root_folder": os.getenv(
            "EMBEDDINGS_ROOT_FOLDER", "embeddings_folder/"
        ),
    }
)

os.environ["workflow"] = json.dumps(
    {
        "db_deleted_user_id": os.getenv("WF_DB_DELETED_USER_ID", "deleted_user_id"),
    }
)
