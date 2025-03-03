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
