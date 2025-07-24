import uuid

from rtl_rag_chatbot_api.app import get_db_session
from rtl_rag_chatbot_api.common.db import (
    check_file_hash_exists,
    insert_file_info_record,
)

# for checking file hash
with get_db_session() as db:
    result = check_file_hash_exists(db, "c6643ee1c09daa74959312f90495cf0a")

    if result["status"] == "success":
        if result["exists"]:
            print(f"File found: {result['data']['file_id']}")
        else:
            print("File not found in database")
    else:
        print(f"Error: {result['message']}")

# For inserting file hash in db
with get_db_session() as db:
    file_id = str(uuid.uuid4())  # Generate a unique file ID
    file_hash = "test_hash_24_july"

    insert_result = insert_file_info_record(db, file_id, file_hash)
    print(f"Insert result: {insert_result}")
