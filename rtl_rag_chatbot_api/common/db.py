import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Column, DateTime, ForeignKey, String
from sqlalchemy.orm import Session, declarative_base

Base = declarative_base()


class Conversation(Base):
    __tablename__ = "Conversation"

    id = Column(String, primary_key=True, unique=True)
    userEmail = Column(String, ForeignKey("user.email"))
    fileId = Column(String, nullable=True)
    fileName = Column(String, nullable=True)
    updatedAt = Column(DateTime)
    createdAt = Column(DateTime)


class FileInfo(Base):
    __tablename__ = "FileInfo"

    id = Column(String, primary_key=True, unique=True)
    file_id = Column(String, nullable=False)
    file_hash = Column(String, nullable=False)
    file_name = Column(
        String, nullable=True
    )  # Using the correct column name from database
    embedding_type = Column(String, nullable=True, default="azure-3-large")
    createdAt = Column(DateTime, nullable=False)


def get_conversations_by_file_ids(session: Session, file_ids: str) -> datetime:
    """"""

    conversations = (
        session.query(Conversation)
        .filter(
            Conversation.fileId.in_(file_ids),
            Conversation.fileName.isnot(None),
            Conversation.userEmail.isnot(None),
        )
        .all()
    )

    return conversations


def find_file_by_hash_db(session: Session, file_hash: str) -> Optional[tuple[str, str]]:
    """
    Find file_id and embedding_type by file_hash from database.

    Args:
        session: Database session
        file_hash: The file hash to search for

    Returns:
        tuple of (file_id, embedding_type) if found, None otherwise
    """
    try:
        result = check_file_hash_exists(session, file_hash)
        if result["status"] == "success" and result["exists"]:
            return result["data"]["file_id"], result["data"]["embedding_type"]
        return None
    except Exception as e:
        logging.error(f"Error finding file by hash in database: {e}")
        return None


def insert_file_info_record(
    session: Session,
    file_id: str,
    file_hash: str,
    filename: str = None,
    embedding_type: str = None,
    configs=None,
) -> Dict[str, Any]:
    """
    Insert a new record into the FileInfo table.

    Args:
        session: Database session
        file_id: The file ID
        file_hash: The file hash
        filename: The original filename (optional)
        embedding_type: The embedding type to use (if None, uses configs.chatbot.default_embedding_type)
        configs: Configuration object (optional, for getting default embedding type)

    Returns:
        Dict containing the result of the operation
    """
    try:
        # Use configurable default embedding type if not provided
        if embedding_type is None:
            if (
                configs
                and hasattr(configs, "chatbot")
                and hasattr(configs.chatbot, "default_embedding_type")
            ):
                embedding_type = configs.chatbot.default_embedding_type
            else:
                embedding_type = "azure-3-large"  # Fallback default

        # Generate a unique ID for the record
        record_id = str(uuid.uuid4())
        logging.warning(
            f"ENTERED insert_file_info_record for file_id={file_id}, "
            f"filename={filename}, embedding_type={embedding_type}"
        )

        # Create new FileInfo record
        new_file_info = FileInfo(
            id=record_id,
            file_id=file_id,
            file_hash=file_hash,
            file_name=filename,  # Using the correct column name
            embedding_type=embedding_type,
            createdAt=datetime.now(),
        )

        # Add to database
        session.add(new_file_info)
        session.commit()

        logging.info(f"Successfully inserted FileInfo with ID: {record_id}")
        return {
            "status": "success",
            "message": "FileInfo record inserted successfully",
            "data": {
                "id": new_file_info.id,
                "file_id": new_file_info.file_id,
                "file_hash": new_file_info.file_hash,
                "filename": new_file_info.file_name,
                "embedding_type": new_file_info.embedding_type,
                "createdAt": new_file_info.createdAt.isoformat(),
            },
        }
    except Exception as e:
        logging.error(f"Error inserting FileInfo record: {e}")
        session.rollback()
        return {
            "status": "error",
            "message": str(e),
            "details": "Database operation failed",
        }


def check_file_hash_exists(session: Session, file_hash: str) -> Dict[str, Any]:
    """
    Check if a file_hash exists in the FileInfo table.

    Args:
        session: Database session
        file_hash: The file hash to check

    Returns:
        Dict containing the result of the check
    """
    try:
        # Query only the essential fields to avoid column issues
        existing_record = (
            session.query(
                FileInfo.id,
                FileInfo.file_id,
                FileInfo.file_hash,
                FileInfo.file_name,
                FileInfo.embedding_type,
                FileInfo.createdAt,
            )
            .filter(FileInfo.file_hash == file_hash)
            .first()
        )

        if existing_record:
            logging.info(
                f"File hash {file_hash} found in database with file_id: {existing_record.file_id}"
            )

            return {
                "status": "success",
                "exists": True,
                "message": "File hash found in database",
                "data": {
                    "id": existing_record.id,
                    "file_id": existing_record.file_id,
                    "file_hash": existing_record.file_hash,
                    "filename": existing_record.file_name,
                    "embedding_type": existing_record.embedding_type,
                    "createdAt": existing_record.createdAt.isoformat(),
                },
            }
        else:
            logging.info(f"File hash {file_hash} not found in database")
            return {
                "status": "success",
                "exists": False,
                "message": "File hash not found in database",
                "data": None,
            }
    except Exception as e:
        logging.error(f"Error checking file hash existence: {e}")
        return {
            "status": "error",
            "exists": False,
            "message": str(e),
            "details": "Database operation failed",
        }


def delete_file_info_by_file_id(session: Session, file_id: str) -> Dict[str, Any]:
    """
    Delete records from the FileInfo table based on file_id.

    Args:
        session: Database session
        file_id: The file ID to delete

    Returns:
        Dict containing the result of the deletion operation
    """
    try:
        # Check if any records exist with the file_id
        record_count = (
            session.query(FileInfo).filter(FileInfo.file_id == file_id).count()
        )

        if record_count == 0:
            logging.info(f"No FileInfo records found for file_id: {file_id}")
            return {
                "status": "success",
                "deleted": False,
                "message": "No records found for the specified file_id",
                "deleted_count": 0,
            }

        # Delete all records with the specified file_id directly
        deleted_count = (
            session.query(FileInfo).filter(FileInfo.file_id == file_id).delete()
        )
        session.commit()

        logging.info(
            f"Successfully deleted {deleted_count} FileInfo records for file_id: {file_id}"
        )
        return {
            "status": "success",
            "deleted": True,
            "message": f"Successfully deleted {deleted_count} record(s)",
            "deleted_count": deleted_count,
        }
    except Exception as e:
        logging.error(f"Error deleting FileInfo records for file_id {file_id}: {e}")
        session.rollback()
        return {
            "status": "error",
            "deleted": False,
            "message": str(e),
            "details": "Database operation failed",
        }


def delete_all_file_info_records(session: Session) -> Dict[str, Any]:
    """
    Delete all records from the FileInfo table.

    Args:
        session: Database session

    Returns:
        Dict containing the result of the deletion operation
    """
    try:
        # Get count of all records before deletion
        total_records = session.query(FileInfo).count()

        if total_records == 0:
            logging.info("No FileInfo records found in database")
            return {
                "status": "success",
                "deleted": False,
                "message": "No records found in FileInfo table",
                "deleted_count": 0,
            }

        # Delete all records from FileInfo table directly
        deleted_count = session.query(FileInfo).delete()
        session.commit()

        logging.warning(
            f"Successfully deleted ALL {deleted_count} FileInfo records from database"
        )
        return {
            "status": "success",
            "deleted": True,
            "message": f"Successfully deleted all {deleted_count} record(s) from FileInfo table",
            "deleted_count": deleted_count,
        }
    except Exception as e:
        logging.error(f"Error deleting all FileInfo records: {e}")
        session.rollback()
        return {
            "status": "error",
            "deleted": False,
            "message": str(e),
            "details": "Database operation failed",
        }


def export_gcs_file_info_to_sql_text(
    output_file_path: str = "./gcs_file_info_inserts.text",
    bucket_name: str = "chatbot-storage-dev-gcs-eu",
    prefix: str = "file-embeddings/",
    default_embedding_type: str = "azure-3-large",
) -> Dict[str, Any]:
    """
    Export INSERT statements for FileInfo records by scanning file_info.json files in GCS.

    This reads all file_info.json objects under the given prefix in the specified bucket,
    extracts (file_id, file_hash, createdAt, file_name, embedding_type), and writes SQL
    INSERT statements into a .text file for later execution.

    Args:
        output_file_path: Path to write the .text file with SQL INSERT statements
        bucket_name: GCS bucket to scan (default: "chatbot-storage-dev-gcs-eu")
        prefix: GCS prefix to search for file_info.json (default: "file-embeddings/")
        default_embedding_type: Default embedding type for files without one (default: "azure-3-large")

    Returns:
        Dict with summary information: total_scanned, total_written, output_file
    """
    # Lazy imports to avoid hard dependency at module import time
    try:
        import json

        from google.cloud import storage
    except Exception as e:
        logging.error(
            "google-cloud-storage is required to export file info from GCS: %s", e
        )
        return {
            "status": "error",
            "message": "Missing dependency google-cloud-storage",
            "details": str(e),
        }

    try:
        storage_client = storage.Client()
        # Access via client directly; no need to keep a bucket reference
        _ = storage_client.bucket(bucket_name)

        # List all blobs under the prefix
        blobs_iter = storage_client.list_blobs(bucket_name, prefix=prefix)

        insert_lines: list[str] = []
        total_scanned = 0
        total_written = 0

        for blob in blobs_iter:
            # Only process file_info.json files
            if not blob.name.endswith("/file_info.json"):
                continue

            total_scanned += 1
            try:
                content_bytes = blob.download_as_bytes()
                info = (
                    json.loads(content_bytes.decode("utf-8")) if content_bytes else {}
                )
            except Exception as e:
                logging.warning(
                    "Skipping %s due to JSON parse error: %s", blob.name, str(e)
                )
                continue

            # Extract fields with fallbacks
            file_id = info.get("file_id")
            if not file_id:
                # Derive file_id from blob path file-embeddings/{file_id}/file_info.json
                parts = blob.name.split("/")
                if len(parts) >= 3:
                    file_id = parts[1]

            file_hash = info.get("file_hash")
            # Map filename
            file_name_value = info.get("original_filename") or info.get("file_name")
            embedding_type_value = info.get("embedding_type", default_embedding_type)

            # Determine createdAt value
            created_at_str = info.get("embeddings_created_at")
            if not created_at_str:
                if getattr(blob, "time_created", None):
                    created_at_str = blob.time_created.isoformat()
                else:
                    created_at_str = datetime.utcnow().isoformat()

            # We require minimally file_id and file_hash to create an INSERT
            if not file_id or not file_hash:
                logging.info(
                    "Skipping %s because required fields are missing (file_id=%s, file_hash=%s)",
                    blob.name,
                    file_id,
                    file_hash,
                )
                continue

            # Prepare values with SQL escaping
            def _q(val: Optional[str]) -> str:
                if val is None:
                    return "NULL"
                # escape single quotes
                escaped = val.replace("'", "''")
                return f"'{escaped}'"

            record_id = str(uuid.uuid4())
            insert_sql = (
                'INSERT INTO "FileInfo" ("id", "file_id", "file_hash", '
                '"file_name", "embedding_type", "createdAt") '
                f"VALUES ({_q(record_id)}, {_q(file_id)}, {_q(file_hash)}, "
                f"{_q(file_name_value)}, {_q(embedding_type_value)}, {_q(created_at_str)});"
            )

            insert_lines.append(insert_sql)
            total_written += 1

        # Write all statements to the output file
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(insert_lines) + ("\n" if insert_lines else ""))

        logging.info(
            "Wrote %s INSERT statements to %s", total_written, output_file_path
        )
        return {
            "status": "success",
            "total_scanned": total_scanned,
            "total_written": total_written,
            "output_file": output_file_path,
        }

    except Exception as e:
        logging.error(
            "Failed to export GCS file info to SQL text: %s", str(e), exc_info=True
        )
        return {
            "status": "error",
            "message": str(e),
        }


if __name__ == "__main__":
    export_gcs_file_info_to_sql_text(output_file_path="./gcs_file_info_inserts.text")
