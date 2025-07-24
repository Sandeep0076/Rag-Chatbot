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


def find_file_by_hash_db(session: Session, file_hash: str) -> Optional[str]:
    """
    Find file_id by file_hash from database.

    Args:
        session: Database session
        file_hash: The file hash to search for

    Returns:
        file_id if found, None otherwise
    """
    try:
        result = check_file_hash_exists(session, file_hash)
        if result["status"] == "success" and result["exists"]:
            return result["data"]["file_id"]
        return None
    except Exception as e:
        logging.error(f"Error finding file by hash in database: {e}")
        return None


def insert_file_info_record(
    session: Session, file_id: str, file_hash: str
) -> Dict[str, Any]:
    """
    Insert a new record into the FileInfo table.

    Args:
        session: Database session
        file_id: The file ID
        file_hash: The file hash

    Returns:
        Dict containing the result of the operation
    """
    try:
        # Generate a unique ID for the record
        record_id = str(uuid.uuid4())
        logging.warning(f"ENTERED insert_file_info_record for file_id={file_id}")

        # Create new FileInfo record
        new_file_info = FileInfo(
            id=record_id,
            file_id=file_id,
            file_hash=file_hash,
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
        # Query for existing file hash
        existing_record = (
            session.query(FileInfo).filter(FileInfo.file_hash == file_hash).first()
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
        # Query for existing records with the file_id
        existing_records = (
            session.query(FileInfo).filter(FileInfo.file_id == file_id).all()
        )

        if not existing_records:
            logging.info(f"No FileInfo records found for file_id: {file_id}")
            return {
                "status": "success",
                "deleted": False,
                "message": "No records found for the specified file_id",
                "deleted_count": 0,
            }

        # Store information about records to be deleted
        deleted_records_info = []
        for record in existing_records:
            deleted_records_info.append(
                {
                    "id": record.id,
                    "file_id": record.file_id,
                    "file_hash": record.file_hash,
                    "createdAt": record.createdAt.isoformat(),
                }
            )

        # Delete all records with the specified file_id
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
            "deleted_records": deleted_records_info,
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

        # Get all records for logging purposes (optional - can be removed for performance)
        all_records = session.query(FileInfo).all()
        deleted_records_info = []
        for record in all_records:
            deleted_records_info.append(
                {
                    "id": record.id,
                    "file_id": record.file_id,
                    "file_hash": record.file_hash,
                    "createdAt": record.createdAt.isoformat(),
                }
            )

        # Delete all records from FileInfo table
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
            "deleted_records": deleted_records_info,
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
