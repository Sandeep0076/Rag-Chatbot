import json
import os
from typing import Dict, List

from google.cloud import storage
from sqlalchemy import Column, DateTime, ForeignKey, String, create_engine
from sqlalchemy.orm import (
    Session,
    declarative_base,
    mapped_column,
    relationship,
    sessionmaker,
)

Base = declarative_base()

DATABASE_USER = os.getenv("DATABASE_USER")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD")
GCP_PROJECT = os.getenv("GCP_PROJECT")
BUCKET_NAME = os.getenv("BUCKET_NAME")


def get_session(database_url: str = None):
    engine = create_engine(database_url)
    LocalSession = sessionmaker(bind=engine)
    return LocalSession()


class User(Base):
    __tablename__ = "User"

    id = Column(String, primary_key=True, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)

    # Relationships:
    conversations = relationship("Conversation", back_populates="user")


class Conversation(Base):
    __tablename__ = "Conversation"

    id = Column(String, primary_key=True, unique=True, nullable=False)
    userEmail = Column(String, nullable=False)
    fileId = Column(String, nullable=True)

    updatedAt = Column(DateTime, nullable=True)

    # Foreign keys:
    userEmail = mapped_column(ForeignKey("User.email"))
    # Relationships:
    user = relationship("User", back_populates="conversations")


def get_user_file_ids(session: Session, users: List) -> Dict[str, List[str]]:
    """
    Returns a dict with the following mapping: username[str](i.e. email):
    List[str](file_ids) for all users in the users list.

    Important note: if the user list is not complete, the username field for a
    particular file ids will not be complete either.
    """

    user_files = (
        session.query(
            Conversation.fileId, Conversation.userEmail, Conversation.updatedAt
        )
        .filter(
            Conversation.userEmail.in_(users),
            Conversation.fileId.isnot(None),
        )
        .order_by(Conversation.updatedAt.desc())
        .all()
    )

    user_file_dict = {}
    file_user_dict = {}
    for user_email, file_id, _ in user_files:
        if user_email not in user_file_dict:
            user_file_dict[user_email] = []
        user_file_dict[user_email].append(file_id)

    for user_email, file_id, _ in user_files:
        if file_id not in file_user_dict:
            file_user_dict[file_id] = []
        file_user_dict[file_id].append(user_email)

    return user_file_dict, file_user_dict


def update_file_info__for_file(bucket, file_id: str, users: List[str]):
    """ """

    print(f"Requested to update file_info for embedding file_id '{file_id}'")
    file_info_path = f"{file_id}/file_info.json"
    blob = bucket.blob(file_info_path)

    if blob.exists():
        file_info = json.loads(blob.download_as_text())
        file_info["username"] = users

        blob.upload_from_string(json.dumps(file_info, indent=2))
        print(
            f"INFO: Set usernames for embedding file_id {file_info_path} as: [{','.join(users)}]"
        )
    else:
        print(f"ERROR: file_info not present for embedding file_id '{file_id}'")


def update_all_file_info(
    bucket: str, user_fileids: Dict[str, List[str]], file_ids_updated: List = []
):
    """
    bucket: The bucket to pass the update to.
    user_fileids: A Dictionary of a list of users mapped to one file id,
        e.g. '23ef12c...' -> [user1, user2, ..]
    file_ids_updated: A list of file ids that were already considered

    Important note: This function just replaces the username field in the file_info.json
    file for each file id in the user_fileids dict.
    It does not gurantee that the username array is complete or correct. For completion,
    the function get_user_file_ids should be called before with the right user list.

    Returns a list of file ids that were updated.
    """

    try:
        for file_id in user_fileids.keys():
            if file_id in file_ids_updated:
                # already handled
                continue

            print(f"Handling file id: {file_id}")
            users = user_fileids[file_id]

            update_file_info__for_file(
                bucket, file_id=f"file-embeddings/{file_id}", users=users
            )

            file_ids_updated.append(file_id)
    finally:
        return file_ids_updated


if __name__ == "__main__":
    storage_client = storage.Client(project=GCP_PROJECT)
    bucket = storage_client.bucket(BUCKET_NAME)

    session = get_session(
        f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@localhost:5432/chatbot_ui"
    )

    # user list is retrieved manually so far
    user_fileids, _ = get_user_file_ids(session, users=["matthaeus.zloch@rtl.de"])
    # updates all file ids and replaces the username field in file_info.json
    update_all_file_info(bucket, user_fileids=user_fileids)

    session.close()
