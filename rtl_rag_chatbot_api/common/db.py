from datetime import datetime

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
