from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import declarative_base, mapped_column, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "User"

    id = Column(String, primary_key=True, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    wf_deletion_candidate = Column(
        Boolean, nullable=False, default=False, server_default="0"
    )
    wf_deletion_timestamp = Column(DateTime, nullable=True)

    # Foreign keys:

    # Relationships:
    conversations = relationship("Conversation", back_populates="user")
    folders = relationship("Folder", back_populates="user")


class Message(Base):
    __tablename__ = "Message"

    id = Column(String, primary_key=True, unique=True, nullable=False)
    role = Column(String, nullable=False)
    content = Column(String, nullable=False)
    createdAt = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    conversationId = Column(String, nullable=False)

    # Foreign keys:
    conversationId = mapped_column(ForeignKey("Conversation.id"))

    # Relationships:
    conversation = relationship("Conversation", back_populates="messages")


class Conversation(Base):
    __tablename__ = "Conversation"

    id = Column(String, primary_key=True, unique=True, nullable=False)
    temperature = Column(Integer, nullable=False)
    folderId = Column(String, nullable=True)
    prompt = Column(String, nullable=False)
    modelId = Column(String, nullable=False)
    name = Column(String, nullable=False)
    userEmail = Column(String, nullable=False)
    updatedAt = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )
    createdAt = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )
    fileId = Column(String, nullable=True)

    # Foreign keys:
    userEmail = mapped_column(ForeignKey("User.email"))
    folderId = mapped_column(ForeignKey("Folder.id"))
    modelId = mapped_column(ForeignKey("Model.id"))

    # Relationships:
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")
    folder = relationship("Folder", back_populates="conversations")
    model = relationship("Model", back_populates="conversations")


class Model(Base):
    __tablename__ = "Model"

    id = Column(String, primary_key=True, unique=True, nullable=False)
    name = Column(String, nullable=False)
    maxLength = Column(Integer, nullable=False)
    tokenLimit = Column(Integer, nullable=False)

    # Foreign keys:

    # Relationships:
    conversations = relationship("Conversation", back_populates="model")


class Folder(Base):
    __tablename__ = "Folder"

    id = Column(String, primary_key=True, unique=True, nullable=False)
    userId = Column(String, nullable=False)
    name = Column(String, nullable=False)
    isRoot = Column(Boolean, nullable=False, default=False)

    # Foreign keys:
    userId = mapped_column(ForeignKey("User.id"))

    # Relationships:
    user = relationship("User", back_populates="folders")
    conversations = relationship("Conversation", back_populates="folder")
