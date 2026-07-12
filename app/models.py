from __future__ import annotations

import enum

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class DocumentStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    filename = Column(String(500), nullable=False)
    storage_path = Column(String(1000), nullable=False)

    # Analysis parameters, mirrored from the Streamlit sidebar options
    detail_level = Column(String(50), default="student", nullable=False)
    num_related_papers = Column(Integer, default=5, nullable=False)
    research_depth = Column(String(50), default="medium", nullable=False)
    output_style = Column(String(50), default="markdown", nullable=False)
    study_sources = Column(String(200), default="books,blogs,papers,github", nullable=False)

    status = Column(Enum(DocumentStatus), default=DocumentStatus.pending, nullable=False, index=True)
    celery_task_id = Column(String(255), nullable=True)
    error_message = Column(Text, nullable=True)

    # Populated once the LangGraph pipeline finishes
    final_report_md = Column(Text, nullable=True)
    raw_text = Column(Text, nullable=True)
    title = Column(String(500), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    owner = relationship("User", back_populates="documents")
    chat_messages = relationship(
        "ChatMessage", back_populates="document", cascade="all, delete-orphan", order_by="ChatMessage.created_at"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)

    role = Column(String(20), nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("Document", back_populates="chat_messages")
