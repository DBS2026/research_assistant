from __future__ import annotations

import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.deps import get_current_user
from app.models import Document, DocumentStatus, User
from app.schemas import DocumentOut, DocumentReportOut
from app.tasks import run_document_analysis

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
def upload_document(
    file: UploadFile = File(...),
    detail_level: str = Form("student"),
    num_related_papers: int = Form(5),
    research_depth: str = Form("medium"),
    output_style: str = Form("markdown"),
    study_sources: str = Form("books,blogs,papers,github"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only PDF files are accepted")

    contents = file.file.read()
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.MAX_UPLOAD_MB}MB limit",
        )
    if len(contents) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")

    stored_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    dest_path = UPLOAD_DIR / stored_name
    dest_path.write_bytes(contents)

    doc = Document(
        user_id=current_user.id,
        filename=file.filename,
        storage_path=str(dest_path),
        detail_level=detail_level,
        num_related_papers=num_related_papers,
        research_depth=research_depth,
        output_style=output_style,
        study_sources=study_sources,
        status=DocumentStatus.pending,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Hand off the heavy PDF-parsing + multi-agent LLM pipeline to a Celery
    # worker so the upload request returns immediately instead of blocking.
    async_result = run_document_analysis.delay(doc.id)
    doc.celery_task_id = async_result.id
    db.commit()
    db.refresh(doc)

    return doc


@router.get("", response_model=List[DocumentOut])
def list_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return (
        db.query(Document)
        .filter(Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
        .all()
    )


def _get_owned_document(document_id: int, db: Session, current_user: User) -> Document:
    doc = db.query(Document).filter(Document.id == document_id, Document.user_id == current_user.id).first()
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc


@router.get("/{document_id}/status", response_model=DocumentOut)
def get_document_status(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return _get_owned_document(document_id, db, current_user)


@router.get("/{document_id}/report", response_model=DocumentReportOut)
def get_document_report(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = _get_owned_document(document_id, db, current_user)
    if doc.status != DocumentStatus.completed:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Document is currently '{doc.status.value}', not ready yet")
    return doc
