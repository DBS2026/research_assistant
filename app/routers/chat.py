from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy.orm import Session

from app.backend import _get_clean_content, get_llm_client
from app.cache import get_cached_chat_answer, set_cached_chat_answer
from app.database import get_db
from app.deps import get_current_user
from app.models import ChatMessage, Document, DocumentStatus, User
from app.routers.documents import _get_owned_document
from app.schemas import ChatMessageOut, ChatRequest

router = APIRouter(prefix="/documents", tags=["chat"])


@router.post("/{document_id}/chat", response_model=ChatMessageOut)
def ask_followup_question(
    document_id: int,
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = _get_owned_document(document_id, db, current_user)
    if doc.status != DocumentStatus.completed:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Document is currently '{doc.status.value}', not ready for chat yet")

    db.add(ChatMessage(document_id=doc.id, role="user", content=payload.query))
    db.commit()

    cached = get_cached_chat_answer(document_id, payload.query)
    if cached is not None:
        answer = cached
    else:
        try:
            chat_model = get_llm_client()
            response = chat_model.invoke([
                SystemMessage(content="You are a research partner. Answer the user's question explicitly by referencing the text of this academic paper context layout payload."),
                HumanMessage(content=f"Context Document Data:\n{doc.raw_text}\n\nUser Question: {payload.query}"),
            ])
            answer = _get_clean_content(response)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Failed to compile response: {e}") from e
        set_cached_chat_answer(document_id, payload.query, answer)

    assistant_msg = ChatMessage(document_id=doc.id, role="assistant", content=answer)
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)
    return assistant_msg


@router.get("/{document_id}/chat", response_model=List[ChatMessageOut])
def get_chat_history(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = _get_owned_document(document_id, db, current_user)
    return doc.chat_messages
