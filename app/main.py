from __future__ import annotations

from fastapi import FastAPI

from app.database import Base, engine
from app.routers import auth, chat, documents

# Simple startup-time schema creation. For a project this size, Alembic
# migrations are overkill; if the schema needs to evolve later, that's the
# natural next step to add.
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Research Intelligence Assistant API",
    description="Backend for multi-agent PDF research analysis, built on FastAPI + PostgreSQL + Redis + Celery.",
    version="1.0.0",
)

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok"}
