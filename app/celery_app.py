from __future__ import annotations

from celery import Celery

from app.config import settings

celery_app = Celery(
    "research_assistant",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # PDF parsing + several sequential/parallel LLM calls can legitimately
    # take minutes — ack late + a soft/hard limit prevents a worker crash
    # from silently swallowing a job while also preventing runaway tasks.
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=600,
    task_time_limit=660,
)
