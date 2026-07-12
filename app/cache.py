from __future__ import annotations

import hashlib
import json
from typing import Optional

import redis

from app.config import settings

_redis_client: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_client


def _chat_cache_key(document_id: int, query: str) -> str:
    query_hash = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()
    return f"chat:doc:{document_id}:{query_hash}"


def get_cached_chat_answer(document_id: int, query: str) -> Optional[str]:
    try:
        raw = get_redis().get(_chat_cache_key(document_id, query))
    except redis.RedisError:
        # Cache is a performance optimization, not a correctness dependency —
        # if Redis is briefly unavailable we fall back to calling the LLM.
        return None
    if raw is None:
        return None
    return json.loads(raw)["answer"]


def set_cached_chat_answer(document_id: int, query: str, answer: str) -> None:
    try:
        get_redis().setex(
            _chat_cache_key(document_id, query),
            settings.CHAT_CACHE_TTL_SECONDS,
            json.dumps({"answer": answer}),
        )
    except redis.RedisError:
        pass
