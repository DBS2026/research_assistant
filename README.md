# AI Research Intelligence Assistant

FastAPI + PostgreSQL + Redis + Celery backend wrapping the existing LangGraph
multi-agent PDF analysis pipeline (`app/backend.py`, unchanged in logic), plus
a Streamlit frontend that talks to that API.

## Architecture

```
Streamlit (frontend.py) → FastAPI (JWT auth) → PostgreSQL (users, documents, chat_history)
                                              → Redis (chat answer cache)
                                              → Celery worker (runs the LangGraph pipeline)
```

The frontend is a thin API client — it holds no LLM/DB code itself. It logs
in against `/auth`, uploads PDFs to `/documents/upload`, polls
`/documents/{id}/status` until the Celery worker finishes, renders
`/documents/{id}/report`, and drives follow-up Q&A through
`/documents/{id}/chat`. This replaces the earlier version of `frontend.py`,
which called the LangGraph pipeline in-process and predates the
auth/DB/Celery split — that version no longer matches the architecture,
since the pipeline now only runs inside the Celery worker.

- **Upload** (`POST /documents/upload`) saves the PDF, creates a `Document`
  row with status `pending`, and enqueues a Celery task. Returns immediately.
- **Worker** runs the full LangGraph graph from `backend.py`, using the
  document's DB id as the LangGraph `thread_id` (so each document's agent
  state is isolated — the original Streamlit app used one shared, hardcoded
  thread_id for every session).
- **Status/report** endpoints let the client poll until `status=completed`.
- **Chat** endpoint answers follow-up questions against the extracted PDF
  text, caching identical questions per document in Redis, and persisting
  the full conversation in Postgres.

## Local development

```bash
cp .env.example .env
# fill in GOOGLE_API_KEY (required) and TAVILY_API_KEY (optional)

docker compose up --build
```

- Frontend (Streamlit): http://localhost:8501
- API + Swagger docs: http://localhost:8000/docs
- Postgres: localhost:5432 (postgres/postgres/research_assistant)
- Redis: localhost:6379

Register a user from the frontend's "Register" tab (or `POST /auth/register`),
then log in and upload a PDF.

## API summary

| Endpoint | Method | Auth | Purpose |
|---|---|---|---|
| `/auth/register` | POST | — | Create a user |
| `/auth/login` | POST | — | Get access + refresh JWT |
| `/auth/refresh` | POST | — | Exchange refresh token for a new access token |
| `/documents/upload` | POST | Bearer | Upload a PDF, kicks off async analysis |
| `/documents` | GET | Bearer | List your documents |
| `/documents/{id}/status` | GET | Bearer | Poll analysis status |
| `/documents/{id}/report` | GET | Bearer | Get the final report (once completed) |
| `/documents/{id}/chat` | POST | Bearer | Ask a follow-up question |
| `/documents/{id}/chat` | GET | Bearer | Get chat history |

All authenticated endpoints expect `Authorization: Bearer <access_token>`.

## Deploying to Render

Render has no native managed MySQL, but **Postgres is first-class** — use
Render's managed Postgres instance directly.

1. **Postgres** — Render dashboard → New → PostgreSQL. Copy the internal
   connection string into `DATABASE_URL`.
2. **Redis** — Render's managed Redis (or Upstash free tier) → copy the URL
   into `REDIS_URL`, `CELERY_BROKER_URL` (use a different DB index, e.g.
   `/1`), and `CELERY_RESULT_BACKEND` (`/2`).
3. **Web service** — New → Web Service → point at this repo, Docker runtime,
   start command left as the Dockerfile default (`uvicorn app.main:app
   --host 0.0.0.0 --port 8000`). Render sets `PORT` for you — if it differs
   from 8000, override the start command to `uvicorn app.main:app --host
   0.0.0.0 --port $PORT`.
4. **Background worker** — New → Background Worker → same repo/image,
   start command: `celery -A app.celery_app worker --loglevel=info`.
5. Set all env vars from `.env.example` on **both** services (web + worker)
   — they both need `DATABASE_URL`, `REDIS_URL`, `GOOGLE_API_KEY`, etc.
6. **Uploads storage**: Render's filesystem is ephemeral on redeploy/restart
   unless you attach a **Render Disk** to both the web and worker service
   mounted at `/code/uploads`. Without it, uploaded PDFs referenced by a
   pending/processing job can disappear if the service restarts mid-job.
7. Render's free-tier web services spin down after inactivity — the first
   request after idle will be slow (cold start).
8. **Frontend** — New → Web Service → same repo, Docker runtime, set
   **Dockerfile Path** to `Dockerfile.frontend`. Set `API_BASE_URL` to the
   API service's public Render URL (e.g. `https://your-api.onrender.com`).

## Notes / intentional simplifications

- Schema is created via `Base.metadata.create_all()` at startup rather than
  Alembic migrations — fine at this scale; add Alembic if the schema needs
  versioned changes later.
- No RBAC — every user can only see their own documents (enforced by
  `user_id` filtering on every query), which is sufficient without a
  multi-role admin panel.
- Chat caching is keyed on `(document_id, normalized query text)`; it's a
  performance optimization only — a Redis outage falls back to calling the
  LLM directly rather than failing the request.
