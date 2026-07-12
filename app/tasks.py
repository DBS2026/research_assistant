from __future__ import annotations

from datetime import datetime, timezone

from app.celery_app import celery_app
from app.database import SessionLocal
from app.models import Document, DocumentStatus

# Reuses the existing LangGraph pipeline untouched — same nodes, prompts,
# and Pydantic schemas as the original Streamlit app.
from app.backend import app as graph_app


@celery_app.task(bind=True, name="run_document_analysis")
def run_document_analysis(self, document_id: int) -> dict:
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc is None:
            return {"status": "error", "detail": f"Document {document_id} not found"}

        doc.status = DocumentStatus.processing
        doc.celery_task_id = self.request.id
        db.commit()

        inputs = {
            "pdf_path": doc.storage_path,
            "detail_level": doc.detail_level,
            "num_related_papers": doc.num_related_papers,
            "study_sources": doc.study_sources.split(",") if doc.study_sources else [],
            "research_depth": doc.research_depth,
            "output_style": doc.output_style,
            "raw_text": "", "sections": None, "active_agents": [], "summary_md": "", "architecture_md": "",
            "architecture_image_path": None, "technologies": [], "technology_md": "", "methodology_md": "",
            "results_md": "", "limitations_md": "", "publisher_future_work_md": "", "open_problems_md": "",
            "related_papers": [], "comparison_md": "", "research_gap_md": "", "ai_suggested_research_md": "",
            "recent_advances": [], "recent_advances_md": "", "domain_keywords": [],
            "study_materials": {}, "learning_roadmap_md": "", "merged_md": "", "final_report_md": "",
        }

        # thread_id is scoped to the document's DB id, so each document keeps
        # fully isolated LangGraph checkpoint state (the original Streamlit
        # app used one shared, hardcoded thread_id for every user/session).
        config = {"configurable": {"thread_id": str(document_id)}}

        final_state = None
        for _ in graph_app.stream(inputs, config, stream_mode="updates"):
            pass
        final_state = graph_app.get_state(config).values

        sections = final_state.get("sections")
        title = getattr(sections, "title", None) if sections is not None else None

        doc.title = title or doc.filename
        doc.raw_text = final_state.get("raw_text", "")
        doc.final_report_md = final_state.get("final_report_md", "")
        doc.status = DocumentStatus.completed
        doc.completed_at = datetime.now(timezone.utc)
        db.commit()
        return {"status": "completed", "document_id": document_id}

    except Exception as e:
        db.rollback()
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc is not None:
            doc.status = DocumentStatus.failed
            doc.error_message = str(e)[:2000]
            db.commit()
        raise
    finally:
        db.close()
