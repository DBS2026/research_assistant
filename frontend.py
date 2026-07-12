from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# -----------------------------
# Config
# -----------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
POLL_SECONDS = 3

st.set_page_config(page_title="AI Research Intelligence Assistant", layout="wide")

for key, default in {
    "access_token": None,
    "refresh_token": None,
    "user_email": None,
    "selected_document_id": None,
}.items():
    st.session_state.setdefault(key, default)


# -----------------------------
# API client helpers
# -----------------------------
def _auth_headers() -> Dict[str, str]:
    if not st.session_state["access_token"]:
        return {}
    return {"Authorization": f"Bearer {st.session_state['access_token']}"}


def _refresh_access_token() -> bool:
    if not st.session_state["refresh_token"]:
        return False
    try:
        r = requests.post(
            f"{API_BASE_URL}/auth/refresh",
            json={"refresh_token": st.session_state["refresh_token"]},
            timeout=15,
        )
    except requests.RequestException:
        return False
    if r.status_code == 200:
        data = r.json()
        st.session_state["access_token"] = data["access_token"]
        st.session_state["refresh_token"] = data["refresh_token"]
        return True
    return False


def api_request(method: str, path: str, auth: bool = True, **kwargs) -> Optional[requests.Response]:
    """Wraps requests.request, injects the bearer token, and retries once on 401."""
    url = f"{API_BASE_URL}{path}"
    headers = kwargs.pop("headers", {}) or {}
    if auth:
        headers.update(_auth_headers())

    try:
        resp = requests.request(method, url, headers=headers, timeout=60, **kwargs)
    except requests.RequestException as e:
        st.error(f"Could not reach the API at {API_BASE_URL}: {e}")
        return None

    if resp.status_code == 401 and auth and st.session_state["refresh_token"]:
        if _refresh_access_token():
            headers.update(_auth_headers())
            try:
                resp = requests.request(method, url, headers=headers, timeout=60, **kwargs)
            except requests.RequestException as e:
                st.error(f"Could not reach the API at {API_BASE_URL}: {e}")
                return None
        else:
            logout()
            st.warning("Your session expired. Please log in again.")
            st.rerun()

    return resp


def logout() -> None:
    st.session_state["access_token"] = None
    st.session_state["refresh_token"] = None
    st.session_state["user_email"] = None
    st.session_state["selected_document_id"] = None


# -----------------------------
# Auth screens
# -----------------------------
def render_login_register() -> None:
    st.title("AI Research Intelligence Assistant")
    st.caption(f"Connected to API: {API_BASE_URL}")

    login_tab, register_tab = st.tabs(["Log in", "Register"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
        if submitted:
            resp = api_request(
                "POST", "/auth/login", auth=False,
                json={"email": email, "password": password},
            )
            if resp is not None:
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state["access_token"] = data["access_token"]
                    st.session_state["refresh_token"] = data["refresh_token"]
                    st.session_state["user_email"] = email
                    st.rerun()
                else:
                    st.error(resp.json().get("detail", "Login failed."))

    with register_tab:
        with st.form("register_form"):
            email = st.text_input("Email", key="register_email")
            password = st.text_input("Password (min 8 characters)", type="password", key="register_password")
            submitted = st.form_submit_button("Create account", use_container_width=True)
        if submitted:
            resp = api_request(
                "POST", "/auth/register", auth=False,
                json={"email": email, "password": password},
            )
            if resp is not None:
                if resp.status_code == 201:
                    st.success("Account created — you can log in now.")
                else:
                    st.error(resp.json().get("detail", "Registration failed."))


# -----------------------------
# Document list / upload (sidebar)
# -----------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.write(f"Logged in as **{st.session_state['user_email']}**")
        if st.button("Log out", use_container_width=True):
            logout()
            st.rerun()

        st.divider()
        st.header("Upload PDF")
        uploaded_pdf = st.file_uploader("Choose File", type=["pdf"])

        detail_level = st.radio(
            "Detail Level",
            options=["abstract", "student", "detailed", "researcher"],
            format_func=lambda x: x.capitalize(), index=1,
        )
        num_related_papers = st.slider("Number of Related Papers", 3, 20, 5)
        research_depth = st.selectbox(
            "Future Research Depth", options=["low", "medium", "high"],
            index=1, format_func=lambda x: x.capitalize(),
        )

        st.subheader("Study Materials")
        source_books = st.checkbox("Books", value=True)
        source_blogs = st.checkbox("Blogs", value=True)
        source_papers = st.checkbox("Research Papers", value=True)
        source_github = st.checkbox("GitHub", value=True)

        output_style = st.radio(
            "Output Style", options=["blog", "markdown", "json"],
            format_func=lambda x: {"blog": "Blog", "markdown": "Markdown", "json": "JSON"}[x],
        )

        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            if uploaded_pdf is None:
                st.warning("Please upload a PDF.")
            else:
                study_sources = ",".join(
                    k for k, v in [
                        ("books", source_books), ("blogs", source_blogs),
                        ("papers", source_papers), ("github", source_github),
                    ] if v
                )
                resp = api_request(
                    "POST", "/documents/upload",
                    files={"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")},
                    data={
                        "detail_level": detail_level,
                        "num_related_papers": str(num_related_papers),
                        "research_depth": research_depth,
                        "output_style": output_style,
                        "study_sources": study_sources,
                    },
                )
                if resp is not None:
                    if resp.status_code == 202:
                        doc = resp.json()
                        st.success(f"Queued '{doc['filename']}' for analysis.")
                        st.session_state["selected_document_id"] = doc["id"]
                        st.rerun()
                    else:
                        st.error(resp.json().get("detail", "Upload failed."))

        st.divider()
        st.header("Your Documents")
        resp = api_request("GET", "/documents")
        docs: List[Dict[str, Any]] = resp.json() if resp is not None and resp.status_code == 200 else []
        if not docs:
            st.caption("No documents yet — upload a PDF to get started.")
        status_icons = {"pending": "⏳", "processing": "⚙️", "completed": "✅", "failed": "❌"}
        for doc in docs:
            label = f"{status_icons.get(doc['status'], '•')} {doc.get('title') or doc['filename']}"
            if st.button(label, key=f"doc_{doc['id']}", use_container_width=True):
                st.session_state["selected_document_id"] = doc["id"]
                st.rerun()


# -----------------------------
# Main document view
# -----------------------------
def render_document(document_id: int) -> None:
    resp = api_request("GET", f"/documents/{document_id}/status")
    if resp is None or resp.status_code != 200:
        st.error("Could not load this document.")
        return
    doc = resp.json()

    st.header(doc.get("title") or doc["filename"])
    st.caption(f"Status: {doc['status']}  ·  Output style: {doc['output_style']}")

    if doc["status"] in ("pending", "processing"):
        st.info("Analysis in progress — this page will refresh automatically.")
        if doc["status"] == "processing":
            st.progress(0.5, text="Running the multi-agent analysis pipeline…")
        else:
            st.progress(0.1, text="Waiting for a worker to pick this up…")
        time.sleep(POLL_SECONDS)
        st.rerun()
        return

    if doc["status"] == "failed":
        st.error(f"Analysis failed: {doc.get('error_message') or 'Unknown error.'}")
        return

    # completed
    report_resp = api_request("GET", f"/documents/{document_id}/report")
    if report_resp is None or report_resp.status_code != 200:
        st.error("Could not load the report.")
        return
    report = report_resp.json()
    final_md = report.get("final_report_md") or ""

    if doc["output_style"] == "json":
        st.subheader("JSON Output")
        try:
            st.json(json.loads(final_md))
        except json.JSONDecodeError:
            st.code(final_md)
    else:
        st.markdown(final_md)

    st.divider()
    render_chat(document_id, ready=True)

    st.divider()
    file_ext = "json" if doc["output_style"] == "json" else "md"
    st.download_button(
        f"⬇️ Download Report (.{file_ext})",
        data=final_md.encode("utf-8"),
        file_name=f"document_{document_id}.{file_ext}",
        mime="application/json" if file_ext == "json" else "text/markdown",
    )


def render_chat(document_id: int, ready: bool) -> None:
    st.subheader("💬 Ask Follow-up Questions")
    if not ready:
        st.caption("Chat becomes available once analysis completes.")
        return

    history_resp = api_request("GET", f"/documents/{document_id}/chat")
    history = history_resp.json() if history_resp is not None and history_resp.status_code == 200 else []
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_query := st.chat_input("Ask about specific equations, choices, or comparisons..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document references..."):
                resp = api_request(
                    "POST", f"/documents/{document_id}/chat",
                    json={"query": user_query},
                )
                if resp is not None and resp.status_code == 200:
                    st.markdown(resp.json()["content"])
                else:
                    detail = resp.json().get("detail", "Request failed.") if resp is not None else "Request failed."
                    st.error(detail)
        st.rerun()


# -----------------------------
# App entry point
# -----------------------------
if not st.session_state["access_token"]:
    render_login_register()
else:
    render_sidebar()
    if st.session_state["selected_document_id"] is None:
        st.title("AI Research Intelligence Assistant")
        st.info("Upload a PDF from the sidebar, or select a document to view its report.")
    else:
        render_document(st.session_state["selected_document_id"])
