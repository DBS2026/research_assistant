from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterator

import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from backend import app, _get_clean_content, get_llm_client

# -----------------------------
# Core Configurations & Stream Utilities
# -----------------------------
def safe_slug(title: str) -> str:
    import re
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "report"

def try_stream(graph_app, inputs: Dict[str, Any], config: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    for step in graph_app.stream(inputs, config, stream_mode="updates"):
        yield ("updates", step)
    out = graph_app.get_state(config).values
    yield ("final", out)

def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            current_state.update(next(iter(step_payload.values())))
        else:
            current_state.update(step_payload)
    return current_state

# -----------------------------
# Interface Layout Options
# -----------------------------
st.set_page_config(page_title="AI Research Intelligence Assistant", layout="wide")
st.title("AI Research Intelligence Assistant")

if "last_out" not in st.session_state:
    st.session_state["last_out"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Sidebar control tracking forms
with st.sidebar:
    st.header("Upload PDF")
    uploaded_pdf = st.file_uploader("Choose File", type=["pdf"])

    st.subheader("Detail Level")
    detail_level = st.radio("Detail Level", options=["abstract", "student", "detailed", "researcher"], format_func=lambda x: x.capitalize(), index=1, label_visibility="collapsed")

    st.subheader("Number of Related Papers")
    num_related_papers = st.slider("Number of Related Papers", 3, 20, 5, label_visibility="collapsed")

    st.subheader("Future Research Depth")
    research_depth = st.selectbox("Future Research Depth", options=["low", "medium", "high"], index=1, format_func=lambda x: x.capitalize(), label_visibility="collapsed")

    st.subheader("Study Materials")
    source_books = st.checkbox("Books", value=True)
    source_blogs = st.checkbox("Blogs", value=True)
    source_papers = st.checkbox("Research Papers", value=True)
    source_github = st.checkbox("GitHub", value=True)

    st.subheader("Output Style")
    output_style = st.radio("Output Style", options=["blog", "markdown", "json"], format_func=lambda x: {"blog": "Blog", "markdown": "Markdown", "json": "JSON"}[x], label_visibility="collapsed")

    run_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

# -----------------------------
# Processing Engine Blocks
# -----------------------------
if run_btn:
    if uploaded_pdf is None:
        st.warning("Please upload a PDF.")
        st.stop()

    study_sources = [k for k, v in [("books", source_books), ("blogs", source_blogs), ("papers", source_papers), ("github", source_github)] if v]

    tmp_dir = Path(tempfile.mkdtemp())
    pdf_path = tmp_dir / uploaded_pdf.name
    pdf_path.write_bytes(uploaded_pdf.getvalue())

    inputs = {
        "pdf_path": str(pdf_path), "detail_level": detail_level, "num_related_papers": num_related_papers,
        "study_sources": study_sources, "research_depth": research_depth, "output_style": output_style,
        "raw_text": "", "sections": None, "active_agents": [], "summary_md": "", "architecture_md": "",
        "architecture_image_path": None, "technologies": [], "technology_md": "", "methodology_md": "",
        "results_md": "", "limitations_md": "", "publisher_future_work_md": "", "open_problems_md": "",
        "related_papers": [], "comparison_md": "", "research_gap_md": "", "ai_suggested_research_md": "",
        "recent_advances": [], "recent_advances_md": "", "domain_keywords": [],
        "study_materials": {}, "learning_roadmap_md": "", "merged_md": "", "final_report_md": ""
    }

    status = st.status("Analyzing paper…", expanded=True)
    progress_area = st.empty()
    current_state: Dict[str, Any] = {}

    config = {"configurable": {"thread_id": "current_paper_analysis"}}
    st.session_state["chat_history"] = []

    try:
        for kind, payload in try_stream(app, inputs, config):
            if kind == "updates":
                node_name = next(iter(payload.keys())) if isinstance(payload, dict) else "Processing"
                status.write(f"➡️ Completed Node: {node_name}")
                current_state = extract_latest_state(current_state, payload)
                progress_area.json({
                    "active_agents": current_state.get("active_agents"),
                    "technologies_found": len(current_state.get("technologies") or []),
                    "related_papers_found": len(current_state.get("related_papers") or []),
                    "recent_advances_found": len(current_state.get("recent_advances") or []),
                    "domain_keywords": current_state.get("domain_keywords")
                })
            elif kind == "final":
                st.session_state["last_out"] = payload
                status.update(label="✅ Analysis complete", state="complete", expanded=False)
                progress_area.empty()
    except RuntimeError as e:
        status.update(label="❌ Analysis Failed", state="error", expanded=True)
        if "Free Tier Request Quota" in str(e):
            st.error("⚠️ **Gemini Daily Quota Reached:** You have exhausted the Google AI Studio Free Tier daily limit. Please upgrade your project billing settings to Pay-As-You-Go or try again tomorrow.")
        else:
            st.error(f"Execution Error: {e}")
        st.stop()

# -----------------------------
# Document Layout Display Rendering
# -----------------------------
out = st.session_state.get("last_out")

if not out:
    st.info("Upload a PDF and click **Analyze** to generate the report.")
else:
    sections = out.get("sections")
    title = (sections.title if hasattr(sections, "title") else None) or (sections.get("title") if isinstance(sections, dict) else None) or "Analysis Document"
    st.header(title)

    if out.get("output_style") == "json":
        st.subheader("JSON Output")
        st.json(json.loads(out["final_report_md"]))
    else:
        section_defs = [
            ("Summary", "summary_md"), ("Architecture", "architecture_md"), ("Technology", "technology_md"),
            ("Methodology", "methodology_md"), ("Results", "results_md"), ("Limitations", "limitations_md"),
            ("Publisher Future Work", "publisher_future_work_md"), ("Open Problems", "open_problems_md"),
            ("AI Suggested Research", "ai_suggested_research_md"), ("Recent Advancements", "recent_advances_md"),
            ("Related Papers", None),
            ("Comparison", "comparison_md"), ("Study Materials", None), ("Learning Roadmap", "learning_roadmap_md")
        ]

        for label, key in section_defs:
            if label == "Related Papers":
                papers = out.get("related_papers") or []
                with st.expander(label):
                    if not papers: st.caption("No literature matches parsed.")
                    for p in papers:
                        st.markdown(f"**[{p['title']}]({p['url']})**")
                        st.write(p.get("summary", ""))
                        cols = st.columns(2)
                        cols[0].markdown("Advantages:\n" + "\n".join(f"- {a}" for a in p.get("advantages", [])))
                        cols[1].markdown("Limitations:\n" + "\n".join(f"- {l}" for l in p.get("limitations", [])))
                        st.markdown(f"_Variance Framework:_ {p.get('difference_from_uploaded', '')}")
                        st.divider()
                continue

            if label == "Study Materials":
                materials = out.get("study_materials") or {}
                with st.expander(label):
                    domain_keywords = out.get("domain_keywords") or []
                    if domain_keywords:
                        st.caption(f"Curated around: {', '.join(domain_keywords)}")
                    lbl_map = {"books": "Books", "blogs": "Blogs", "papers": "Research Papers", "github": "GitHub"}
                    found = False
                    for sk, sl in lbl_map.items():
                        items = materials.get(sk, [])
                        if items:
                            found = True
                            st.markdown(f"**{sl}**")
                            for it in items: st.markdown(f"- [{it['title']}]({it['url']}) — _{it['description']}_")
                    if not found: st.caption("No complementary guides loaded.")
                continue

            content = out.get(key, "")
            if content:
                with st.expander(label, expanded=(label == "Summary")):
                    st.markdown(content)

    # -----------------------------
    # Interactive Follow-up Conversational RAG Module
    # -----------------------------
    st.divider()
    st.subheader("💬 Ask Follow-up Questions")

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if user_query := st.chat_input("Ask about specific equations, choices, or comparisons..."):
        with st.chat_message("user"): st.markdown(user_query)
        st.session_state["chat_history"].append({"role": "user", "content": user_query})

        config = {"configurable": {"thread_id": "current_paper_analysis"}}
        historical_raw_text = app.get_state(config).values.get("raw_text", "")

        with st.chat_message("assistant"):
            with st.spinner("Analyzing document references..."):
                try:
                    chat_model = get_llm_client()
                    response = chat_model.invoke([
                        SystemMessage(content="You are a research partner. Answer the user's question explicitly by referencing the text of this academic paper context layout payload."),
                        HumanMessage(content=f"Context Document Data:\n{historical_raw_text}\n\nUser Question: {user_query}")
                    ])
                    answer = _get_clean_content(response)
                    st.markdown(answer)
                    st.session_state["chat_history"].append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Failed to compile response text: {e}")

    st.divider()
    final_md = out.get("final_report_md", "")
    file_ext = "json" if out.get("output_style") == "json" else "md"
    st.download_button(f"⬇️ Download Report (.{file_ext})", data=final_md.encode("utf-8"), file_name=f"{safe_slug(title)}.{file_ext}", mime="application/json" if file_ext == "json" else "text/markdown")