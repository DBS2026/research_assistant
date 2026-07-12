from __future__ import annotations

import base64
import json
import operator
import os
import re
import time
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Dict, Annotated, Tuple

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from dotenv import load_dotenv

# Initialize database-backed persistence cache
set_llm_cache(SQLiteCache(database_path=".langchain_llm_cache.db"))
load_dotenv()

from app.academic_search import search_all_academic_sources

# -----------------------------
# 1) Schemas & Pydantic Elements
# -----------------------------
DetailLevel = Literal["abstract", "student", "detailed", "researcher"]
ResearchDepth = Literal["low", "medium", "high"]
OutputStyle = Literal["blog", "markdown", "json"]

class PaperSections(BaseModel):
    title: str = ""
    abstract: str = ""
    introduction: str = ""
    methodology: str = ""
    results: str = ""
    conclusion: str = ""
    references: str = ""

class Technology(BaseModel):
    name: str
    category: str = Field(..., description="framework, model, dataset, optimizer, hardware component")
    role_in_paper: str = Field(..., description="How this is used in the paper's text. Grounded strictly.")
    description: str = Field(..., description="Brief general-context sentence independent of this paper.")
    why_learn_it: str = Field(..., description="Educational impact statement on why a researcher/engineer must learn this tech.")

class CoreAnalysisPack(BaseModel):
    summary_md: str = Field(..., description="## Summary section matching the detail profile.")
    methodology_md: str = Field(..., description="## Methodology step-by-step documentation workflow.")
    results_md: str = Field(..., description="## Results metrics breakdown.")
    limitations_md: str = Field(..., description="## Limitations acknowledged by authors. [Confidence: ★★★★☆]")
    publisher_future_work_md: str = Field(..., description="## Publisher's Future Work explicit tracking statements. [Confidence: ★★★★★]")
    open_problems_md: str = Field(..., description="## Open Problems inferred conceptually. [Confidence: ★★★☆☆]")
    technologies: List[Technology] = Field(default_factory=list, description="Up to 6 core tech components.")
    domain_keywords: List[str] = Field(default_factory=list, description="2-4 field-level domain terms.")

class RelatedPaper(BaseModel):
    title: str
    url: str
    summary: str
    methodology: str
    key_results: str
    advantages: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    difference_from_uploaded: str
    publication_year: Optional[int] = Field(default=None, description="Year the paper was published, if determinable from the candidate source.")
    venue: Optional[str] = Field(default=None, description="Publication venue/publisher, e.g. IEEE, ACM, Springer, Elsevier, arXiv. Use null if unknown — do not guess.")

class RelatedPapersPack(BaseModel):
    papers: List[RelatedPaper] = Field(default_factory=list)

class RecentAdvance(BaseModel):
    title: str
    url: str
    year: Optional[int] = None
    summary: str
    relevance_to_paper: str

class GapAndAdvancesPack(BaseModel):
    research_trend_md: str
    current_sota_md: str
    identified_gaps: List[str] = Field(default_factory=list)
    ai_suggested_directions: List[str] = Field(default_factory=list)
    recent_advances: List[RecentAdvance] = Field(default_factory=list)

class LearningRoadmapStep(BaseModel):
    topic: str
    reason: str
    order: int
    difficulty: Literal["beginner", "intermediate", "advanced"] = "intermediate"
    estimated_hours: int

class BuildGuideStep(BaseModel):
    stage: str
    paper_grounding: str
    starter_tools: List[str] = Field(default_factory=list)
    prototype_goal: str

class LearningRoadmap(BaseModel):
    steps: List[LearningRoadmapStep] = Field(default_factory=list)
    build_guide: List[BuildGuideStep] = Field(default_factory=list)

# -----------------------------
# 2) State Definition & Reducers
# -----------------------------
def merge_study_materials(left: Optional[Dict[str, List[dict]]], right: Optional[Dict[str, List[dict]]]) -> Dict[str, List[dict]]:
    if left is None: left = {}
    if right is None: right = {}
    merged = {k: list(v) for k, v in left.items()}
    for key, value in right.items():
        if key in merged and isinstance(merged[key], list) and isinstance(value, list):
            merged[key].extend(value)
        else:
            merged[key] = value

    # Dedup by title+URL. This matters beyond a single run: with the
    # MemorySaver checkpointer, re-invoking the same thread_id calls this
    # reducer again with `left` = previously persisted state, so without this
    # step the same resource keeps accumulating across reruns even though
    # study_material_agent already dedupes within one run. Title is checked
    # too (not just URL) to catch mirrors of the same resource hosted at a
    # different URL.
    for key, items in merged.items():
        if isinstance(items, list):
            merged[key] = dedupe_study_items(items)

    return merged

class State(TypedDict):
    pdf_path: str
    detail_level: DetailLevel
    num_related_papers: int
    study_sources: List[str]
    research_depth: ResearchDepth
    output_style: OutputStyle

    raw_text: str
    sections: Optional[PaperSections]
    active_agents: List[str]

    summary_md: str
    methodology_md: str
    results_md: str
    limitations_md: str
    publisher_future_work_md: str
    open_problems_md: str

    architecture_md: str
    architecture_image_path: Optional[str]

    technologies: List[dict]
    technology_md: str

    related_papers: List[dict]
    comparison_md: str
    research_gap_md: str
    ai_suggested_research_md: str

    recent_advances: List[dict]
    recent_advances_md: str

    domain_keywords: List[str]
    study_materials: Annotated[Dict[str, List[dict]], merge_study_materials]
    learning_roadmap_md: str

    merged_md: str
    final_report_md: str

# -----------------------------
# 3) Markdown Processing Pipeline
# -----------------------------
def clean_markdown(text: str) -> str:
    if not text:
        return ""

    # Clean standard and carriage-return newlines
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\\n", "\n")
    text = text.replace("\\t", " ")

    # Eliminate horizontal tab and multi-space clustering
    text = re.sub(r"[ \t]+", " ", text)

    # Fix inline math styling artifacts
    text = re.sub(r'\$\s+([^$]+?)\s+\$', r'$\1$', text)

    # Remove excessive blank lines and stray punctuation spaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([.,:;])", r"\1", text)

    return text.strip()

def dedupe_study_items(items: List[dict]) -> List[dict]:
    """Dedupe study-material items on BOTH normalized title and URL.

    URL-only dedup misses mirrors/re-hosts of the same resource under a
    different URL (e.g. "FPGA Books" vs "FPGA Books (mirror)"). Title-only
    dedup risks collapsing genuinely distinct resources that happen to share
    a generic title. Checking both catches the mirror case while still
    letting distinct URLs with distinct titles through.
    """
    seen_titles = set()
    seen_urls = set()
    deduped = []
    for item in items:
        if not isinstance(item, dict):
            deduped.append(item)
            continue
        title_key = (item.get("title") or "").strip().lower()
        url_key = item.get("url")
        if title_key and title_key in seen_titles:
            continue
        if url_key and url_key in seen_urls:
            continue
        if title_key:
            seen_titles.add(title_key)
        if url_key:
            seen_urls.add(url_key)
        deduped.append(item)
    return deduped

# -----------------------------
# 4) Resilient LLM Providers
# -----------------------------
FORMATTING_RULES = """
Formatting Rules:
- Use clear Markdown headings (e.g., ## and ###).
- Use clean bullet points where appropriate.
- Write mathematical equations using standard LaTeX notations enclosed within dollar signs.
  Example: Inline equations must be formatted like $e^x$ or $\\frac{1}{1+e^{-x}}$. Standalone math blocks must use $$.
- Never split mathematical components or equations across multiple lines. Keep statements unbroken.
- Citation Rules: Whenever you describe an equation, a table, or a figure from the document, append its explicit source if visible (e.g., "[Equation 7]", "[Table 2]").
"""

def get_llm_client(structured_output_cls=None):
    client = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
        # Large structured-output requests (see CoreAnalysisPack) can
        # legitimately take well over a minute to generate. Without an
        # explicit timeout, some environments' default is short enough that
        # the connection gets dropped mid-generation, surfacing as
        # httpx.RemoteProtocolError rather than a clean timeout error.
        timeout=300,
    )
    if structured_output_cls:
        return client.with_structured_output(structured_output_cls)
    return client

def _validate_messages(messages) -> None:
    for m in messages:
        content = getattr(m, "content", None)
        # content can be a non-empty string, or a list of content blocks
        # (e.g. text + image_url dicts) — only reject truly empty/blank content.
        if content is None:
            raise ValueError(f"Empty message detected: {type(m).__name__} has no content")
        if isinstance(content, str) and not content.strip():
            raise ValueError(f"Empty message detected: {type(m).__name__} has blank string content")
        if isinstance(content, list) and not content:
            raise ValueError(f"Empty message detected: {type(m).__name__} has an empty content list")

def invoke_llm_with_retry(messages, structured_output_cls=None):
    import httpx
    import ssl

    _validate_messages(messages)
    model = get_llm_client(structured_output_cls)
    # Transient network faults: the Gemini backend (or an intermediary proxy/
    # VPN/firewall) can silently drop the connection on slow requests (large
    # structured-output payloads regularly exceed ~60s), which surfaces as
    # httpx.RemoteProtocolError / ssl.SSLError rather than an HTTP status
    # code. These are safe to retry as-is; a quota or auth error is not.
    TRANSIENT_EXCEPTIONS = (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout, ssl.SSLError)

    last_err: Optional[Exception] = None
    for attempt in range(5):
        try:
            return model.invoke(messages)
        except TRANSIENT_EXCEPTIONS as e:
            last_err = e
            time.sleep(5 * (attempt + 1))
            continue
        except Exception as e:
            err_str = str(e)
            if "GenerateRequestsPerDayPerProjectPerModel" in err_str or "quota exceeded" in err_str.lower():
                raise RuntimeError("❌ Critical Error: Gemini Free Tier Request Quota exhausted.") from e
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                last_err = e
                time.sleep(10 * (attempt + 1))
                continue
            raise e
    raise RuntimeError(
        f"❌ Gemini request failed after 5 attempts due to a network/connection error: {last_err}"
    ) from last_err

def _get_clean_content(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        extracted = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                extracted.append(item["text"])
            elif isinstance(item, str):
                extracted.append(item)
        return clean_markdown("".join(extracted))
    return clean_markdown(str(content))

# -----------------------------
# 5) Fallback-Enabled Search Mechanics
# -----------------------------
def _fallback_search_chain(query: str, max_results: int = 4) -> List[dict]:
    results = []
    tavily_res = _tavily_search(query, max_results=max_results)
    if tavily_res:
        return tavily_res

    try:
        academic_res = search_all_academic_sources(query, limit_per_source=max_results)
        if academic_res:
            for item in academic_res:
                results.append({
                    "title": item.get("title") or "Academic Document",
                    "url": item.get("url") or "",
                    "snippet": item.get("snippet") or item.get("abstract") or ""
                })
            return results
    except Exception: pass
    return results

def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        tool = TavilySearchResults(max_results=max_results)
        res = tool.invoke({"query": query})
        return [{"title": r.get("title") or "", "url": r.get("url") or "", "snippet": r.get("content") or r.get("snippet") or ""} for r in res or []]
    except Exception:
        return []

# -----------------------------
# 6) Execution Graph Nodes
# -----------------------------
def pdf_extraction_node(state: State) -> dict:
    import fitz
    doc = fitz.open(state["pdf_path"])
    raw_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    sections_map = {"title": "", "abstract": "", "introduction": "", "methodology": "", "results": "", "conclusion": "", "references": ""}
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    if lines:
        sections_map["title"] = lines[0][:200]

    patterns = [
        ("abstract", r"(?i)\b(abstract|summary)\b"),
        ("introduction", r"(?i)\b(introduction|background)\b"),
        ("methodology", r"(?i)\b("
                         r"methodology|methods|"
                         r"proposed\s+approach|"
                         r"proposed\s+method|"
                         r"proposed\s+architecture|"
                         r"proposed\s+framework|"
                         r"system\s+architecture|"
                         r"system\s+design|"
                         r"implementation|"
                         r"experimental\s+setup|"
                         r"hardware\s+design|"
                         r"design\s+method"
                         r")\b"),
        ("results", r"(?i)\b(results|experiments|evaluation|discussion)\b"),
        ("conclusion", r"(?i)\b(conclusion|concluding\s+remarks)\b"),
        ("references", r"(?i)\b(references|bibliography)\b")
    ]

    matches = sorted([(m.start(), key) for key, r in patterns for m in re.finditer(r, raw_text)], key=lambda x: x[0])
    if matches:
        for i, (start_pos, key) in enumerate(matches):
            end_pos = matches[i+1][0] if i + 1 < len(matches) else len(raw_text)
            sections_map[key] = raw_text[start_pos:end_pos].strip()
    else:
        sections_map["abstract"] = raw_text[:3000]
        sections_map["methodology"] = raw_text[3000:40000]
        sections_map["results"] = raw_text[40000:]

    # Safety net: heading-based regex matching can miss sections entirely
    # (e.g. a paper uses "Approach"/"Framework"/"Our Method" instead of the
    # recognized keywords). Any required section still left empty here would
    # later be passed as empty message content to the LLM API, which raises
    # "ValueError: contents are required." Fall back to raw_text slices so
    # every required section is guaranteed non-empty.
    if not sections_map["abstract"].strip():
        sections_map["abstract"] = raw_text[:3000].strip()
    if not sections_map["methodology"].strip():
        sections_map["methodology"] = (raw_text[3000:40000] or raw_text).strip()
    if not sections_map["results"].strip():
        sections_map["results"] = raw_text[40000:].strip() or raw_text[-3000:].strip()

    return {"raw_text": raw_text, "sections": PaperSections(**sections_map)}

def router_node(state: State) -> dict:
    level = state.get("detail_level", "student")
    agents = ["unified_report_agent"]
    if level in ["detailed", "researcher"]:
        agents.extend(["architecture_agent", "related_paper_agent", "learning_roadmap_agent"])
    return {"active_agents": agents}

def route_from_router(state: State) -> List[str]:
    return state["active_agents"]

# -----------------------------
# 7) Core Synthesis Layer
# -----------------------------
# NOTE: FORMATTING_RULES contains raw LaTeX braces (e.g. \frac{1}{1+e^{-x}}).
# MONOLITHIC_SYSTEM_PROMPT is later filled in via a plain str.replace() call
# (NOT str.format()) specifically to avoid Python's .format() trying to parse
# those literal braces as positional replacement fields. Do not switch this
# back to .format() unless every brace in FORMATTING_RULES is escaped as {{ }}.
MONOLITHIC_SYSTEM_PROMPT = """You are an advanced academic research engine. Analyze the provided sections of the academic paper and generate a comprehensive, structured output pack filling all markdown sections.
""" + FORMATTING_RULES + """
Target Audience Profile: {level}

Confidence Indicators:
- Label the headers of inferred sections explicitly with the following strict indicators:
  - Limitations: Include '[Confidence: ★★★★☆]' next to its header
  - Open Problems: Include '[Confidence: ★★★☆☆]' next to its header
"""

def unified_report_agent(state: State) -> dict:
    s = state["sections"]
    payload = f"Title: {s.title}\n\nAbstract:\n{s.abstract}\n\nIntro:\n{s.introduction}\n\nMethodology:\n{s.methodology}\n\nResults:\n{s.results}\n\nConclusion:\n{s.conclusion}"

    # Using .replace() instead of .format() here — see note above MONOLITHIC_SYSTEM_PROMPT.
    system_prompt = MONOLITHIC_SYSTEM_PROMPT.replace("{level}", state["detail_level"])

    pack = invoke_llm_with_retry(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=payload)
        ],
        structured_output_cls=CoreAnalysisPack
    )
    return {
        "summary_md": clean_markdown(pack.summary_md),
        "methodology_md": clean_markdown(pack.methodology_md),
        "results_md": clean_markdown(pack.results_md),
        "limitations_md": clean_markdown(pack.limitations_md),
        "publisher_future_work_md": clean_markdown(pack.publisher_future_work_md),
        "open_problems_md": clean_markdown(pack.open_problems_md),
        "technologies": [t.model_dump() for t in pack.technologies],
        "domain_keywords": pack.domain_keywords,
    }

# -----------------------------
# 8) Architecture Processing
# -----------------------------
def architecture_agent(state: State) -> dict:
    img = _extract_figure(state["pdf_path"])
    s = state["sections"]
    system_prompt = f"Analyze the architecture flow and produce a valid ```mermaid flowchart block. Keep the block encapsulated correctly. {FORMATTING_RULES}"

    # Defensive fallback: never let an empty string reach HumanMessage(content=...).
    # The Gemini SDK raises "ValueError: contents are required." if a message has
    # no content at all, which crashes the whole graph run.
    methodology_text = (s.methodology or "").strip()
    if not methodology_text:
        methodology_text = (s.abstract or "").strip() or state.get("raw_text", "")[:4000].strip()
    if not methodology_text:
        methodology_text = "No methodology text was extracted from the source document."

    if img:
        try:
            b64_str = base64.b64encode(Path(img).read_bytes()).decode("utf-8")
            msg = HumanMessage(content=[{"type": "text", "text": f"Context:\n{methodology_text[:3000]}"}, {"type": "image_url", "image_url": f"data:image/png;base64,{b64_str}"}])
            res = invoke_llm_with_retry([SystemMessage(content=system_prompt), msg])
            return {"architecture_md": f"## Architecture\n\n{_get_clean_content(res)}\n\n![System Diagram]({img})", "architecture_image_path": img}
        except Exception: pass

    res = invoke_llm_with_retry([SystemMessage(content=system_prompt), HumanMessage(content=methodology_text)])
    return {"architecture_md": f"## Architecture\n\n{_get_clean_content(res)}", "architecture_image_path": None}

def _extract_figure(pdf_path: str) -> Optional[str]:
    import fitz
    scoring_keywords = {"architecture": 10, "pipeline": 10, "framework": 8, "overview": 8}
    try:
        doc = fitz.open(pdf_path)
        best_candidate = None
        for page_index, page in enumerate(doc):
            text = page.get_text().lower()
            page_score = sum(w for kw, w in scoring_keywords.items() if kw in text)
            if page_score == 0: continue
            imgs = page.get_images(full=True)
            for img in imgs:
                xref = img[0]
                rects = page.get_image_rects(xref)
                if not rects: continue
                total_score = page_score + ((rects[0].width * rects[0].height) / 50000)
                if best_candidate is None or total_score > best_candidate[0]:
                    best_candidate = (total_score, page_index, xref)
        if best_candidate:
            _, p_idx, xref = best_candidate
            pix = fitz.Pixmap(doc, xref)
            out_path = Path("extracted_figures") / f"arch_{Path(pdf_path).stem}.png"
            out_path.parent.mkdir(exist_ok=True)
            if pix.n - pix.alpha >= 4: pix = fitz.Pixmap(fitz.csRGB, pix)
            pix.save(str(out_path))
            doc.close()
            return str(out_path)
        doc.close()
    except Exception: pass
    return None

# -----------------------------
# 9) Technology Section Formatting
# -----------------------------
def technology_formatting_node(state: State) -> dict:
    techs = state.get("technologies", [])
    if not techs:
        return {"technology_md": "## Technologies\n\n_No core components discovered._"}

    blocks = ["## Technologies"]
    for tech in techs:
        block_lines = [
            f"### {tech['name']} ({tech['category']})",
            f"**Purpose in paper:** {tech['role_in_paper']}",
            f"**General context:** {tech['description']}",
            f"**Why learn it?** {tech.get('why_learn_it', 'Essential foundational optimization tool for domain-specific execution pipelines.')}"
        ]
        blocks.append("\n\n".join(block_lines))

    return {"technology_md": clean_markdown("\n\n".join(blocks))}

# -----------------------------
# 10) Context Literature Agents
# -----------------------------
def related_paper_agent(state: State) -> dict:
    s = state["sections"]
    n = state.get("num_related_papers", 5)

    # Keep the title snippet short — a full ~200-char title plus keywords
    # makes for a noisy search query, so cap it to the first several words.
    title_snippet = " ".join((s.title or "").split()[:12])

    domain_kws = state.get("domain_keywords") or []
    if domain_kws:
        concept_query = f"{title_snippet} {' '.join(domain_kws[:3])} research paper".strip()
    else:
        core_terms = _extract_core_terms(state)
        concept_query = f"{title_snippet} {' '.join(core_terms[:3])} research paper".strip()

    final_candidates = _fallback_search_chain(concept_query, max_results=n)
    if not final_candidates: return {"related_papers": []}

    pack = invoke_llm_with_retry(
        [
            SystemMessage(content=(
                f"Build a comparative literature pack. {FORMATTING_RULES}\n"
                "For each related paper, also identify publication_year and venue "
                "(e.g. IEEE, ACM, Springer, Elsevier, arXiv) strictly from the candidate "
                "source data provided. If a candidate does not clearly indicate a year "
                "or venue, leave that field null rather than guessing."
            )),
            HumanMessage(content=f"Baseline Paper: {s.title}\nMethodology:\n{s.methodology[:1500]}\n\nCandidates:\n{final_candidates}")
        ],
        structured_output_cls=RelatedPapersPack
    )
    return {"related_papers": [p.model_dump() for p in pack.papers]}

def gap_and_advances_agent(state: State) -> dict:
    related = state.get("related_papers", [])
    s = state["sections"]

    def _format_related_row(p: dict) -> str:
        meta_bits = []
        if p.get("venue"):
            meta_bits.append(str(p["venue"]))
        if p.get("publication_year"):
            meta_bits.append(str(p["publication_year"]))
        meta_suffix = f" ({', '.join(meta_bits)})" if meta_bits else ""
        return f"- **{p['title']}**{meta_suffix}: {p['summary']}"

    rows = "\n".join(_format_related_row(p) for p in related)
    comp_md = f"## Comparison with Related Papers\n\n{rows or '_No references found._'}"

    domain_kws = state.get("domain_keywords") or ["hardware architecture"]
    trend_query = f"{' '.join(domain_kws[:3])} recent research advancements"
    candidates = _fallback_search_chain(trend_query, max_results=5)

    res = invoke_llm_with_retry(
        [
            SystemMessage(content=(
                "Synthesize current domain trends and research directions.\n" + FORMATTING_RULES +
                "\nAppend '[Confidence: ★★★☆☆]' to AI Suggested Future Directions header."
            )),
            HumanMessage(content=f"Title: {s.title}\nAbstract:\n{s.abstract}\nCandidates:\n{candidates}")
        ],
        structured_output_cls=GapAndAdvancesPack
    )

    gap_md = f"## Research Gap Analysis [Confidence: ★★★★☆]\n\n### Research Trend\n{res.research_trend_md}\n\n### Current SOTA\n{res.current_sota_md}"
    sug_md = f"## AI-Suggested Future Directions [Confidence: ★★★☆☆]\n\n" + "\n".join(f"- {d}" for d in res.ai_suggested_directions)

    adv_lines = [f"### [{a.title}]({a.url})\n{a.summary}\n_Relevance:_ {a.relevance_to_paper}" for a in res.recent_advances]
    recent_md = "## Recent Advancements\n\n" + "\n\n".join(adv_lines) if adv_lines else "## Recent Advancements\n\n_No recent developments located._"

    return {
        "comparison_md": clean_markdown(comp_md),
        "research_gap_md": clean_markdown(gap_md),
        "ai_suggested_research_md": clean_markdown(sug_md),
        "recent_advances_md": clean_markdown(recent_md),
        "recent_advances": [a.model_dump() for a in res.recent_advances]
    }

def _extract_core_terms(state: State, max_terms: int = 12) -> List[str]:
    s = state["sections"]
    text = f"{s.title} {s.abstract} {s.methodology}"[:3000]
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", text)
    freq: Dict[str, int] = {}
    for tok in tokens:
        key = tok.lower()
        if key in _STOPWORDS: continue
        freq[key] = freq.get(key, 0) + 1
    scored = sorted(freq.items(), key=lambda x: -x[1])
    return [k for k, _ in scored[:max_terms]]

_STOPWORDS = {"the", "and", "for", "with", "based", "using", "paper", "approach", "method", "system"}

# -----------------------------
# 11) Enhanced Study Material & Strict Deduplication
# -----------------------------
def study_material_agent(state: State) -> dict:
    if state.get("detail_level") == "abstract":
        return {"study_materials": {}}

    keywords = state.get("domain_keywords") or ["hardware acceleration"]
    target_keywords = keywords[:4]

    materials = {"books": [], "blogs": [], "papers": [], "github": []}

    for kw in target_keywords:
        found_links = _fallback_search_chain(f'"{kw}" source documentation tutorial textbook', max_results=3)
        for item in found_links:
            url = (item.get("url") or "").lower()

            # Classify using explicit URL properties. Known technical-book
            # publishers are checked explicitly so they land in "books"
            # rather than silently falling through to the generic "blogs"
            # bucket (Springer, Wiley, MIT Press, O'Reilly, and Packt are all
            # frequent hits for the "textbook" query above).
            book_publisher_domains = [
                "books.google", "openlibrary",
                "springer.com", "wiley.com", "mitpress.mit.edu",
                "oreilly.com", "packtpub.com",
            ]
            if "github.com" in url:
                category = "github"
            elif any(domain in url for domain in book_publisher_domains):
                category = "books"
            elif any(domain in url for domain in ["semanticscholar", "arxiv.org", "openalex"]):
                category = "papers"
            else:
                category = "blogs"

            materials[category].append({
                "title": item["title"],
                "url": item["url"],
                "description": item["snippet"][:200]
            })

    # Dedup by title+URL (see dedupe_study_items) rather than URL alone, so
    # mirrors of the same resource under a different URL don't both survive.
    for category in ["books", "blogs", "papers", "github"]:
        materials[category] = dedupe_study_items(materials[category])

    return {"study_materials": materials}

# -----------------------------
# 12) Learning Roadmap Custom Formatting
# -----------------------------
def learning_roadmap_agent(state: State) -> dict:
    s = state["sections"]
    planner = invoke_llm_with_retry(
        [
            SystemMessage(content=f"Build pedagogical engineering steps and pipeline prototyping structures. {FORMATTING_RULES}"),
            HumanMessage(content=f"Abstract:\n{s.abstract}\nMethodology:\n{s.methodology[:2000]}")
        ],
        structured_output_cls=LearningRoadmap
    )

    roadmap_blocks = ["## Recommended Learning Roadmap"]
    sorted_steps = sorted(planner.steps, key=lambda x: x.order)

    for step in sorted_steps:
        step_entry = (
            f"### Step {step.order}: {step.topic}\n"
            f"- **Difficulty:** {step.difficulty.upper()}\n"
            f"- **Estimated Time Allocation:** {step.estimated_hours} Hours\n"
            f"- **Pedagogical Objective:** {step.reason}"
        )
        roadmap_blocks.append(step_entry)

    return {"learning_roadmap_md": clean_markdown("\n\n".join(roadmap_blocks))}

def join_node(state: State) -> dict: return {}

# -----------------------------
# 13) Master Report Construction Execution
# -----------------------------
def merge_report(state: State) -> dict:
    parts = []
    for key, _ in SECTION_ORDER:
        if key == "study_materials_md":
            mat = state.get("study_materials", {})
            blocks = ["## Study Materials\n"]
            labels = {"books": "Books", "blogs": "Blogs", "papers": "Research Papers", "github": "GitHub Repositories"}
            for category_key, title_label in labels.items():
                unique_items = dedupe_study_items(mat.get(category_key) or [])
                if unique_items:
                    blocks.append(f"### {title_label}\n" + "\n".join(f"- [{it['title']}]({it['url']})" for it in unique_items))
            if len(blocks) > 1: parts.append("\n\n".join(blocks))
        else:
            v = state.get(key)
            if v: parts.append(v)

    title = state["sections"].title or "Analysis Report"
    return {"merged_md": clean_markdown(f"# {title}\n\n" + "\n\n".join(parts))}

def dedupe_and_flow(state: State) -> dict:
    return {"merged_md": clean_markdown(state["merged_md"])}

def generate_blog(state: State) -> dict:
    return {"final_report_md": clean_markdown(state["merged_md"])}

SECTION_ORDER = [
    ("summary_md", "summary"), ("architecture_md", "architecture"), ("technology_md", "technology"),
    ("methodology_md", "methodology"), ("results_md", "results"), ("limitations_md", "limitations"),
    ("open_problems_md", "open_problems"), ("research_gap_md", "research_gap"),
    ("ai_suggested_research_md", "ai_suggested_research"), ("recent_advances_md", "recent_advances"),
    ("comparison_md", "comparison"), ("study_materials_md", "study_materials"), ("learning_roadmap_md", "learning_roadmap")
]

report_graph = StateGraph(State)
report_graph.add_node("merge_report", merge_report)
report_graph.add_node("dedupe_and_flow", dedupe_and_flow)
report_graph.add_node("generate_blog", generate_blog)
report_graph.add_edge(START, "merge_report")
report_graph.add_edge("merge_report", "dedupe_and_flow")
report_graph.add_edge("dedupe_and_flow", "generate_blog")
report_graph.add_edge("generate_blog", END)
report_subgraph = report_graph.compile()

# -----------------------------
# 14) Master Graph Infrastructure
# -----------------------------
g = StateGraph(State)
g.add_node("pdf_extraction", pdf_extraction_node)
g.add_node("router", router_node)
g.add_node("unified_report_agent", unified_report_agent)
g.add_node("technology_formatting_node", technology_formatting_node)
g.add_node("study_material_agent", study_material_agent)
g.add_node("architecture_agent", architecture_agent)
g.add_node("related_paper_agent", related_paper_agent)
g.add_node("gap_and_advances_agent", gap_and_advances_agent)
g.add_node("learning_roadmap_agent", learning_roadmap_agent)
g.add_node("join", join_node)
g.add_node("report_writer", report_subgraph)

g.add_edge(START, "pdf_extraction")
g.add_edge("pdf_extraction", "router")
g.add_conditional_edges("router", route_from_router, ["unified_report_agent", "architecture_agent", "related_paper_agent", "learning_roadmap_agent"])

g.add_edge("unified_report_agent", "technology_formatting_node")
g.add_edge("technology_formatting_node", "study_material_agent")
g.add_edge("study_material_agent", "join")
g.add_edge("architecture_agent", "join")
g.add_edge("related_paper_agent", "gap_and_advances_agent")
g.add_edge("gap_and_advances_agent", "join")
g.add_edge("learning_roadmap_agent", "join")
g.add_edge("join", "report_writer")
g.add_edge("report_writer", END)

memory = MemorySaver()
app = g.compile(checkpointer=memory)