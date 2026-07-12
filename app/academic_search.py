from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

import requests

TIMEOUT = 10
HEADERS = {
    "User-Agent": "ResearchIntelligenceAssistant/1.0 (mailto:your-email@example.com)"
}

def search_semantic_scholar(query: str, limit: int = 5) -> List[Dict]:
    try:
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": limit, "fields": "title,abstract,url,year,venue"},
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        out = []
        for p in data:
            out.append({
                "title": p.get("title") or "",
                "url": p.get("url") or "",
                "snippet": (p.get("abstract") or "")[:500],
                "source": "Semantic Scholar",
                "year": p.get("year"),
            })
        return out
    except Exception:
        return []


def search_openalex(query: str, limit: int = 5) -> List[Dict]:
    try:
        params = {"search": query, "per-page": limit, "mailto": "your-email@example.com"}
        resp = requests.get(
            "https://api.openalex.org/works",
            params=params,
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        out = []
        for w in results:
            abstract = ""
            inv_idx = w.get("abstract_inverted_index")
            if inv_idx:
                positions: Dict[int, str] = {}
                for word, idxs in inv_idx.items():
                    for i in idxs:
                        positions[i] = word
                abstract = " ".join(positions[i] for i in sorted(positions))[:500]
            out.append({
                "title": w.get("title") or w.get("display_name") or "",
                "url": w.get("id") or (w.get("primary_location") or {}).get("landing_page_url") or "",
                "snippet": abstract,
                "source": "OpenAlex",
                "year": w.get("publication_year"),
            })
        return out
    except Exception:
        return []


def search_arxiv(query: str, limit: int = 5) -> List[Dict]:
    try:
        resp = requests.get(
            "http://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "start": 0, "max_results": limit},
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        out = []
        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            link_el = entry.find("atom:id", ns)
            published_el = entry.find("atom:published", ns)
            
            year = None
            if published_el is not None and published_el.text:
                try:
                    year = int(published_el.text.split("-")[0])
                except ValueError:
                    pass

            out.append({
                "title": (title_el.text or "").strip() if title_el is not None else "",
                "url": (link_el.text or "").strip() if link_el is not None else "",
                "snippet": (summary_el.text or "").strip()[:500] if summary_el is not None else "",
                "source": "arXiv",
                "year": year,
            })
        return out
    except Exception:
        return []


def search_all_academic_sources(query: str, limit_per_source: int = 4) -> List[Dict]:
    """Queries Semantic Scholar, OpenAlex, and arXiv concurrently."""
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_ss = executor.submit(search_semantic_scholar, query, limit_per_source)
        future_oa = executor.submit(search_openalex, query, limit_per_source)
        future_ax = executor.submit(search_arxiv, query, limit_per_source)
        
        combined = future_ss.result() + future_oa.result() + future_ax.result()

    seen = set()
    deduped = []
    for item in combined:
        key = item["title"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped