import re
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Optional
import requests
from papertrail.schemas.schema import Paper


class ArxivClient:
    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        max_results: int = 10,
        query: Optional[str] = None,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ):
        self.categories = categories or []
        self.max_results = max_results
        self.query = query
        self.sort_by = sort_by
        self.sort_order = sort_order

    def build_query_url(self) -> str:
        query_parts = []
        if self.query and self.query.strip():
            q = self.query.strip()
            # If user hasn't specified explicit field prefix (ti:, all:, abs:, au:, cat:)
            if not any(q.startswith(prefix) for prefix in ("all:", "ti:", "au:", "abs:", "cat:")):
                # Use all: with quote encoding
                query_parts.append(f"all:{urllib.parse.quote_plus(q)}")
            else:
                query_parts.append(urllib.parse.quote_plus(q))

        if self.categories:
            valid_cats = [c.strip() for c in self.categories if c and c.strip()]
            if valid_cats:
                cat_expr = "+OR+".join([f"cat:{c}" for c in valid_cats])
                if query_parts:
                    query_parts.append(f"%28{cat_expr}%29")
                else:
                    query_parts.append(cat_expr)

        if not query_parts:
            search_query = "cat:cs.AI+OR+cat:cs.LG"
        elif len(query_parts) > 1:
            search_query = "+AND+".join(query_parts)
        else:
            search_query = query_parts[0]

        url = (
            f"{self.BASE_URL}?search_query={search_query}"
            f"&max_results={self.max_results}"
            f"&sortBy={self.sort_by}&sortOrder={self.sort_order}"
        )
        return url

    def fetch_papers(self) -> List[Paper]:
        url = self.build_query_url()
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        papers = []
        for entry in root.findall("atom:entry", ns):
            id_elem = entry.find("atom:id", ns)
            if id_elem is None or not id_elem.text:
                continue
            arxiv_id = id_elem.text.split("/")[-1]

            title_elem = entry.find("atom:title", ns)
            raw_title = title_elem.text if title_elem is not None and title_elem.text else ""
            title = " ".join(raw_title.split())

            summary_elem = entry.find("atom:summary", ns)
            raw_abstract = summary_elem.text if summary_elem is not None and summary_elem.text else ""
            abstract = " ".join(raw_abstract.split())

            authors = [
                a.find("atom:name", ns).text.strip()
                for a in entry.findall("atom:author", ns)
                if a.find("atom:name", ns) is not None and a.find("atom:name", ns).text
            ]

            primary_cat_elem = entry.find("arxiv:primary_category", ns)
            if primary_cat_elem is not None and "term" in primary_cat_elem.attrib:
                primary_category = primary_cat_elem.attrib["term"]
            else:
                primary_category = self.categories[0] if self.categories else "cs.AI"

            categories = [
                c.attrib["term"]
                for c in entry.findall("atom:category", ns)
                if "term" in c.attrib
            ]
            if not categories and primary_category:
                categories = [primary_category]

            pub_elem = entry.find("atom:published", ns)
            published = pub_elem.text if pub_elem is not None else ""

            upd_elem = entry.find("atom:updated", ns)
            updated = upd_elem.text if upd_elem is not None else published

            pdf_url = None
            for l in entry.findall("atom:link", ns):
                if l.attrib.get("title") == "pdf" or l.attrib.get("type") == "application/pdf":
                    pdf_url = l.attrib.get("href")
                    break
            if not pdf_url:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            paper = Paper(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=authors,
                primary_category=primary_category,
                categories=categories,
                published=published,
                updated=updated,
                pdf_url=pdf_url,
            )
            papers.append(paper)
        return papers


if __name__ == "__main__":
    client = ArxivClient(query="transformer attention mechanism", max_results=3)
    papers = client.fetch_papers()
    for p in papers:
        print(f"[{p.arxiv_id}] {p.title} ({', '.join(p.authors[:2])})")
