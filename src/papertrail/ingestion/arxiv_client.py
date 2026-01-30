import requests
import xml.etree.ElementTree as ET
from typing import List
from papertrail.schemas.schema import Paper

class ArxivClient:
    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self, categories: List[str], max_results: int = 10):
        self.categories = categories
        self.max_results = max_results

    def build_query_url(self) -> str:
        # Join categories with OR
        cat_query = "+OR+".join([f"cat:{cat}" for cat in self.categories])
        url = f"{self.BASE_URL}?search_query={cat_query}&max_results={self.max_results}"
        return url

    def fetch_papers(self) -> List[Paper]:
        url = self.build_query_url()
        response = requests.get(url)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom"
        }

        papers = []
        for entry in root.findall("atom:entry", ns):
            arxiv_id = entry.find("atom:id", ns).text.split("/")[-1]
            title = entry.find("atom:title", ns).text.strip()
            abstract = entry.find("atom:summary", ns).text.strip()
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
            primary_category = entry.find("arxiv:primary_category", ns).attrib["term"]
            categories = [c.attrib["term"] for c in entry.findall("atom:category", ns)]
            published = entry.find("atom:published", ns).text
            updated = entry.find("atom:updated", ns).text
            pdf_url = next((l.attrib["href"] for l in entry.findall("atom:link", ns) if l.attrib.get("title")=="pdf"), None)

            paper = Paper(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=authors,
                primary_category=primary_category,
                categories=categories,
                published=published,
                updated=updated,
                pdf_url=pdf_url
            )
            papers.append(paper)
        return papers

if __name__ == "__main__":
    client = ArxivClient(categories=["cs.AI", "cs.LG", "cs.CL", "cs.CV"], max_results=5)
    papers = client.fetch_papers()
    for p in papers:
        print(p.title, "-", p.arxiv_id, "-", p.authors, "-", p.abstract)
