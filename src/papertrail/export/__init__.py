"""
Export utilities – export research reports to Obsidian, Notion, and other formats.

TODO: Implement export integrations.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from papertrail.schemas.schema import ResearchReport, Paper

_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
EXPORT_DIR = _DATA_DIR / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# Obsidian export
# ──────────────────────────────────────────────────────────────

class ObsidianExporter:
    """
    Export papers and reports to Obsidian vault format.

    TODO
    ----
    - Create linked Markdown notes with proper wikilinks
    - Generate MOC (Map of Content) pages
    - Support Obsidian properties/frontmatter
    - Create citation backlinks between papers
    """

    def __init__(self, vault_path: Optional[Path] = None) -> None:
        self.vault_path = vault_path or EXPORT_DIR / "obsidian"
        self.vault_path.mkdir(parents=True, exist_ok=True)

    def export_paper(self, paper: Paper) -> Path:
        """
        Export a single paper as an Obsidian note.

        Returns
        -------
        Path
            Path to the created note file.

        TODO
        ----
        - YAML frontmatter with metadata
        - Wikilinks to related papers
        - Tags from categories
        - Callouts for key findings
        """
        note_path = self.vault_path / f"{paper.arxiv_id}.md"

        content = f"""---
arxiv_id: "{paper.arxiv_id}"
title: "{paper.title}"
authors: {paper.authors}
published: "{paper.published}"
categories: {paper.categories}
tags: [paper, {paper.primary_category}]
---

# {paper.title}

**Authors:** {', '.join(paper.authors)}
**Published:** {paper.published}
**Categories:** {', '.join(paper.categories)}
**arXiv:** [Link](https://arxiv.org/abs/{paper.arxiv_id})

## Abstract

{paper.abstract}

## Notes

<!-- Your notes here -->

## Related Papers

<!-- Links to related papers will be auto-generated -->

"""
        note_path.write_text(content, encoding="utf-8")
        return note_path

    def export_report(self, report: ResearchReport) -> Path:
        """
        Export a research report as an Obsidian note.

        TODO
        ----
        - Link to source paper notes
        - Collapsible sections for synthesis/critique
        - Evaluation scores as metadata
        """
        timestamp = report.created_at.strftime("%Y%m%d_%H%M%S")
        note_path = self.vault_path / f"Report_{timestamp}.md"

        content = f"""---
type: research-report
question: "{report.question}"
faithfulness: {report.faithfulness_score:.2f}
coverage: {report.coverage_score:.2f}
created: "{report.created_at}"
tags: [report, research]
---

# Research Report

**Question:** {report.question}

## Synthesis

{report.synthesis}

## Critique

{report.critique}

## Sources

"""
        for i, src in enumerate(report.sources, 1):
            content += f"{i}. [[{src.paper_id}]] – {src.title or 'Untitled'}\n"

        note_path.write_text(content, encoding="utf-8")
        return note_path

    def generate_moc(self, title: str = "Paper Index") -> Path:
        """
        Generate a Map of Content for all exported papers.

        TODO
        ----
        - Group by category/date
        - Include key stats
        - Auto-update on new exports
        """
        moc_path = self.vault_path / f"{title}.md"
        # Stub
        return moc_path


# ──────────────────────────────────────────────────────────────
# Notion export
# ──────────────────────────────────────────────────────────────

class NotionExporter:
    """
    Export papers and reports to Notion using the Notion API.

    TODO
    ----
    - Implement Notion API integration
    - Create database entries for papers
    - Rich text formatting with blocks
    - Bi-directional sync
    """

    def __init__(self, api_key: Optional[str] = None, database_id: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("NOTION_API_KEY")
        self.database_id = database_id or os.getenv("NOTION_DATABASE_ID")
        self._client = None  # TODO: Initialize notion_client

    def _get_client(self):
        """
        Initialize the Notion client.

        TODO
        ----
        - pip install notion-client
        - Authenticate with API key
        """
        if self._client is None:
            raise NotImplementedError("Notion client not configured. Set NOTION_API_KEY.")
        return self._client

    def export_paper(self, paper: Paper) -> str:
        """
        Create a Notion page for a paper.

        Returns
        -------
        str
            Notion page ID.

        TODO
        ----
        - Create page in database
        - Set properties (title, authors, date, etc.)
        - Add content blocks
        """
        raise NotImplementedError("Notion export not yet implemented")

    def export_report(self, report: ResearchReport) -> str:
        """
        Create a Notion page for a research report.

        TODO
        ----
        - Create page with toggle blocks for sections
        - Link to source paper pages
        - Add evaluation metrics
        """
        raise NotImplementedError("Notion export not yet implemented")

    def sync_papers(self, papers: List[Paper]) -> Dict[str, str]:
        """
        Sync multiple papers to Notion, skipping existing ones.

        TODO
        ----
        - Query existing pages by arxiv_id
        - Batch create/update
        - Return mapping of arxiv_id -> page_id
        """
        raise NotImplementedError("Notion sync not yet implemented")


# ──────────────────────────────────────────────────────────────
# Other export formats
# ──────────────────────────────────────────────────────────────

def export_to_json(report: ResearchReport, path: Optional[Path] = None) -> Path:
    """
    Export a report to JSON format.

    TODO
    ----
    - Full schema serialization
    - Pretty-print option
    - Include nested objects
    """
    import json
    path = path or EXPORT_DIR / f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    data = report.model_dump(mode="json")
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def export_to_bibtex(papers: List[Paper], path: Optional[Path] = None) -> Path:
    """
    Export papers to BibTeX format.

    TODO
    ----
    - Generate proper BibTeX entries
    - Handle special characters
    - Include DOI when available
    """
    path = path or EXPORT_DIR / "papers.bib"

    entries = []
    for paper in papers:
        author = " and ".join(paper.authors[:5])
        year = paper.published.year
        entry = f"""@article{{{paper.arxiv_id},
  title = {{{paper.title}}},
  author = {{{author}}},
  year = {{{year}}},
  eprint = {{{paper.arxiv_id}}},
  archivePrefix = {{arXiv}},
  primaryClass = {{{paper.primary_category}}}
}}
"""
        entries.append(entry)

    path.write_text("\n".join(entries), encoding="utf-8")
    return path


def export_to_csv(papers: List[Paper], path: Optional[Path] = None) -> Path:
    """
    Export papers to CSV format.

    TODO
    ----
    - Include all metadata fields
    - Handle commas in titles/abstracts
    - Support custom column selection
    """
    import csv
    path = path or EXPORT_DIR / "papers.csv"

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["arxiv_id", "title", "authors", "published", "category", "abstract"])
        for p in papers:
            writer.writerow([
                p.arxiv_id,
                p.title,
                "; ".join(p.authors),
                str(p.published.date()),
                p.primary_category,
                p.abstract[:500],
            ])

    return path
