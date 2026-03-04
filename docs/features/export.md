# Export to Obsidian / Notion

Export papers and research reports to popular note-taking applications.

## Status

🔲 **Not Implemented** – Skeleton available

## Overview

Export functionality allows researchers to integrate PaperTrail with their existing knowledge management workflows in Obsidian, Notion, or other tools.

## Files

- [export/__init__.py](../src/papertrail/export/__init__.py) – Export utilities

## Features to Implement

### Obsidian Export

```python
from papertrail.export import ObsidianExporter

exporter = ObsidianExporter(vault_path="/path/to/vault")

# Export single paper
note_path = exporter.export_paper(paper)

# Export research report
report_path = exporter.export_report(report)

# Generate Map of Content
moc_path = exporter.generate_moc("AI Papers Index")
```

### Notion Export

```python
from papertrail.export import NotionExporter

exporter = NotionExporter(
    api_key="ntn_xxx",
    database_id="xxx-xxx-xxx"
)

# Export paper to database
page_id = exporter.export_paper(paper)

# Bulk sync
mapping = exporter.sync_papers(papers)
```

### Other Formats

```python
from papertrail.export import export_to_json, export_to_bibtex, export_to_csv

# JSON export
json_path = export_to_json(report)

# BibTeX for citation managers
bib_path = export_to_bibtex(papers)

# CSV for spreadsheet analysis
csv_path = export_to_csv(papers)
```

## Implementation Tasks

### Obsidian

- [ ] YAML frontmatter with metadata
- [ ] Wikilinks between related papers `[[paper_id]]`
- [ ] Tags from arXiv categories
- [ ] Callout blocks for key findings
- [ ] Map of Content (MOC) generation
- [ ] Template customization
- [ ] Daily notes integration
- [ ] Dataview-compatible properties

### Notion

- [ ] Notion API client setup
- [ ] Database schema creation
- [ ] Page creation with properties
- [ ] Rich text block formatting
- [ ] Toggle blocks for sections
- [ ] Relation properties for linked papers
- [ ] Bi-directional sync
- [ ] Cover images from figures

### Other Formats

- [ ] JSON export (full schema)
- [ ] BibTeX bibliography
- [ ] CSV export with custom columns
- [ ] Markdown archive
- [ ] Zotero RDF export
- [ ] EndNote XML export

## Obsidian Note Template

```markdown
---
arxiv_id: "2303.08774"
title: "GPT-4 Technical Report"
authors: ["OpenAI"]
published: "2023-03-15"
categories: ["cs.CL", "cs.AI"]
tags: [paper, cs.CL, language-model]
status: unread
---

# {{title}}

**Authors:** {{authors}}
**Published:** {{published}}
**arXiv:** [Link](https://arxiv.org/abs/{{arxiv_id}})

## Abstract

{{abstract}}

## Notes

<!-- Your notes here -->

## Key Findings

> [!note] Finding 1
> ...

## Related Papers

- [[related_paper_1]]
- [[related_paper_2]]

## Citations

```bibtex
@article{...}
```
```

## Notion Database Schema

| Property | Type | Description |
|---|---|---|
| Title | Title | Paper title |
| arXiv ID | Text | Unique identifier |
| Authors | Multi-select | Author names |
| Published | Date | Publication date |
| Categories | Multi-select | arXiv categories |
| Status | Select | unread/reading/read |
| Abstract | Text | Full abstract |
| PDF | URL | Link to PDF |
| Notes | Relation | Linked note pages |
| Rating | Number | 1-5 star rating |

## CLI Integration

```bash
# Export to Obsidian vault
papertrail export obsidian --vault ~/Notes/Research

# Export to Notion
papertrail export notion --database "Papers"

# Export bibliography
papertrail export bibtex --output papers.bib

# Export all to JSON
papertrail export json --output backup.json
```

## Dependencies

```toml
# Notion
notion-client = ">=2.2.0"

# No additional deps for Obsidian (plain Markdown)
```

## References

- [Obsidian](https://obsidian.md/)
- [Notion API](https://developers.notion.com/)
- [notion-client (Python)](https://github.com/ramnes/notion-sdk-py)
- [BibTeX Format](http://www.bibtex.org/Format/)
