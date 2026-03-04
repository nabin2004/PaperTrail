"""
Metadata store – persists Paper objects as JSON files under data/metadata/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from papertrail.schemas.schema import Paper

_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
METADATA_DIR = _DATA_DIR / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# Write
# ──────────────────────────────────────────────────────────────

def save_metadata(paper: Paper) -> Path:
    """Persist a Paper object to a JSON file.  Returns the written path."""
    path = METADATA_DIR / f"{paper.arxiv_id}.json"
    data = paper.model_dump(mode="json")
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


# ──────────────────────────────────────────────────────────────
# Read
# ──────────────────────────────────────────────────────────────

def load_metadata(arxiv_id: str) -> Optional[Paper]:
    """Load a single Paper by arXiv ID.  Returns None if not found."""
    path = METADATA_DIR / f"{arxiv_id}.json"
    if not path.exists():
        return None
    return Paper(**json.loads(path.read_text(encoding="utf-8")))


def load_all_metadata() -> List[Paper]:
    """Load all stored papers."""
    papers: List[Paper] = []
    for f in sorted(METADATA_DIR.glob("*.json")):
        try:
            papers.append(Paper(**json.loads(f.read_text(encoding="utf-8"))))
        except Exception:
            pass  # skip malformed files
    return papers


def list_paper_ids() -> List[str]:
    """Return arXiv IDs of all stored papers."""
    return [f.stem for f in sorted(METADATA_DIR.glob("*.json"))]


def metadata_exists(arxiv_id: str) -> bool:
    return (METADATA_DIR / f"{arxiv_id}.json").exists()


def get_paper_index() -> Dict[str, str]:
    """Return a dict mapping arxiv_id → title for quick look-ups."""
    index: Dict[str, str] = {}
    for paper in load_all_metadata():
        index[paper.arxiv_id] = paper.title
    return index
