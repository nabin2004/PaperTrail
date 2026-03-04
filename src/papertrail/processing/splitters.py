"""
Text splitter – reads a processed .txt file, splits into overlapping chunks,
and writes a JSONL file under data/chunks/.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CHUNKS_DIR = _DATA_DIR / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)


def split_and_save(file_path: str | Path) -> Path:
    """
    Split *file_path* (cleaned .txt) into overlapping chunks and save as JSONL.

    Returns the path of the written .jsonl file.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    chunks: List[str] = _splitter.split_text(text)

    output_path = CHUNKS_DIR / f"{file_path.stem}.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks):
            record: Dict = {
                "paper_id": file_path.stem,
                "chunk_id": idx,
                "text": chunk,
                "source": str(file_path),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Saved %d chunks → %s", len(chunks), output_path)
    return output_path


def load_chunks(paper_id: str) -> List[Dict]:
    """Load previously saved chunks for a given paper ID."""
    path = CHUNKS_DIR / f"{paper_id}.jsonl"
    if not path.exists():
        return []
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks

