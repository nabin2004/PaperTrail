"""
PDF download, text extraction and initial processing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union

import fitz  # PyMuPDF
import requests

_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

RAW_DIR = _DATA_DIR / "raw_papers"
RAW_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = _DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# PDF handling
# ──────────────────────────────────────────────────────────────

def download_pdf(paper) -> Union[Path, None]:
    """
    Download the PDF for *paper* (Paper schema object) to RAW_DIR.
    Returns the local path, or None if there is no pdf_url.
    Skips the download if the file already exists.
    """
    if not getattr(paper, "pdf_url", None):
        return None
    file_path = RAW_DIR / f"{paper.arxiv_id}.pdf"
    if file_path.exists():
        return file_path
    r = requests.get(str(paper.pdf_url), timeout=60)
    r.raise_for_status()
    file_path.write_bytes(r.content)
    return file_path


def extract_text(pdf_path: Union[str, Path]) -> str:
    """Extract raw text from *pdf_path* using PyMuPDF."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    return text.strip()


def save_processed_text(pdf_path: Union[str, Path], clean: bool = True) -> Path:
    """
    Extract text from *pdf_path*, optionally clean it, and write a .txt file to
    PROCESSED_DIR.  Returns the path of the processed text file.
    """
    pdf_path = Path(pdf_path)
    text = extract_text(pdf_path)

    if clean:
        from papertrail.processing.cleaners import clean_text
        text = clean_text(text)

    processed_path = PROCESSED_DIR / f"{pdf_path.stem}.txt"
    processed_path.write_text(text, encoding="utf-8")
    return processed_path


# ──────────────────────────────────────────────────────────────
# Batch helpers
# ──────────────────────────────────────────────────────────────

def extract_texts(pdf_paths: List[Union[str, Path]]) -> List[str]:
    """Extract text from multiple PDFs."""
    return [extract_text(p) for p in pdf_paths]

