"""
Text cleaning utilities for raw PDF-extracted text.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import List, Union


# ──────────────────────────────────────────────────────────────
# Core cleaner
# ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean raw PDF-extracted text for downstream embedding and analysis.

    Steps:
      1. Normalise unicode
      2. Collapse whitespace
      3. Re-join hyphenated line-wrapped words
      4. Remove lone page numbers / header/footer noise
      5. Strip excessive blank lines
    """
    # 1. Normalise unicode to NFC and replace common curly quotes / dashes
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "--")
    text = text.replace("\u00a0", " ")          # non-breaking space
    text = text.replace("\u200b", "")            # zero-width space

    # 2. Re-join hyphenated words split across lines  (e.g. "trans-\nformer")
    text = re.sub(r"-\n(\w)", r"\1", text)

    # 3. Collapse multiple spaces/tabs within a line
    text = re.sub(r"[ \t]+", " ", text)

    # 4. Remove lines that look like lone page numbers (just digits / roman numerals)
    text = re.sub(r"^\s*[ivxlcdmIVXLCDM\d]+\s*$", "", text, flags=re.MULTILINE)

    # 5. Collapse 3+ consecutive blank lines → double blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ──────────────────────────────────────────────────────────────
# File-level helpers
# ──────────────────────────────────────────────────────────────

def clean_file(file_path: Union[str, Path]) -> str:
    """Read a text file, clean it, and return the result."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return clean_text(path.read_text(encoding="utf-8"))


def clean_files(file_paths: Union[str, Path, List[Union[str, Path]]]) -> List[str]:
    """Clean one or many text files; returns list of cleaned strings."""
    if isinstance(file_paths, (str, Path)):
        file_paths = [file_paths]
    return [clean_file(fp) for fp in file_paths]


def clean_and_save(file_path: Union[str, Path], output_path: Union[str, Path, None] = None) -> Path:
    """
    Clean a text file and write the result to *output_path*.
    If output_path is None the same file is overwritten.
    Returns the path of the cleaned file.
    """
    src = Path(file_path)
    dst = Path(output_path) if output_path else src
    dst.write_text(clean_text(src.read_text(encoding="utf-8")), encoding="utf-8")
    return dst
