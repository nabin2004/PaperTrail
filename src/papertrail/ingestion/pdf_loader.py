import requests
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Union
import re

RAW_DIR = Path("data/raw_papers")
RAW_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ------------------ Text Cleaning ------------------ #
def clean_text(text: str) -> str:
    """
    Cleans raw text from PDFs for better embeddings.
    """
    # Normalize unicode
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u2013', "-").replace('\u2014', "--")

    # Collapse multiple newlines and spaces
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    # Join lines that don't end with punctuation
    text = re.sub(r'(?<![\.\?!])\n', ' ', text)

    # Remove references / footnotes (lines starting with numbers or [number])
    text = re.sub(r'\n?\s*\[?\d+\]?\s?.*$', '', text, flags=re.MULTILINE)

    return text.strip()


def clean_file(file_path: Union[str, Path]) -> str:
    """
    Reads a text file, cleans it, and returns cleaned text.
    """
    file_path = Path(file_path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    raw_text = file_path.read_text(encoding='utf-8')
    return clean_text(raw_text)


def clean_files(file_paths: Union[Path, str, List[Union[str, Path]]]) -> List[str]:
    """
    Cleans a single or list of text files and returns cleaned texts.
    """
    if isinstance(file_paths, (str, Path)):
        file_paths = [file_paths]
    return [clean_file(fp) for fp in file_paths]


# ------------------ PDF Handling ------------------ #
def download_pdf(paper) -> Union[Path, None]:
    """Download PDF from Arxiv Client output"""
    if not getattr(paper, "pdf_url", None):
        return None
    file_path = RAW_DIR / f"{paper.arxiv_id}.pdf"
    if file_path.exists():
        return file_path
    r = requests.get(paper.pdf_url)
    r.raise_for_status()
    file_path.write_bytes(r.content)
    return file_path


def extract_text(pdf_path: Union[str, Path]) -> str:
    """Extract text from a PDF using PyMuPDF"""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()


def save_processed_text(pdf_path: Union[str, Path], clean: bool = True) -> Path:
    """
    Extracts text from a PDF, optionally cleans it, and saves it to the processed folder.
    Returns the path of the processed text file.
    """
    pdf_path = Path(pdf_path)
    text = extract_text(pdf_path)
    # if clean:
    #     text = clean_text(text)
    processed_path = PROCESSED_DIR / f"{pdf_path.stem}.txt"
    processed_path.write_text(text, encoding="utf-8")
    return processed_path
