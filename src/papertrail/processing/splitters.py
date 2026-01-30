import json
import logging
from pathlib import Path
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)

PROCESSED_DIR = Path("data")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_DIR = PROCESSED_DIR / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

def split_and_save(file_path: str) -> Path:
    """
    Splits a cleaned text file into chunks and saves them as JSONL.
    """
    file_path = Path(file_path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    chunks = text_splitter.split_text(text)

    output_path = CHUNKS_DIR / f"{file_path.stem}.jsonl"

    with output_path.open("w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks):
            record: Dict = {
                "paper_id": file_path.stem,
                "chunk_id": idx,
                "text": chunk,
                "source": str(file_path)
            }
            f.write(json.dumps(record) + "\n")

    logging.info(f"Saved {len(chunks)} chunks → {output_path}")
    return output_path
