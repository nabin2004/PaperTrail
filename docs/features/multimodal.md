# Multi-Modal Support

Extract and analyze figures, tables, and equations from research papers.

## Status

🔲 **Not Implemented** – Skeleton available

## Overview

Multi-modal support enables PaperTrail to understand and search not just text, but also figures, tables, and equations within papers – providing comprehensive research coverage.

## Files

- [processing/multimodal.py](../src/papertrail/processing/multimodal.py) – Multi-modal extraction and processing

## Features to Implement

### Figure Extraction

```python
from papertrail.processing.multimodal import extract_figures, describe_figure

figures = extract_figures(pdf_path)
for fig in figures:
    fig.description = describe_figure(fig)
```

### Table Extraction

```python
from papertrail.processing.multimodal import extract_tables, table_to_markdown

tables = extract_tables(pdf_path)
for table in tables:
    print(table_to_markdown(table))
```

### Equation Extraction

```python
from papertrail.processing.multimodal import extract_equations

equations = extract_equations(pdf_path)
for eq in equations:
    print(eq.latex)
```

### Multi-Modal Search

```python
from papertrail.processing.multimodal import multimodal_search

results = multimodal_search(
    "attention mechanism diagram",
    include_figures=True,
    include_tables=True,
)
```

## Implementation Tasks

### Figure Processing

- [ ] Extract images from PDF using PyMuPDF
- [ ] Detect figure bounding boxes and captions
- [ ] Integrate vision model (GPT-4V, LLaVA, or BLIP-2)
- [ ] Create CLIP embeddings for image search
- [ ] Store figure embeddings in FAISS index

### Table Processing

- [ ] Extract tables using camelot-py or tabula-py
- [ ] Convert to pandas DataFrame
- [ ] Generate Markdown representation
- [ ] Handle multi-page tables
- [ ] Store table embeddings for search

### Equation Processing

- [ ] Extract equations using Nougat or Mathpix
- [ ] Convert to LaTeX format
- [ ] Optional: Integrate with SymPy for symbolic math

### Unified Search

- [ ] Score fusion across modalities
- [ ] Filter by content type
- [ ] Return mixed results with type indicators

## Data Classes

```python
@dataclass
class Figure:
    paper_id: str
    figure_id: int
    image_path: Path
    caption: str
    page_number: int
    bbox: tuple          # (x0, y0, x1, y1)
    description: str     # Vision model output
    embedding: List[float]

@dataclass
class Table:
    paper_id: str
    table_id: int
    raw_text: str
    structured_data: Dict
    caption: str
    markdown: str

@dataclass
class Equation:
    paper_id: str
    equation_id: int
    latex: str
    description: str
```

## Dependencies

```toml
# Figure extraction
camelot-py = ">=0.11.0"       # Table extraction
tabula-py = ">=2.9.0"         # Alternative table extraction

# Vision models
transformers = ">=4.40.0"     # For BLIP-2, LLaVA
openai = ">=1.0.0"            # For GPT-4V

# Image embedding
sentence-transformers = ">=3.0.0"  # Already installed, has CLIP support

# Equation OCR (optional)
nougat-ocr = ">=0.1.0"        # Open-source equation OCR
```

## Architecture

```
PDF
 │
 ├─► Text Extraction (existing) ─► Text Chunks ─► Text Embeddings
 │
 ├─► Figure Extraction ─► Images ─► Vision Description ─► CLIP Embeddings
 │
 ├─► Table Extraction ─► Structured Data ─► Markdown ─► Text Embeddings
 │
 └─► Equation Extraction ─► LaTeX ─► Description ─► Text Embeddings
                                      │
                                      ▼
                               Unified Index
```

## References

- [PyMuPDF Image Extraction](https://pymupdf.readthedocs.io/en/latest/recipes-images.html)
- [Camelot-py](https://camelot-py.readthedocs.io/)
- [CLIP (OpenAI)](https://github.com/openai/CLIP)
- [Nougat (Meta)](https://github.com/facebookresearch/nougat)
- [BLIP-2](https://huggingface.co/docs/transformers/model_doc/blip-2)
