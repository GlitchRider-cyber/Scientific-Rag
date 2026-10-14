import os
from ingestion.pdf_extractor import extract_text_from_pdf
from ingestion.ocr_processor import extract_text_from_image
from ingestion.chunker import semantic_chunk

# Supported file types
PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
TEXT_EXTS = {".txt", ".md"}


def extract_text(file_path: str) -> dict:
    """Extract text from any supported file type."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext in PDF_EXTS:
        result = extract_text_from_pdf(file_path)
        return {"source": file_path, "text": result["full_text"]}

    elif ext in IMAGE_EXTS:
        return extract_text_from_image(file_path)

    elif ext in TEXT_EXTS:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return {"source": file_path, "text": text.strip()}

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def ingest_file(file_path: str) -> list[dict]:
    """Full pipeline: extract text -> semantic chunk."""
    extracted = extract_text(file_path)
    chunks = semantic_chunk(extracted["text"], source=extracted["source"])
    return chunks


def ingest_multiple(file_paths: list[str]) -> list[dict]:
    """Ingest multiple files and return combined chunks."""
    all_chunks = []
    for path in file_paths:
        chunks = ingest_file(path)
        all_chunks.extend(chunks)
    return all_chunks
