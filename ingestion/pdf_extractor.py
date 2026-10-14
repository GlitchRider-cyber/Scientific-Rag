import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str) -> dict:
    """Extract text from each page of a PDF. Returns dict with metadata."""
    doc = fitz.open(pdf_path)
    pages = []
    full_text = []

    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"page": i + 1, "text": text.strip()})
            full_text.append(text.strip())

    doc.close()

    return {
        "source": pdf_path,
        "total_pages": len(pages),
        "pages": pages,
        "full_text": "\n\n".join(full_text),
    }
