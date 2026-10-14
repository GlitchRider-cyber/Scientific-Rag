import easyocr

# Lazy-load reader to avoid slow startup
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def extract_text_from_image(image_path: str) -> dict:
    """Run OCR on an image and return extracted text."""
    reader = _get_reader()
    results = reader.readtext(image_path, detail=0)  # detail=0 returns text only
    text = " ".join(results)

    return {
        "source": image_path,
        "text": text.strip(),
    }
