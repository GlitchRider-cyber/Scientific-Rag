import os
from pathlib import Path


def get_filename(path: str) -> str:
    """Extract filename from a path."""
    return os.path.basename(path)


def allowed_file(filename: str) -> bool:
    """Check if file type is supported."""
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".txt", ".md"}
    ext = Path(filename).suffix.lower()
    return ext in allowed
