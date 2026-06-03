"""
Document processor: converts any supported file to a list of base64 PNG page images
for the Claude Vision API.
Supports: PDF (native + scanned), PNG, JPG, TIFF, BMP, WEBP
"""
import base64
import io
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from PIL import Image

MAX_DIMENSION = 2048   # Max width or height — fits Claude's optimal vision range
RENDER_DPI    = 180    # Balance of clarity vs token cost (72 DPI = screen, 300 = print)


def _pil_to_b64(img: Image.Image) -> str:
    """Resize if oversized, then return base64-encoded PNG string."""
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if img.width > MAX_DIMENSION or img.height > MAX_DIMENSION:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def _process_pdf(file_bytes: bytes) -> List[str]:
    """Render every PDF page to a high-res PNG and return base64 strings."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages: List[str] = []
    scale = RENDER_DPI / 72.0
    mat = fitz.Matrix(scale, scale)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pages.append(_pil_to_b64(img))
    doc.close()
    return pages


def _process_image(file_bytes: bytes) -> List[str]:
    """Convert a single image file to one base64 PNG."""
    img = Image.open(io.BytesIO(file_bytes))
    return [_pil_to_b64(img)]


def process_document(file_bytes: bytes, filename: str) -> List[str]:
    """
    Return a list of base64-encoded PNG images, one per page/frame.
    Raises ValueError for unsupported formats.
    """
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _process_pdf(file_bytes)
    elif ext in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}:
        return _process_image(file_bytes)
    else:
        raise ValueError(
            f"Unsupported format: '{ext}'. "
            "Supported: PDF, PNG, JPG, JPEG, TIFF, TIF, BMP, WEBP"
        )
