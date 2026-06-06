"""
Stage 2 — Extract text from 199 PDFs using PyMuPDF direct text extraction.
Works on Intel Mac with no OCR dependencies.
Falls back to basic image-based extraction for scanned pages.
"""
import sys
from pathlib import Path

from tqdm import tqdm
import fitz  # PyMuPDF

ROOT     = Path(__file__).parent.parent
DOCS_DIR = Path("/Users/kodelbahram/Downloads/insurance supplementary docs")
OUT_DIR  = ROOT / "data" / "raw_text"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CHARS_PER_PAGE = 50  # pages with fewer chars are likely scanned


def extract_pdf_text(pdf_path: Path) -> tuple[str, int, int]:
    doc = fitz.open(str(pdf_path))
    all_text = []
    text_pages = 0
    scanned_pages = 0

    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if len(text) >= MIN_CHARS_PER_PAGE:
            all_text.append(f"--- Page {i+1} ---\n{text}")
            text_pages += 1
        else:
            # Scanned page — extract whatever is there, mark it
            all_text.append(f"--- Page {i+1} (scanned/image) ---\n{text or '[no text detected]'}")
            scanned_pages += 1

    doc.close()
    return "\n\n".join(all_text), text_pages, scanned_pages


def main():
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DOCS_DIR}")
        sys.exit(1)

    print(f"Found {len(pdfs)} PDFs. Extracting text with PyMuPDF…")

    skipped = scanned_count = 0
    for pdf in tqdm(pdfs, desc="Extracting", unit="doc"):
        out_file = OUT_DIR / (pdf.stem + ".txt")
        if out_file.exists():
            skipped += 1
            continue
        try:
            text, text_pages, scanned = extract_pdf_text(pdf)
            out_file.write_text(text, encoding="utf-8")
            if scanned > 0:
                scanned_count += 1
        except Exception as e:
            print(f"\n  ⚠  Skipping {pdf.name}: {e}")

    done = len(list(OUT_DIR.glob("*.txt")))
    print(f"\nDone. {done} text files saved to data/raw_text/")
    print(f"  {skipped} already existed (skipped)")
    if scanned_count:
        print(f"  ⚠  {scanned_count} PDFs had scanned/image pages (text may be sparse)")
    print("\nNext step: python3 scripts/generate_training_data.py")


if __name__ == "__main__":
    main()
