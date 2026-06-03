"""Export extraction results to CSV or JSON."""
import io
import json
from typing import List, Tuple

import pandas as pd


def to_dataframe(result: dict, filename: str) -> pd.DataFrame:
    rows = []
    meta = {
        "Source File":    filename,
        "Document Type":  result.get("document_type", ""),
        "Insurer":        result.get("insurer", ""),
        "Language":       result.get("document_language", ""),
    }
    for f in result.get("extracted_fields", []):
        rows.append({
            **meta,
            "Field":      f.get("field", ""),
            "Value":      f.get("value", ""),
            "Confidence": f.get("confidence", ""),
            "Page":       f.get("page", ""),
            "Notes":      f.get("notes") or "",
            "Status":     "found",
        })
    for field in result.get("not_found", []):
        rows.append({
            **meta,
            "Field":      field,
            "Value":      "NOT FOUND",
            "Confidence": "",
            "Page":       "",
            "Notes":      "",
            "Status":     "not_found",
        })
    return pd.DataFrame(rows)


def export_csv(results: List[Tuple[dict, str]]) -> bytes:
    if not results:
        return b""
    df = pd.concat([to_dataframe(r, f) for r, f in results], ignore_index=True)
    df = df.drop(columns=["Status"])
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")  # utf-8-sig = Excel-compatible BOM
    return buf.getvalue()


def export_json(results: List[Tuple[dict, str]]) -> bytes:
    out = [{"filename": f, "extraction": r} for r, f in results]
    return json.dumps(out, ensure_ascii=False, indent=2).encode("utf-8")
