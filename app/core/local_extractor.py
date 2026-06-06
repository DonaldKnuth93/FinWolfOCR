"""
Stage 6 — Local model extractor (drop-in replacement for extractor.py)
Uses fine-tuned Qwen2.5-7B via PyTorch + PEFT instead of Gemini API.
Intel Mac compatible.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Generator

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT        = Path(__file__).parent.parent.parent
ADAPTER_DIR = ROOT / "models" / "finwolf-7b-adapter"
BASE_MODEL  = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = """Du bist ein Spezialist für Schweizer Versicherungs- und Finanzdokumente.
Extrahiere die angeforderten Felder aus dem Dokumenttext und antworte ausschliesslich mit einem JSON-Objekt.

JSON-Format:
{
  "document_type": "string",
  "insurer": "string or null",
  "document_language": "DE|FR|IT|EN",
  "extracted_fields": [
    {"field": "Feldname", "value": "Wert", "confidence": "high|medium|low", "page": 1, "notes": "optional"}
  ],
  "not_found": ["Feldname1", "Feldname2"],
  "summary": "Kurze Zusammenfassung des Dokuments"
}"""

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            str(ADAPTER_DIR) if ADAPTER_DIR.exists() else BASE_MODEL,
            trust_remote_code=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if ADAPTER_DIR.exists():
            _model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
        else:
            _model = base
        _model.eval()
    return _model, _tokenizer


def extract_with_ocr(ocr_text: str, query: str, filename: str) -> Generator[str, None, None]:
    """Entry point that accepts pre-extracted OCR text. Preferred for accuracy."""
    model, tokenizer = _load_model()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Dokument: {filename}\n\n{ocr_text[:6000]}\n\nExtrahiere: {query}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    yield result


def extract(pages: list, query: str, filename: str, max_pages: int = 10) -> Generator[str, None, None]:
    """Compatibility shim — pages are image objects, we pass a placeholder."""
    placeholder = "\n".join(f"[Seite {i+1}]" for i in range(min(len(pages), max_pages)))
    yield from extract_with_ocr(placeholder, query, filename)


def parse_result(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "document_type": "Unbekannt",
            "insurer": None,
            "document_language": "DE",
            "extracted_fields": [],
            "not_found": [],
            "summary": f"JSON-Parsing fehlgeschlagen. Rohantwort: {raw[:200]}",
        }
