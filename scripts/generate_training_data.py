"""
Stage 3 — Generate training data from extracted text using Qwen2.5:32b via Ollama
Output: data/training/finwolf_train.jsonl + finwolf_valid.jsonl
"""
from __future__ import annotations
import json
import random
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

ROOT     = Path(__file__).parent.parent
TEXT_DIR = ROOT / "data" / "raw_text"
OUT_DIR  = ROOT / "data" / "training"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL   = "http://localhost:11434/api/generate"
TEACHER_MODEL = "qwen2.5:7b"
VALID_SPLIT  = 0.1  # 10% validation

QUERIES = [
    ("contract",    "Police Nr., Antragsnummer, Versicherungsgesellschaft, Versicherungsart (KVG/VVG), Versicherungsbeginn, Versicherungsende"),
    ("personal",    "Name, Vorname, Geburtsdatum, Adresse, AHV-Nummer, Versicherten-Nummer"),
    ("premiums",    "Monatsprämie, Jahresprämie, Franchise (Jahresfranchise), Selbstbehalt, Zahlungsrhythmus"),
    ("supplemental","Gewählte Produkte und Tarife, eingeschlossene Leistungen, Spitalabteilung (Allgemein/Halbprivat/Privat), Wartezeiten"),
    ("vehicle",     "Kennzeichen, Fahrzeugtyp, Fahrleistung km/Jahr, Haftpflichtdeckung, Kasko (Teil/Voll), Prämie"),
    ("all",         "Extrahiere alle relevanten Felder: Police/Vertragsnummer, Versicherungsnehmer, Geburtsdatum, Versicherungsgesellschaft, Versicherungsart, alle Prämienbeträge, Franchise, Selbstbehalt, Vertragsbeginn, Deckungsumfang"),
]

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


def build_prompt(doc_text: str, query: str) -> str:
    truncated = doc_text[:3000]  # keep context short for CPU speed
    return f"Dokument:\n{truncated}\n\nExtrahiere: {query}"


def call_ollama(prompt: str, retries: int = 3) -> str | None:
    payload = {
        "model": TEACHER_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1024},
    }
    for attempt in range(retries):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=600)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                print(f"\n  ⚠  Ollama error: {e}")
    return None


def parse_json_response(raw: str) -> dict | None:
    raw = raw.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(TEACHER_MODEL in m for m in models):
            print(f"❌  Teacher model '{TEACHER_MODEL}' not found in Ollama.")
            print(f"    Run: ollama pull {TEACHER_MODEL}")
            sys.exit(1)
        print(f"✅  Ollama running · teacher model {TEACHER_MODEL} ready")
    except Exception:
        print("❌  Ollama is not running. Start it with: ollama serve")
        sys.exit(1)


def main():
    check_ollama()

    txt_files = sorted(TEXT_DIR.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {TEXT_DIR}. Run extract_text.py first.")
        sys.exit(1)

    print(f"Found {len(txt_files)} text files · {len(QUERIES)} queries each = up to {len(txt_files)*len(QUERIES)} examples")

    examples = []
    train_file = OUT_DIR / "finwolf_train.jsonl"
    valid_file = OUT_DIR / "finwolf_valid.jsonl"

    # Resume support — skip already-processed docs
    done_stems = set()
    for f in [train_file, valid_file]:
        if f.exists():
            for line in f.read_text().splitlines():
                try:
                    obj = json.loads(line)
                    done_stems.add(obj.get("_source", ""))
                except Exception:
                    pass

    with open(train_file, "a", encoding="utf-8") as ftrain, \
         open(valid_file, "a", encoding="utf-8") as fvalid:

        for txt_path in tqdm(txt_files, desc="Documents", unit="doc"):
            doc_text = txt_path.read_text(encoding="utf-8")

            for query_name, query_fields in tqdm(QUERIES, desc=f"  {txt_path.stem[:30]}", leave=False):
                key = f"{txt_path.stem}::{query_name}"
                if key in done_stems:
                    continue

                raw = call_ollama(build_prompt(doc_text, query_fields))
                if not raw:
                    continue

                parsed = parse_json_response(raw)
                if not parsed:
                    continue

                example = {
                    "messages": [
                        {"role": "system",    "content": SYSTEM_PROMPT},
                        {"role": "user",      "content": build_prompt(doc_text, query_fields)},
                        {"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)},
                    ],
                    "_source": key,
                }

                if random.random() < VALID_SPLIT:
                    fvalid.write(json.dumps(example, ensure_ascii=False) + "\n")
                else:
                    ftrain.write(json.dumps(example, ensure_ascii=False) + "\n")

    train_count = sum(1 for _ in open(train_file))
    valid_count = sum(1 for _ in open(valid_file))
    print(f"\nDone. {train_count} training · {valid_count} validation examples")
    print(f"  → {train_file}")
    print(f"  → {valid_file}")


if __name__ == "__main__":
    main()
