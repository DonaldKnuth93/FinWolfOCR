"""
Stage 5 — Test the fine-tuned model on 5 sample documents
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT        = Path(__file__).parent.parent
TEXT_DIR    = ROOT / "data" / "raw_text"
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

TEST_QUERY = "Police Nr., Versicherungsgesellschaft, Monatsprämie, Franchise, Versicherungsnehmer, Geburtsdatum, Vertragsbeginn"


def load_model():
    if not ADAPTER_DIR.exists():
        print(f"❌  Adapter not found at {ADAPTER_DIR}. Run finetune.py first.")
        sys.exit(1)

    print(f"Loading base model {BASE_MODEL}…")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    model.eval()
    print("✅  Model loaded\n")
    return model, tokenizer


def run_inference(model, tokenizer, doc_text: str) -> dict | None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Dokument:\n{doc_text[:4000]}\n\nExtrahiere: {TEST_QUERY}"},
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

    raw = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def main():
    txt_files = sorted(TEXT_DIR.glob("*.txt"))
    if not txt_files:
        print(f"❌  No text files in {TEXT_DIR}. Run extract_text.py first.")
        sys.exit(1)

    samples = txt_files[:5]
    model, tokenizer = load_model()

    total_found = total_high = total_valid = 0

    for i, txt_path in enumerate(samples, 1):
        print(f"[{i}/5] Testing on {txt_path.name}…")
        doc_text = txt_path.read_text(encoding="utf-8")
        result = run_inference(model, tokenizer, doc_text)

        if result is None:
            print("  ❌  Failed to parse JSON output\n")
            continue

        fields = result.get("extracted_fields", [])
        high   = sum(1 for f in fields if f.get("confidence") == "high")
        total_found += len(fields)
        total_high  += high
        total_valid += 1

        print(f"  Document type : {result.get('document_type','?')}")
        print(f"  Insurer       : {result.get('insurer','?')}")
        print(f"  Fields found  : {len(fields)}  (high:{high})")
        print(f"  Not found     : {len(result.get('not_found',[]))}")
        print(f"  Summary       : {result.get('summary','')[:80]}…")
        print()

    print("=" * 50)
    print(f"Results: {total_valid}/5 valid JSON responses")
    if total_valid:
        print(f"Avg fields  : {total_found/total_valid:.1f} per doc")
        print(f"High conf   : {round(total_high/max(total_found,1)*100)}%")

    if total_valid >= 4:
        print("\n✅  Model looks good — proceed to Stage 6 (integrate into app)")
    else:
        print("\n⚠️   Some outputs failed — check training data quality")


if __name__ == "__main__":
    main()
