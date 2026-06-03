"""
Extraction engine: sends document page images to Google Gemini 2.0 Flash
and returns structured JSON results.
"""
import base64
import json
import re
from typing import Generator, List

import google.generativeai as genai

# ---------------------------------------------------------------------------
# System prompt — tuned for Swiss insurance & finance documents
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are FinWolf OCR — a precision document intelligence engine built for the Swiss insurance and finance market.

## Languages
You read and extract from documents written in any combination of:
German (DE-CH), French (FR-CH), Italian (IT-CH), English, Romansch (RM).

## Document Types You Handle
- Antrag / Demande — insurance application forms
- Police / Polizza — policy documents (Grundversicherung KVG + Zusatzversicherung VVG)
- Prämienübersicht / Aperçu des primes — premium overviews
- Angebotsübersicht / Offre d'assurance — insurance offers and quotes
- Personalienblatt — personal data sheets
- Lohnausweis — salary certificates
- Kontoauszug / Relevé de compte — bank statements
- BVG/LPP-Ausweis — pension fund statements
- Lebensversicherung Säule 3a — pillar 3a life insurance
- Motorfahrzeug / MFZ Versicherung — vehicle insurance
- Hausrat- und Privathaftpflichtversicherung — household and liability insurance

## Swiss Insurers You Recognize
AXA, Allianz, Baloise, CSS, Arcosana, Concordia, Generali, Helvetia, Helsana, KPT,
Mobiliar, Sanitas, Sumisura, Swiss Life, Sympany, Visana, Zurich, SWICA,
Groupe Mutuel, Intras, Innova, Assura, Supra, EGK, ÖKK, Atupri.

## Key Swiss Field Names
- Policy number: Police Nr. / Antragsnummer / Versicherten-Nr. / N° de police
- Insured person: Versicherungsnehmer / Versicherte Person / Assuré(e)
- Date of birth: Geburtsdatum / Date de naissance
- AHV number: AHV-Nr. / AHVN13 / N° AVS (format: 756.XXXX.XXXX.XX)
- Premium: Prämie / Monatsprämie / Jahresprämie / Prime mensuelle
- Deductible: Franchise / Jahresfranchise / Franchise annuelle
- Co-insurance: Selbstbehalt / Quote-part
- Start date: Versicherungsbeginn / Beginn / Début d'assurance
- Coverage: Deckung / Versicherungsschutz / Couverture
- Hospital: Allgemein / Halbprivat / Privat / Division commune / mi-privée / privée

## Swiss Formatting — PRESERVE EXACTLY
- Amounts: CHF 1'234.56 or CHF 245.–
- Dates: DD.MM.YYYY
- AHV: 756.XXXX.XXXX.XX
- IBAN: CH XX XXXX XXXX XXXX XXXX X

## Output Format
Respond with ONLY a valid JSON object — no markdown, no extra text:

{
  "document_type": "detected type e.g. Zusatzversicherung VVG – Prämienübersicht",
  "insurer": "insurance company name or null",
  "document_language": "e.g. German, French, mixed DE/FR",
  "extracted_fields": [
    {
      "field": "exact field name from the query",
      "value": "extracted value exactly as in the document",
      "confidence": "high | medium | low",
      "page": 1,
      "notes": "optional clarification or null"
    }
  ],
  "not_found": ["list of field names not found in the document"],
  "summary": "one concise sentence describing what was found"
}

## Accuracy Rules
1. NEVER invent or infer values — only extract what is explicitly visible
2. Preserve original language and formatting exactly
3. high = clearly printed, medium = partially visible, low = barely legible
4. For tables with multiple rows, extract all relevant rows with notes
5. If document has multiple persons, extract fields for ALL persons"""


def _init(api_key: str) -> genai.GenerativeModel:
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=SYSTEM_PROMPT,
    )


def extract(
    api_key: str,
    images: List[str],
    query: str,
    filename: str,
    max_pages: int = 25,
) -> Generator[str, None, None]:
    """
    Stream extraction from document page images via Gemini 2.0 Flash.
    Yields text chunks. Caller accumulates full text, then calls parse_result().
    """
    model = _init(api_key)
    pages = images[:max_pages]

    content: list = [
        f"Document: «{filename}» — {len(pages)} page(s). Read every page carefully.\n"
    ]
    for i, img_b64 in enumerate(pages, start=1):
        content.append(f"[Page {i}]")
        content.append(
            {"mime_type": "image/jpeg", "data": base64.b64decode(img_b64)}
        )
    content.append(f"\nExtraction request: {query}")

    response = model.generate_content(
        content,
        stream=True,
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=2048,
        ),
        safety_settings={
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        },
    )

    for chunk in response:
        if chunk.text:
            yield chunk.text


def parse_result(raw: str) -> dict:
    """Parse JSON from model response, stripping any markdown fences."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```\s*$", "", raw)
    return json.loads(raw.strip())
