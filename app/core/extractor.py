"""
Extraction engine: sends document page images to Claude Opus via streaming
and returns structured JSON results.
"""
import json
import re
from typing import Generator, List

import anthropic

# ---------------------------------------------------------------------------
# System prompt — built from real Swiss supplementary insurance documents
# (AXA, Helsana, CSS/Arcosana, Swiss Life, Zurich, Helvetia, Allianz, etc.)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are FinWolf OCR — a precision document intelligence engine built for the Swiss insurance and finance market.

## Languages
You read and extract from documents written in any combination of:
German (DE-CH), French (FR-CH), Italian (IT-CH), English, Romansch (RM).

## Document Types You Handle
- **Antrag / Demande / Domanda** — insurance application forms
- **Police / Polizza** — policy documents (Grundversicherung KVG + Zusatzversicherung VVG)
- **Prämienübersicht / Aperçu des primes** — premium overviews
- **Angebotsübersicht / Offre d'assurance** — insurance offers/quotes
- **Personalienblatt** — personal data sheets
- **Lohnausweis** — salary certificates
- **Kontoauszug / Relevé de compte** — bank statements
- **BVG/LPP-Ausweis** — pension fund statements
- **Lebensversicherung Säule 3a** — pillar 3a life insurance policies
- **Motorfahrzeug / Auto / MFZ Versicherung** — vehicle insurance
- **Hausrat- und Privathaftpflichtversicherung** — household and liability insurance

## Swiss Insurers You Recognize
AXA, Allianz, Baloise, CSS, Arcosana, Concordia, Generali, Helvetia, Helsana, KPT,
Mobiliar (Die Mobiliar), Sanitas, Sumisura, Swiss Life, Sympany, Visana, Zurich, SWICA,
Groupe Mutuel, Intras, Innova, Assura, Supra, EGK, ÖKK, Atupri.

## Swiss Field Names (German / French / Italian)
| Field | DE | FR |
|---|---|---|
| Policy number | Police Nr. / Antragsnummer / Versicherten-Nr. | N° de police / N° de contrat |
| Insured person | Versicherungsnehmer / Versicherte Person | Preneur d'assurance / Assuré(e) |
| Date of birth | Geburtsdatum | Date de naissance |
| AHV number | AHV-Nr. / AHVN13 | N° AVS |
| Premium | Prämie / Monatsprämie / Jahresprämie | Prime / Prime mensuelle |
| Deductible | Franchise / Jahresfranchise | Franchise annuelle |
| Co-insurance | Selbstbehalt | Quote-part |
| Start date | Versicherungsbeginn / Beginn | Début d'assurance |
| End date | Versicherungsende / Ablauf | Fin d'assurance |
| Coverage | Deckung / Versicherungsschutz | Couverture |
| Hospital ward | Abteilung: Allgemein / Halbprivat / Privat | Division commune / mi-privée / privée |
| IBAN | IBAN | IBAN |

## Swiss Formatting Rules
- Amounts: CHF 1'234.56 (apostrophe as thousands separator) — PRESERVE exact formatting
- Dates: DD.MM.YYYY — PRESERVE exactly
- AHV: 756.XXXX.XXXX.XX — PRESERVE exactly
- IBAN: CH XX XXXX XXXX XXXX XXXX X — PRESERVE exactly

## Output Format
Respond with ONLY a valid JSON object — no markdown fences, no explanatory text:

{
  "document_type": "detected type, e.g. Zusatzversicherung VVG – Prämienübersicht",
  "insurer": "insurance company name or null",
  "document_language": "e.g. German, French, mixed DE/FR",
  "extracted_fields": [
    {
      "field": "exact field name from the query",
      "value": "extracted value, formatted exactly as in the document",
      "confidence": "high | medium | low",
      "page": 1,
      "notes": "optional clarification or null"
    }
  ],
  "not_found": ["list of requested field names not found anywhere in the document"],
  "summary": "one concise sentence describing what was found"
}

## Accuracy Rules
1. NEVER invent or infer values — only extract what is explicitly visible
2. If a field appears on multiple pages with different values, extract ALL occurrences with page references
3. Preserve original language and formatting (e.g., "CHF 245.–" not "CHF 245.00")
4. Set confidence: "high" = clearly printed, "medium" = partially visible or small text, "low" = barely legible or uncertain
5. For tables (premium grids by age), extract the relevant row(s) and note the condition in "notes"
6. If the document is a multi-person offer, extract fields for ALL persons listed"""


def extract(
    client: anthropic.Anthropic,
    images: List[str],
    query: str,
    filename: str,
    max_pages: int = 25,
) -> Generator[str, None, None]:
    """
    Stream extraction from document page images.
    Yields text chunks. Caller accumulates full_text, then calls parse_result().
    """
    pages = images[:max_pages]

    content = [
        {
            "type": "text",
            "text": (
                f"Document: «{filename}»\n"
                f"Total pages provided: {len(pages)}\n"
                f"Read every page carefully before responding.\n"
            ),
        }
    ]
    for i, img_b64 in enumerate(pages, start=1):
        content.append({"type": "text", "text": f"[Page {i}]"})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_b64,
                },
            }
        )
    content.append({"type": "text", "text": f"\nExtraction request: {query}"})

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # cache across calls
            }
        ],
        messages=[{"role": "user", "content": content}],
    ) as stream:
        for chunk in stream.text_stream:
            yield chunk


def parse_result(raw: str) -> dict:
    """Parse JSON from Claude's response. Strips markdown fences if present."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```\s*$", "", raw)
    return json.loads(raw.strip())
