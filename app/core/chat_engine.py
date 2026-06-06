"""
Chat engine — streams answers about the uploaded document via Ollama.
Falls back gracefully if Ollama is not running.
"""
from __future__ import annotations
import json
import requests
from typing import Generator

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
DEFAULT_MODEL   = "qwen2.5:7b"

SYSTEM_TEMPLATE = """You are FinWolf OCR Assistant — an expert in Swiss insurance and finance documents.
You have been given the following document to analyze:

--- DOCUMENT ---
{doc_text}
--- END ---

Rules:
- Answer only from the document content above
- If the answer is not in the document, say so clearly
- Preserve Swiss formatting (CHF, dates DD.MM.YYYY, AHV numbers)
- Cite the page number when possible (e.g. "Found on page 2")
- Respond in the same language as the user's question
- Be concise and precise"""


def is_available() -> tuple[bool, str]:
    """Check if Ollama is running and return (available, model_name)."""
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        for preferred in [DEFAULT_MODEL, "qwen2.5:14b", "qwen2.5:32b"]:
            if any(preferred in m for m in models):
                return True, preferred
        if models:
            return True, models[0]
        return False, ""
    except Exception:
        return False, ""


def chat(doc_text: str, history: list[dict], question: str, model: str = DEFAULT_MODEL) -> Generator[str, None, None]:
    """Stream a chat response. Yields text chunks."""
    system = SYSTEM_TEMPLATE.format(doc_text=doc_text[:8000])
    messages = [{"role": "system", "content": system}]
    for msg in history[-10:]:  # keep last 10 turns for context
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={"model": model, "messages": messages, "stream": True},
            stream=True,
            timeout=300,
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
                if data.get("done"):
                    break
    except requests.exceptions.ConnectionError:
        yield "⚠️ Ollama ist nicht erreichbar. Bitte starten Sie den lokalen Modell-Service."
    except Exception as e:
        yield f"⚠️ Fehler: {e}"


def quick_summary(doc_text: str, model: str = DEFAULT_MODEL) -> str:
    """Return a one-sentence document summary (non-streaming)."""
    try:
        r = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Summarize the document in one sentence in German. Be concise."},
                    {"role": "user", "content": f"Document:\n{doc_text[:3000]}"},
                ],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 100},
            },
            timeout=120,
        )
        return r.json().get("message", {}).get("content", "").strip()
    except Exception:
        return ""
