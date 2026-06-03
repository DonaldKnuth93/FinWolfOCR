"""
FinWolf OCR — Swiss Insurance & Finance Document Intelligence
ClickUp-inspired dark UI with purple wolf branding.
"""
import base64
import os
import sys
from pathlib import Path

import anthropic
import streamlit as st
from dotenv import load_dotenv

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.core.document_processor import process_document
from app.core.extractor import extract, parse_result
from app.core.exporter import export_csv, export_json, to_dataframe

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinWolf OCR",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── helpers ───────────────────────────────────────────────────────────────────
def _logo_b64() -> str | None:
    path = ROOT / "assets" / "logo.png"
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return None


def _badge(confidence: str) -> str:
    conf = (confidence or "").lower()
    if conf == "high":
        return '<span class="badge badge-high">● High</span>'
    elif conf == "medium":
        return '<span class="badge badge-medium">● Medium</span>'
    elif conf == "low":
        return '<span class="badge badge-low">● Low</span>'
    return ""


def _render_results_html(result: dict, filename: str) -> str:
    doc_type  = result.get("document_type", "Unknown")
    insurer   = result.get("insurer") or ""
    lang      = result.get("document_language", "")
    fields    = result.get("extracted_fields", [])
    not_found = result.get("not_found", [])
    summary   = result.get("summary", "")

    insurer_chip = f'<span class="chip chip-insurer">{insurer}</span>' if insurer else ""
    lang_chip    = f'<span class="chip chip-lang">{lang}</span>' if lang else ""

    rows_html = ""
    for f in fields:
        value = f.get("value", "")
        page  = f.get("page", "")
        notes = f.get("notes") or ""
        rows_html += f"""
        <tr>
          <td class="td-field">{f.get('field','')}</td>
          <td class="td-value">{value}</td>
          <td>{_badge(f.get('confidence',''))}</td>
          <td class="td-page">{page}</td>
          <td class="td-notes">{notes}</td>
        </tr>"""

    for nf in not_found:
        rows_html += f"""
        <tr class="tr-missing">
          <td class="td-field">{nf}</td>
          <td class="td-value td-missing">— not found —</td>
          <td><span class="badge badge-missing">N/A</span></td>
          <td class="td-page"></td>
          <td class="td-notes"></td>
        </tr>"""

    summary_block = f'<div class="fw-summary">💡 {summary}</div>' if summary else ""

    return f"""
<div class="fw-result-card">
  <div class="fw-doc-header">
    <div class="fw-doc-left">
      <span class="fw-doc-icon">📄</span>
      <span class="fw-doc-name">{filename}</span>
    </div>
    <div class="fw-doc-right">
      {insurer_chip}{lang_chip}
      <span class="fw-doc-type">{doc_type}</span>
    </div>
  </div>
  <div class="fw-table-wrap">
    <table class="fw-table">
      <thead>
        <tr>
          <th>FIELD</th>
          <th>VALUE</th>
          <th>CONFIDENCE</th>
          <th>PAGE</th>
          <th>NOTES</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
  {summary_block}
</div>"""


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, .stApp {
    background: #09080F !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color: #E2E8F0 !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; height: 0; }
.block-container {
    max-width: 1280px !important;
    padding: 0 2.5rem 5rem !important;
    margin-top: 0 !important;
}
[data-testid="stDecoration"] { display: none; }

/* ── Header ── */
.fw-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 28px 0 28px;
    border-bottom: 1px solid rgba(124,58,237,0.18);
    margin-bottom: 36px;
}
.fw-logo-img { height: 52px; width: auto; }
.fw-logo-mono {
    width: 52px; height: 52px;
    background: linear-gradient(135deg,#7C3AED,#4C1D95);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem; font-weight: 900; color: #fff;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4);
    flex-shrink: 0;
}
.fw-title-block { flex: 1; }
.fw-title {
    font-size: 1.75rem; font-weight: 800;
    color: #F1F5F9; letter-spacing: -0.5px; line-height: 1;
}
.fw-subtitle {
    font-size: 0.72rem; font-weight: 600;
    color: #7C3AED; letter-spacing: 2px;
    text-transform: uppercase; margin-top: 4px;
}
.fw-env-badge {
    background: rgba(124,58,237,0.12);
    border: 1px solid rgba(124,58,237,0.28);
    color: #A78BFA;
    padding: 5px 14px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.5px;
}

/* ── Section labels ── */
.fw-section-label {
    font-size: 0.65rem; font-weight: 700;
    color: #4B5563; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 12px;
}

/* ── Cards ── */
.fw-card {
    background: #100F1C;
    border: 1px solid rgba(124,58,237,0.14);
    border-radius: 16px; padding: 22px;
    margin-bottom: 16px;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.fw-card:hover {
    border-color: rgba(124,58,237,0.32);
    box-shadow: 0 0 0 1px rgba(124,58,237,0.08), 0 8px 32px rgba(0,0,0,0.4);
}

/* ── File list ── */
.fw-file-list { display: flex; flex-direction: column; gap: 8px; margin-top: 12px; }
.fw-file-item {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 14px;
    background: rgba(124,58,237,0.06);
    border: 1px solid rgba(124,58,237,0.12);
    border-radius: 10px;
}
.fw-file-icon { font-size: 1rem; flex-shrink: 0; }
.fw-file-name {
    flex: 1; font-size: 0.82rem; font-weight: 500; color: #CBD5E1;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.fw-file-size { font-size: 0.72rem; color: #4B5563; white-space: nowrap; }

/* ── Template pills ── */
.fw-pills { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 14px; }
.fw-pill {
    background: rgba(124,58,237,0.09);
    border: 1px solid rgba(124,58,237,0.22);
    color: #A78BFA; padding: 5px 13px;
    border-radius: 20px; font-size: 0.72rem; font-weight: 600;
    cursor: pointer; transition: all 0.15s; white-space: nowrap;
}
.fw-pill:hover {
    background: rgba(124,58,237,0.2);
    border-color: rgba(124,58,237,0.5); color: #C4B5FD;
}

/* ── Results ── */
.fw-result-card {
    background: #100F1C;
    border: 1px solid rgba(124,58,237,0.18);
    border-radius: 16px; overflow: hidden;
    margin-bottom: 20px;
}
.fw-doc-header {
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 10px;
    padding: 14px 20px;
    background: rgba(124,58,237,0.07);
    border-bottom: 1px solid rgba(124,58,237,0.12);
}
.fw-doc-left { display: flex; align-items: center; gap: 10px; }
.fw-doc-icon { font-size: 1.1rem; }
.fw-doc-name { font-size: 0.9rem; font-weight: 700; color: #F1F5F9; }
.fw-doc-right { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.fw-doc-type { font-size: 0.72rem; color: #7C3AED; font-weight: 600; }

.chip {
    padding: 3px 10px; border-radius: 20px;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.3px;
}
.chip-insurer { background: rgba(16,185,129,0.12); color: #10B981; }
.chip-lang    { background: rgba(59,130,246,0.12);  color: #60A5FA; }

/* ── Table ── */
.fw-table-wrap { overflow-x: auto; }
.fw-table {
    width: 100%; border-collapse: collapse;
    font-size: 0.82rem;
}
.fw-table thead th {
    padding: 9px 18px;
    font-size: 0.6rem; font-weight: 700;
    letter-spacing: 1.8px; text-transform: uppercase;
    color: #4B5563; text-align: left;
    border-bottom: 1px solid rgba(124,58,237,0.1);
    background: rgba(0,0,0,0.2);
    white-space: nowrap;
}
.fw-table tbody td {
    padding: 11px 18px;
    border-bottom: 1px solid rgba(255,255,255,0.035);
    vertical-align: middle;
}
.fw-table tbody tr:last-child td { border-bottom: none; }
.fw-table tbody tr:hover td { background: rgba(124,58,237,0.05); }

.td-field { color: #94A3B8; font-weight: 500; white-space: nowrap; }
.td-value {
    color: #F1F5F9; font-weight: 500;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 0.8rem;
}
.td-page  { color: #6B7280; text-align: center; font-size: 0.75rem; }
.td-notes { color: #6B7280; font-size: 0.75rem; font-style: italic; }
.td-missing { color: #374151 !important; font-style: italic; }
.tr-missing td { opacity: 0.55; }

/* Confidence badges */
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.66rem; font-weight: 700; white-space: nowrap;
}
.badge-high    { background: rgba(16,185,129,0.14); color: #10B981; }
.badge-medium  { background: rgba(245,158,11,0.14); color: #F59E0B; }
.badge-low     { background: rgba(239,68,68,0.14);  color: #EF4444; }
.badge-missing { background: rgba(75,85,99,0.2);    color: #4B5563; }

/* Summary */
.fw-summary {
    padding: 11px 20px;
    background: rgba(124,58,237,0.05);
    border-top: 1px solid rgba(124,58,237,0.1);
    font-size: 0.78rem; color: #6B7280; font-style: italic;
    line-height: 1.5;
}

/* ── Export row ── */
.fw-export-row {
    display: flex; align-items: center; gap: 12px;
    padding: 18px 0 6px;
}
.fw-export-label {
    font-size: 0.65rem; font-weight: 700;
    color: #4B5563; letter-spacing: 2px; text-transform: uppercase;
    margin-right: 4px;
}

/* ── Streamlit component overrides ── */

/* File uploader */
[data-testid="stFileUploader"] > div:first-child {
    background: #100F1C !important;
    border: 2px dashed rgba(124,58,237,0.3) !important;
    border-radius: 14px !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploader"] > div:first-child:hover {
    border-color: rgba(124,58,237,0.65) !important;
    background: rgba(124,58,237,0.04) !important;
}
[data-testid="stFileUploader"] label {
    color: #94A3B8 !important; font-size: 0.75rem !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
}

/* Text area */
[data-testid="stTextArea"] textarea {
    background: #0D0C18 !important;
    border: 1px solid rgba(124,58,237,0.25) !important;
    border-radius: 12px !important;
    color: #E2E8F0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    line-height: 1.6 !important;
    resize: vertical !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #7C3AED !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.18) !important;
    outline: none !important;
}
[data-testid="stTextArea"] label {
    color: #6B7280 !important; font-size: 0.65rem !important;
    font-weight: 700 !important; letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #0D0C18 !important;
    border: 1px solid rgba(124,58,237,0.25) !important;
    border-radius: 10px !important; color: #E2E8F0 !important;
}
[data-testid="stSelectbox"] label {
    color: #6B7280 !important; font-size: 0.65rem !important;
    font-weight: 700 !important; letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

/* Slider */
[data-testid="stSlider"] label {
    color: #6B7280 !important; font-size: 0.65rem !important;
    font-weight: 700 !important; letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

/* Primary button (Extract) */
.stButton > button {
    background: linear-gradient(135deg, #7C3AED 0%, #5B21B6 100%) !important;
    color: #FFFFFF !important; border: none !important;
    border-radius: 11px !important; font-weight: 700 !important;
    font-size: 0.9rem !important; letter-spacing: 0.3px !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: 0 4px 18px rgba(124,58,237,0.35) !important;
    transition: all 0.2s ease !important;
    height: 46px !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 28px rgba(124,58,237,0.55) !important;
    transform: translateY(-1px) !important;
    background: linear-gradient(135deg, #8B4CF6 0%, #6D28D9 100%) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
.stButton > button:disabled {
    background: rgba(75,85,99,0.3) !important;
    color: #4B5563 !important; box-shadow: none !important;
    transform: none !important;
}

/* Download buttons */
.stDownloadButton > button {
    background: rgba(124,58,237,0.09) !important;
    color: #A78BFA !important;
    border: 1px solid rgba(124,58,237,0.28) !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 0.82rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.15s !important;
}
.stDownloadButton > button:hover {
    background: rgba(124,58,237,0.2) !important;
    border-color: rgba(124,58,237,0.55) !important;
    color: #C4B5FD !important;
    box-shadow: 0 0 12px rgba(124,58,237,0.2) !important;
}

/* Alerts */
.stAlert { border-radius: 12px !important; border-left-width: 3px !important; }

/* Divider */
hr { border-color: rgba(124,58,237,0.12) !important; margin: 28px 0 !important; }

/* Progress bar */
[data-testid="stProgressBar"] > div { background: rgba(124,58,237,0.2) !important; }
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #7C3AED, #A78BFA) !important;
    border-radius: 4px !important;
}

/* Spinner */
.stSpinner > div { border-top-color: #7C3AED !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124,58,237,0.5); }
</style>
"""

# ── Template queries ───────────────────────────────────────────────────────────
TEMPLATES = {
    "🧾 Vertragsdaten":     "Police Nr., Antragsnummer, Versicherungsgesellschaft, Versicherungsart (KVG/VVG), Versicherungsbeginn, Versicherungsende",
    "👤 Personendaten":     "Name, Vorname, Geburtsdatum, Adresse, AHV-Nummer, Versicherten-Nummer",
    "💰 Prämien & Kosten":  "Monatsprämie, Jahresprämie, Franchise (Jahresfranchise), Selbstbehalt, Zahlungsrhythmus",
    "🏥 Zusatzversicherung":"Gewählte Produkte und Tarife, eingeschlossene Leistungen, Spitalabteilung (Allgemein/Halbprivat/Privat), Wartezeiten",
    "🚗 Fahrzeugversicherung":"Kennzeichen, Fahrzeugtyp, Fahrleistung km/Jahr, Haftpflichtdeckung, Kasko (Teil/Voll), Prämie",
    "📋 Alle Schlüsselfelder":"Extrahiere alle relevanten Felder: Police/Vertragsnummer, Versicherungsnehmer, Geburtsdatum, Versicherungsgesellschaft, Versicherungsart, alle Prämienbeträge, Franchise, Selbstbehalt, Vertragsbeginn, Deckungsumfang",
}

# ── Main app ───────────────────────────────────────────────────────────────────
@st.cache_resource
def get_client() -> anthropic.Anthropic:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        st.error("⚠️  ANTHROPIC_API_KEY not set. Copy .env.example → .env and add your key.")
        st.stop()
    return anthropic.Anthropic(api_key=key)


def build_header() -> str:
    logo_b64 = _logo_b64()
    if logo_b64:
        logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="fw-logo-img" alt="FinWolf">'
    else:
        logo_html = '<div class="fw-logo-mono">FW</div>'

    return f"""
<div class="fw-header">
  {logo_html}
  <div class="fw-title-block">
    <div class="fw-title">FinWolf OCR</div>
    <div class="fw-subtitle">Swiss Insurance &amp; Finance · Document Intelligence</div>
  </div>
  <div class="fw-env-badge">Claude Opus 4.6</div>
</div>"""


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(build_header(), unsafe_allow_html=True)

    client = get_client()

    # ── Session state ─────────────────────────────────────────────────────────
    if "results" not in st.session_state:
        st.session_state.results = []
    if "selected_template" not in st.session_state:
        st.session_state.selected_template = ""

    # ── Two-column input layout ───────────────────────────────────────────────
    col_upload, col_query = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="fw-section-label">Documents</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "UPLOAD FILES",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded:
            items = "".join(
                f'<div class="fw-file-item">'
                f'<span class="fw-file-icon">{"📄" if f.name.endswith(".pdf") else "🖼️"}</span>'
                f'<span class="fw-file-name">{f.name}</span>'
                f'<span class="fw-file-size">{f.size/1024:.0f} KB</span>'
                f'</div>'
                for f in uploaded
            )
            st.markdown(f'<div class="fw-file-list">{items}</div>', unsafe_allow_html=True)

    with col_query:
        st.markdown('<div class="fw-section-label">Extraction Query</div>', unsafe_allow_html=True)

        # Template pills
        pill_html = '<div class="fw-pills">' + "".join(
            f'<span class="fw-pill">{label}</span>' for label in TEMPLATES
        ) + "</div>"
        st.markdown(pill_html, unsafe_allow_html=True)

        # Dropdown for template selection
        template_choice = st.selectbox(
            "QUICK TEMPLATE",
            ["— or write your own query below —"] + list(TEMPLATES.keys()),
            label_visibility="visible",
        )
        default_query = (
            TEMPLATES[template_choice]
            if template_choice in TEMPLATES
            else st.session_state.selected_template
        )

        query = st.text_area(
            "QUERY",
            value=default_query,
            height=110,
            placeholder="e.g.  Police Nr., Monatsprämie, Franchise, Versicherungsnehmer, Beginn",
            label_visibility="visible",
        )

        max_pages = st.slider("MAX PAGES PER DOCUMENT", 5, 40, 20)

        run = st.button(
            "Extract →",
            type="primary",
            disabled=not (uploaded and query.strip()),
            use_container_width=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Extraction ────────────────────────────────────────────────────────────
    if run and uploaded and query.strip():
        st.session_state.results = []
        progress = st.progress(0, text="Starting extraction…")

        for idx, uf in enumerate(uploaded):
            pct = idx / len(uploaded)
            progress.progress(pct, text=f"Processing {uf.name}…")
            file_bytes = uf.read()

            try:
                with st.spinner(f"Rendering pages — {uf.name}"):
                    pages = process_document(file_bytes, uf.name)

                truncated = len(pages) > max_pages
                page_count = min(len(pages), max_pages)

                status_text = f"Extracting {page_count} page(s) from {uf.name}"
                if truncated:
                    status_text += f" (document has {len(pages)} pages, showing first {max_pages})"

                with st.spinner(status_text):
                    raw = ""
                    for chunk in extract(client, pages, query, uf.name, max_pages):
                        raw += chunk

                result = parse_result(raw)
                st.session_state.results.append((result, uf.name))

            except ValueError as e:
                st.error(f"**{uf.name}** — {e}")
            except Exception as e:
                st.error(f"**{uf.name}** — Extraction failed: {e}")

        progress.progress(1.0, text="Done!")

    # ── Results display (persists across reruns / download clicks) ────────────
    if st.session_state.results:
        st.markdown(
            '<div class="fw-section-label" style="margin-bottom:16px;">Results</div>',
            unsafe_allow_html=True,
        )

        for result, filename in st.session_state.results:
            st.markdown(_render_results_html(result, filename), unsafe_allow_html=True)

        # ── Export ──
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="fw-section-label">Export</div>', unsafe_allow_html=True)

        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            st.download_button(
                "⬇  Download CSV",
                data=export_csv(st.session_state.results),
                file_name="finwolf_extraction.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "⬇  Download JSON",
                data=export_json(st.session_state.results),
                file_name="finwolf_extraction.json",
                mime="application/json",
                use_container_width=True,
            )

        field_count = sum(
            len(r.get("extracted_fields", [])) for r, _ in st.session_state.results
        )
        st.caption(
            f"{len(st.session_state.results)} document(s) · "
            f"{field_count} field(s) extracted · "
            f"Powered by Claude Opus 4.6"
        )

    elif not uploaded:
        st.markdown(
            """
<div style="text-align:center;padding:60px 0;color:#2D2640;">
  <div style="font-size:3.5rem;margin-bottom:16px;">📂</div>
  <div style="font-size:1rem;font-weight:600;color:#4B5563;">
    Upload a document to get started
  </div>
  <div style="font-size:0.8rem;color:#374151;margin-top:8px;">
    PDF, PNG, JPG, TIFF — Swiss insurance & finance documents
  </div>
</div>""",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
