"""
FinWolf OCR — Swiss Insurance & Finance Document Intelligence
Login-protected · German UI · Light mode · Gemini 2.0 Flash
"""
import base64
import os
import sys
import time
from pathlib import Path

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.core.document_processor import process_document
from app.core.extractor import extract, parse_result
from app.core.exporter import export_csv, export_json, to_dataframe

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinWolf OCR",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── helpers ───────────────────────────────────────────────────────────────────
def _logo_b64() -> str | None:
    p = ROOT / "assets" / "logo.png"
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else None


def _check_credentials(username: str, password: str) -> bool:
    try:
        valid_user = st.secrets["auth"]["username"]
        valid_pass = st.secrets["auth"]["password"]
    except Exception:
        valid_user = os.getenv("APP_USERNAME", "admin")
        valid_pass = os.getenv("APP_PASSWORD", "FinWolf2024!")
    return username.strip() == valid_user and password == valid_pass


@st.cache_resource
def get_api_key() -> str:
    key = (
        st.secrets.get("GEMINI_API_KEY", "")
        or os.getenv("GEMINI_API_KEY", "")
    )
    if not key:
        st.error("⚠️  GEMINI_API_KEY fehlt. Füge ihn in .env oder den Streamlit-Secrets hinzu.")
        st.stop()
    return key


def _badge(confidence: str) -> str:
    c = (confidence or "").lower()
    if c == "high":   return '<span class="badge badge-high">● Hoch</span>'
    if c == "medium": return '<span class="badge badge-medium">● Mittel</span>'
    if c == "low":    return '<span class="badge badge-low">● Tief</span>'
    return ""


def _render_result_html(result: dict, filename: str) -> str:
    doc_type  = result.get("document_type", "Unbekannt")
    insurer   = result.get("insurer") or ""
    lang      = result.get("document_language", "")
    fields    = result.get("extracted_fields", [])
    not_found = result.get("not_found", [])
    summary   = result.get("summary", "")

    insurer_chip = f'<span class="chip chip-insurer">{insurer}</span>' if insurer else ""
    lang_chip    = f'<span class="chip chip-lang">{lang}</span>'       if lang    else ""

    rows = ""
    for f in fields:
        rows += f"""<tr>
          <td class="td-field">{f.get('field','')}</td>
          <td class="td-value">{f.get('value','')}</td>
          <td>{_badge(f.get('confidence',''))}</td>
          <td class="td-page">{f.get('page','')}</td>
          <td class="td-notes">{f.get('notes') or ''}</td>
        </tr>"""
    for nf in not_found:
        rows += f"""<tr class="tr-missing">
          <td class="td-field">{nf}</td>
          <td class="td-value td-missing">— nicht gefunden —</td>
          <td><span class="badge badge-missing">N/A</span></td>
          <td></td><td></td>
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
      <thead><tr>
        <th>FELD</th><th>WERT</th><th>KONFIDENZ</th><th>SEITE</th><th>HINWEISE</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  {summary_block}
</div>"""


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, .stApp {
    background: #F5F3FF !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: #1E1A3A !important;
}
#MainMenu, footer, header { visibility: hidden; height: 0; }
[data-testid="stDecoration"] { display: none; }
.block-container {
    max-width: 1280px !important;
    padding: 0 2.5rem 5rem !important;
}

/* ── Header ── */
.fw-header {
    display: flex; align-items: center; gap: 16px;
    padding: 28px 0 28px;
    border-bottom: 1px solid rgba(124,58,237,0.15);
    margin-bottom: 36px;
}
.fw-logo-img  { height: 52px; width: auto; }
.fw-logo-mono {
    width: 52px; height: 52px;
    background: linear-gradient(135deg,#7C3AED,#4C1D95);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem; font-weight: 900; color: #fff;
    box-shadow: 0 4px 20px rgba(124,58,237,0.3);
    flex-shrink: 0;
}
.fw-title-block { flex: 1; }
.fw-title {
    font-size: 1.75rem; font-weight: 800;
    color: #1E1A3A; letter-spacing: -0.5px; line-height: 1;
}
.fw-subtitle {
    font-size: 0.72rem; font-weight: 600;
    color: #7C3AED; letter-spacing: 2px;
    text-transform: uppercase; margin-top: 4px;
}
.fw-env-badge {
    background: rgba(124,58,237,0.08);
    border: 1px solid rgba(124,58,237,0.2);
    color: #7C3AED; padding: 5px 14px;
    border-radius: 20px; font-size: 0.72rem; font-weight: 700;
}

/* ── Section labels ── */
.fw-section-label {
    font-size: 0.65rem; font-weight: 700;
    color: #9CA3AF; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 12px;
}

/* ── File list ── */
.fw-file-list { display: flex; flex-direction: column; gap: 8px; margin-top: 12px; }
.fw-file-item {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 14px;
    background: rgba(124,58,237,0.05);
    border: 1px solid rgba(124,58,237,0.12);
    border-radius: 10px;
}
.fw-file-name { flex:1; font-size:0.82rem; font-weight:500; color:#374151; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.fw-file-size { font-size:0.72rem; color:#9CA3AF; white-space:nowrap; }

/* ── Results ── */
.fw-result-card {
    background: #FFFFFF;
    border: 1px solid rgba(124,58,237,0.14);
    border-radius: 16px; overflow: hidden;
    margin-bottom: 20px;
    box-shadow: 0 2px 12px rgba(124,58,237,0.06);
}
.fw-doc-header {
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 10px;
    padding: 14px 20px;
    background: rgba(124,58,237,0.04);
    border-bottom: 1px solid rgba(124,58,237,0.1);
}
.fw-doc-left  { display:flex; align-items:center; gap:10px; }
.fw-doc-icon  { font-size:1.1rem; }
.fw-doc-name  { font-size:0.9rem; font-weight:700; color:#1E1A3A; }
.fw-doc-right { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.fw-doc-type  { font-size:0.72rem; color:#7C3AED; font-weight:600; }

.chip { padding:3px 10px; border-radius:20px; font-size:0.68rem; font-weight:700; }
.chip-insurer { background:rgba(16,185,129,0.1); color:#059669; }
.chip-lang    { background:rgba(59,130,246,0.1);  color:#2563EB; }

/* ── Table ── */
.fw-table-wrap { overflow-x: auto; }
.fw-table { width:100%; border-collapse:collapse; font-size:0.82rem; }
.fw-table thead th {
    padding: 9px 18px;
    font-size:0.6rem; font-weight:700; letter-spacing:1.8px;
    text-transform:uppercase; color:#9CA3AF; text-align:left;
    border-bottom:1px solid rgba(0,0,0,0.07);
    background: #FAFAFA;
    white-space:nowrap;
}
.fw-table tbody td {
    padding:11px 18px;
    border-bottom:1px solid rgba(0,0,0,0.05);
    vertical-align:middle;
}
.fw-table tbody tr:last-child td { border-bottom:none; }
.fw-table tbody tr:hover td { background:rgba(124,58,237,0.03); }

.td-field   { color:#6B7280; font-weight:500; white-space:nowrap; }
.td-value   { color:#111827; font-weight:600; font-family:'JetBrains Mono','Courier New',monospace; font-size:0.8rem; }
.td-page    { color:#9CA3AF; text-align:center; font-size:0.75rem; }
.td-notes   { color:#9CA3AF; font-size:0.75rem; font-style:italic; }
.td-missing { color:#D1D5DB !important; font-style:italic; }
.tr-missing td { opacity:0.6; }

.badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.66rem; font-weight:700; white-space:nowrap; }
.badge-high    { background:rgba(16,185,129,0.12); color:#059669; }
.badge-medium  { background:rgba(245,158,11,0.12);  color:#D97706; }
.badge-low     { background:rgba(239,68,68,0.12);   color:#DC2626; }
.badge-missing { background:rgba(156,163,175,0.15); color:#9CA3AF; }

.fw-summary {
    padding:11px 20px;
    background:rgba(124,58,237,0.03);
    border-top:1px solid rgba(124,58,237,0.08);
    font-size:0.78rem; color:#9CA3AF; font-style:italic; line-height:1.5;
}

/* ── Login page ── */
.fw-login-wrap {
    display:flex; flex-direction:column; align-items:center;
    padding: 48px 0 24px;
}
.fw-login-logo-mono {
    width:72px; height:72px;
    background:linear-gradient(135deg,#7C3AED,#4C1D95);
    border-radius:20px;
    display:flex; align-items:center; justify-content:center;
    font-size:1.8rem; font-weight:900; color:#fff;
    box-shadow:0 8px 30px rgba(124,58,237,0.3);
    margin-bottom:20px;
}
.fw-login-logo-img { height:72px; width:auto; margin-bottom:20px; }
.fw-login-title {
    font-size:2rem; font-weight:800; color:#1E1A3A;
    letter-spacing:-0.5px; text-align:center;
}
.fw-login-subtitle {
    font-size:0.8rem; color:#7C3AED; font-weight:600;
    letter-spacing:2px; text-transform:uppercase;
    margin-top:6px; text-align:center; margin-bottom:32px;
}
.fw-login-divider {
    height:1px; background:rgba(124,58,237,0.1);
    margin:24px 0;
}

/* ── Streamlit component overrides ── */
[data-testid="stFileUploader"] > div:first-child {
    background: #FFFFFF !important;
    border: 2px dashed rgba(124,58,237,0.3) !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploader"] > div:first-child:hover {
    border-color: rgba(124,58,237,0.6) !important;
    background: rgba(124,58,237,0.02) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stTextArea"] label,
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label {
    color: #9CA3AF !important; font-size:0.65rem !important;
    font-weight:700 !important; text-transform:uppercase !important;
    letter-spacing:1.2px !important;
}
[data-testid="stTextArea"] textarea {
    background: #FFFFFF !important;
    border: 1px solid rgba(124,58,237,0.22) !important;
    border-radius: 12px !important;
    color: #1E1A3A !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #7C3AED !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #FFFFFF !important;
    border: 1px solid rgba(124,58,237,0.22) !important;
    border-radius: 10px !important;
    color: #1E1A3A !important;
}
[data-testid="stTextInput"] input {
    background: #FFFFFF !important;
    border: 1px solid rgba(124,58,237,0.22) !important;
    border-radius: 10px !important;
    color: #1E1A3A !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #7C3AED !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important;
}
[data-testid="stTextInput"] label {
    color: #9CA3AF !important; font-size:0.65rem !important;
    font-weight:700 !important; text-transform:uppercase !important;
    letter-spacing:1.2px !important;
}
.stButton > button {
    background: linear-gradient(135deg,#7C3AED 0%,#5B21B6 100%) !important;
    color: #FFFFFF !important; border: none !important;
    border-radius: 11px !important; font-weight: 700 !important;
    font-size: 0.9rem !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: 0 4px 18px rgba(124,58,237,0.28) !important;
    transition: all 0.2s ease !important;
    height: 46px !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 28px rgba(124,58,237,0.45) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:disabled {
    background: #E5E7EB !important;
    color: #9CA3AF !important;
    box-shadow: none !important; transform: none !important;
}
.stDownloadButton > button {
    background: rgba(124,58,237,0.07) !important;
    color: #7C3AED !important;
    border: 1px solid rgba(124,58,237,0.25) !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
}
.stDownloadButton > button:hover {
    background: rgba(124,58,237,0.14) !important;
    border-color: rgba(124,58,237,0.45) !important;
}
.stFormSubmitButton > button {
    background: linear-gradient(135deg,#7C3AED 0%,#5B21B6 100%) !important;
    color: #FFFFFF !important; border: none !important;
    border-radius: 11px !important; font-weight: 700 !important;
    font-size: 0.95rem !important; height: 48px !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: 0 4px 18px rgba(124,58,237,0.28) !important;
    transition: all 0.2s !important;
}
.stFormSubmitButton > button:hover {
    box-shadow: 0 6px 28px rgba(124,58,237,0.45) !important;
    transform: translateY(-1px) !important;
}
hr { border-color: rgba(124,58,237,0.1) !important; margin:28px 0 !important; }
[data-testid="stProgressBar"] > div { background:rgba(124,58,237,0.15) !important; border-radius:4px !important; }
[data-testid="stProgressBar"] > div > div { background:linear-gradient(90deg,#7C3AED,#A78BFA) !important; border-radius:4px !important; }
.stSpinner > div { border-top-color:#7C3AED !important; }
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:rgba(124,58,237,0.25); border-radius:3px; }
</style>
"""

# ── Template queries (German) ─────────────────────────────────────────────────
VORLAGEN = {
    "🧾 Vertragsdaten":      "Police Nr., Antragsnummer, Versicherungsgesellschaft, Versicherungsart (KVG/VVG), Versicherungsbeginn, Versicherungsende",
    "👤 Personendaten":      "Name, Vorname, Geburtsdatum, Adresse, AHV-Nummer, Versicherten-Nummer",
    "💰 Prämien & Kosten":   "Monatsprämie, Jahresprämie, Franchise (Jahresfranchise), Selbstbehalt, Zahlungsrhythmus",
    "🏥 Zusatzversicherung": "Gewählte Produkte und Tarife, eingeschlossene Leistungen, Spitalabteilung (Allgemein/Halbprivat/Privat), Wartezeiten",
    "🚗 Fahrzeugversicherung":"Kennzeichen, Fahrzeugtyp, Fahrleistung km/Jahr, Haftpflichtdeckung, Kasko (Teil/Voll), Prämie",
    "📋 Alle Schlüsselfelder":"Extrahiere alle relevanten Felder: Police/Vertragsnummer, Versicherungsnehmer, Geburtsdatum, Versicherungsgesellschaft, Versicherungsart, alle Prämienbeträge, Franchise, Selbstbehalt, Vertragsbeginn, Deckungsumfang",
}

# ── Login page ────────────────────────────────────────────────────────────────
def show_login():
    st.markdown(CSS, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 1.1, 1])
    with center:
        logo_b64 = _logo_b64()
        if logo_b64:
            st.markdown(
                f'<div class="fw-login-wrap"><img src="data:image/png;base64,{logo_b64}" class="fw-login-logo-img"></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="fw-login-wrap"><div class="fw-login-logo-mono">FW</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="fw-login-title">FinWolf OCR</div>'
            '<div class="fw-login-subtitle">Swiss Document Intelligence</div>',
            unsafe_allow_html=True,
        )

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Benutzername")
            password = st.text_input("Passwort", type="password")
            submitted = st.form_submit_button("Anmelden →", use_container_width=True)

            if submitted:
                if _check_credentials(username, password):
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Falscher Benutzername oder Passwort.")

        st.markdown(
            '<p style="text-align:center;font-size:0.72rem;color:#D1D5DB;margin-top:24px;">'
            'FinWolf OCR · Swiss Insurance & Finance Intelligence</p>',
            unsafe_allow_html=True,
        )


# ── Main app ──────────────────────────────────────────────────────────────────
def build_header() -> str:
    logo_b64 = _logo_b64()
    logo_html = (
        f'<img src="data:image/png;base64,{logo_b64}" class="fw-logo-img" alt="FinWolf">'
        if logo_b64 else
        '<div class="fw-logo-mono">FW</div>'
    )
    return f"""
<div class="fw-header">
  {logo_html}
  <div class="fw-title-block">
    <div class="fw-title">FinWolf OCR</div>
    <div class="fw-subtitle">Schweizer Versicherungs- &amp; Finanz-Dokumente</div>
  </div>
  <div class="fw-env-badge">Gemini 2.0 Flash</div>
</div>"""


def main():
    # ── Auth gate ─────────────────────────────────────────────────────────────
    if not st.session_state.get("authenticated"):
        show_login()
        st.stop()

    # ── Sidebar logout ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### FinWolf OCR")
        st.markdown("---")
        if st.button("Abmelden", use_container_width=True):
            st.session_state["authenticated"] = False
            st.rerun()

    # ── Main UI ───────────────────────────────────────────────────────────────
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(build_header(), unsafe_allow_html=True)

    api_key = get_api_key()

    if "results" not in st.session_state:
        st.session_state.results = []

    # ── Input columns ─────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="fw-section-label">Dokumente</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "DATEIEN HOCHLADEN",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            items = "".join(
                f'<div class="fw-file-item">'
                f'<span>{"📄" if f.name.endswith(".pdf") else "🖼️"}</span>'
                f'<span class="fw-file-name">{f.name}</span>'
                f'<span class="fw-file-size">{f.size/1024:.0f} KB</span>'
                f'</div>'
                for f in uploaded
            )
            st.markdown(f'<div class="fw-file-list">{items}</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="fw-section-label">Extraktions-Anfrage</div>', unsafe_allow_html=True)

        vorlage = st.selectbox(
            "SCHNELLVORLAGE",
            ["— oder eigene Anfrage unten eingeben —"] + list(VORLAGEN.keys()),
        )
        default_q = VORLAGEN[vorlage] if vorlage in VORLAGEN else ""

        query = st.text_area(
            "ANFRAGE",
            value=default_q,
            height=110,
            placeholder="z.B.  Police Nr., Monatsprämie, Franchise, Versicherungsnehmer, Beginn",
        )
        max_pages = st.slider("MAX. SEITEN PRO DOKUMENT", 5, 40, 10)

        run = st.button(
            "Extrahieren →",
            type="primary",
            disabled=not (uploaded and query.strip()),
            use_container_width=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Extraction ────────────────────────────────────────────────────────────
    if run and uploaded and query.strip():
        st.session_state.results = []
        progress = st.progress(0, text="Extraktion wird gestartet…")

        for idx, uf in enumerate(uploaded):
            progress.progress(idx / len(uploaded), text=f"Verarbeite {uf.name}…")
            file_bytes = uf.read()

            try:
                with st.spinner(f"Seiten werden gerendert — {uf.name}"):
                    pages = process_document(file_bytes, uf.name)

                page_count = min(len(pages), max_pages)
                note = f" (Dokument hat {len(pages)} Seiten, erste {max_pages} werden analysiert)" if len(pages) > max_pages else ""

                st.markdown(
                    f'<div class="fw-section-label" style="margin-top:8px;">KI liest {page_count} Seite(n) aus {uf.name}{note}</div>',
                    unsafe_allow_html=True,
                )
                stream_box = st.empty()
                raw = ""
                for attempt in range(3):
                    try:
                        raw = ""
                        for chunk in extract(api_key, pages, query, uf.name, max_pages):
                            raw += chunk
                            # Show last ~280 chars so the user sees live output
                            stream_box.markdown(
                                f'<div style="font-family:monospace;font-size:0.72rem;'
                                f'color:#7C3AED;background:rgba(124,58,237,0.05);'
                                f'border-radius:8px;padding:10px 14px;max-height:80px;'
                                f'overflow:hidden;white-space:pre-wrap;line-height:1.4;">'
                                f'{raw[-280:].replace("<","&lt;")}</div>',
                                unsafe_allow_html=True,
                            )
                        break
                    except Exception as e:
                        if "429" in str(e) and attempt < 2:
                            time.sleep(15 * (attempt + 1))
                        else:
                            raise
                stream_box.empty()

                result = parse_result(raw)
                st.session_state.results.append((result, uf.name))

            except ValueError as e:
                st.error(f"**{uf.name}** — {e}")
            except Exception as e:
                err = str(e)
                if "429" in err:
                    st.error(
                        f"**{uf.name}** — API-Limit erreicht. "
                        "Bitte prüfe dein Gemini-Kontingent unter "
                        "[aistudio.google.com](https://aistudio.google.com/app/apikey) "
                        "und stelle sicher, dass Billing aktiviert ist."
                    )
                else:
                    st.error(f"**{uf.name}** — Extraktion fehlgeschlagen: {e}")

        progress.progress(1.0, text="Fertig!")

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state.results:
        st.markdown(
            '<div class="fw-section-label" style="margin-bottom:16px;">Ergebnisse</div>',
            unsafe_allow_html=True,
        )
        for result, filename in st.session_state.results:
            st.markdown(_render_result_html(result, filename), unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="fw-section-label">Export</div>', unsafe_allow_html=True)

        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            st.download_button(
                "⬇  CSV herunterladen",
                data=export_csv(st.session_state.results),
                file_name="finwolf_extraktion.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "⬇  JSON herunterladen",
                data=export_json(st.session_state.results),
                file_name="finwolf_extraktion.json",
                mime="application/json",
                use_container_width=True,
            )

        total_fields = sum(len(r.get("extracted_fields", [])) for r, _ in st.session_state.results)
        st.caption(
            f"{len(st.session_state.results)} Dokument(e) · "
            f"{total_fields} Feld(er) extrahiert · "
            f"Powered by Gemini 2.0 Flash"
        )

    elif not uploaded:
        st.markdown(
            '<div style="text-align:center;padding:60px 0;color:#D1D5DB;">'
            '<div style="font-size:3.5rem;margin-bottom:16px;">📂</div>'
            '<div style="font-size:1rem;font-weight:600;color:#9CA3AF;">Dokument hochladen um zu starten</div>'
            '<div style="font-size:0.8rem;color:#C4B5FD;margin-top:8px;">PDF, PNG, JPG, TIFF · DE / FR / IT / EN / RM</div>'
            '</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
