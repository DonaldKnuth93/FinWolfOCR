"""
FinWolf OCR — Swiss Insurance & Finance Document Intelligence
"""
from __future__ import annotations
import base64
import json
import os
import sys
from pathlib import Path

import fitz
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.core.exporter import export_csv, export_json
from app.core import chat_engine
from app.core import report_generator

st.set_page_config(
    page_title="FinWolf OCR",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _check_credentials(u: str, p: str) -> bool:
    try:
        return u.strip() == st.secrets["auth"]["username"] and p == st.secrets["auth"]["password"]
    except Exception:
        return u.strip() == os.getenv("APP_USERNAME", "admin") and p == os.getenv("APP_PASSWORD", "FinWolf2024!")

def _logo_b64() -> str | None:
    p = ROOT / "assets" / "logo.png"
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else None

def _extract_text(file_bytes: bytes, filename: str) -> str:
    if Path(filename).suffix.lower() != ".pdf":
        return ""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        t = page.get_text("text").strip()
        if t:
            pages.append(f"--- Page {i+1} ---\n{t}")
    doc.close()
    return "\n\n".join(pages)

def _run_extraction(doc_text: str, query: str, filename: str, model: str) -> dict:
    import requests
    system = """You are FinWolf OCR — a document intelligence engine for the Swiss insurance and finance market.
Extract the requested fields and respond ONLY with a valid JSON object. No markdown, no extra text.

Format:
{
  "document_type": "detected type",
  "insurer": "insurance company or null",
  "document_language": "DE|FR|IT|EN",
  "extracted_fields": [
    {"field": "field name", "value": "exact value from document", "confidence": "high|medium|low", "page": 1, "notes": null}
  ],
  "not_found": ["fields not found"],
  "summary": "one sentence summary in English"
}"""
    try:
        r = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Document: {filename}\n\n{doc_text[:5000]}\n\nExtract: {query}"},
                ],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 1500},
            },
            timeout=300,
        )
        raw = r.json().get("message", {}).get("content", "").strip()
        raw = raw.strip("```json").strip("```").strip()
        return json.loads(raw)
    except Exception as e:
        return {"document_type": "Error", "insurer": None, "document_language": "—",
                "extracted_fields": [], "not_found": [], "summary": str(e)}

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp { font-family: 'Inter', sans-serif !important; background: #F0EFF4 !important; color: #0D0D12 !important; }
#MainMenu, footer, header { visibility: hidden; height: 0; }
[data-testid="stDecoration"] { display: none; }
.block-container { max-width: 1360px !important; padding: 0 2.5rem 5rem !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── Topbar ── */
.fw-topbar {
  display: flex; align-items: center; gap: 0;
  background: #0D0D12;
  padding: 0 32px;
  height: 60px;
  margin: -1rem -2.5rem 2rem;
  position: sticky; top: 0; z-index: 999;
}
.fw-topbar-logo { display: flex; align-items: center; gap: 10px; margin-right: 36px; }
.fw-topbar-logo img { height: 28px; }
.fw-topbar-logo-name { font-size: 0.95rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.2px; }
.fw-topbar-logo-sep  { width: 1px; height: 18px; background: rgba(255,255,255,0.12); margin: 0 20px; }
.fw-nav { display: flex; gap: 2px; flex: 1; }
.fw-nav-btn {
  padding: 8px 18px; border-radius: 6px; border: none;
  font-size: 0.82rem; font-weight: 500; cursor: pointer;
  font-family: 'Inter', sans-serif;
  color: rgba(255,255,255,0.45);
  background: transparent;
  transition: all 0.15s ease;
  letter-spacing: 0.1px;
}
.fw-nav-btn:hover { color: rgba(255,255,255,0.85); background: rgba(255,255,255,0.06); }
.fw-nav-btn.active { color: #FFFFFF; background: rgba(255,255,255,0.1); font-weight: 600; }
.fw-topbar-right { display: flex; align-items: center; gap: 16px; margin-left: auto; }
.fw-model-pill {
  display: flex; align-items: center; gap: 7px;
  padding: 5px 14px; border-radius: 20px;
  background: rgba(255,255,255,0.07);
  border: 1px solid rgba(255,255,255,0.1);
  font-size: 0.72rem; font-weight: 600;
  color: rgba(255,255,255,0.55);
  font-family: 'JetBrains Mono', monospace;
}
.fw-dot { width: 6px; height: 6px; border-radius: 50%; }
.fw-dot.on  { background: #10B981; box-shadow: 0 0 5px #10B981; }
.fw-dot.off { background: #EF4444; }

/* ── Panels ── */
.fw-panel {
  background: #FFFFFF; border: 1px solid #E2E2EA;
  border-radius: 12px; padding: 24px;
  margin-bottom: 20px;
}
.fw-panel-title {
  font-size: 0.62rem; font-weight: 700; color: #9CA3AF;
  letter-spacing: 2px; text-transform: uppercase;
  margin-bottom: 18px; display: block;
}

/* ── File rows ── */
.fw-file-row {
  display: flex; align-items: center; gap: 12px;
  padding: 9px 13px; margin-bottom: 5px;
  background: #F9F9FB; border: 1px solid #E2E2EA;
  border-radius: 7px;
}
.fw-file-dot { width: 6px; height: 6px; background: #7B2FE0; border-radius: 1px; flex-shrink: 0; }
.fw-file-name { flex: 1; font-size: 0.81rem; font-weight: 500; color: #374151; }
.fw-file-size { font-size: 0.7rem; color: #9CA3AF; }

/* ── Template chips ── */
.fw-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }
.fw-chip {
  padding: 5px 11px; border-radius: 5px; cursor: default;
  font-size: 0.71rem; font-weight: 600;
  background: #F3F4F6; color: #4B5563;
  border: 1px solid #E5E7EB;
}

/* ── Result card ── */
.fw-result {
  background: #FFFFFF; border: 1px solid #E2E2EA;
  border-radius: 12px; overflow: hidden; margin-bottom: 14px;
}
.fw-result-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 20px; background: #FAFAFA;
  border-bottom: 1px solid #E2E2EA;
}
.fw-result-name { font-size: 0.88rem; font-weight: 700; color: #0D0D12; }
.fw-tags { display: flex; gap: 6px; flex-wrap: wrap; }
.fw-tag {
  padding: 3px 9px; border-radius: 4px;
  font-size: 0.67rem; font-weight: 600;
}
.fw-tag-i { background: #ECFDF5; color: #065F46; }
.fw-tag-l { background: #EFF6FF; color: #1E40AF; }
.fw-tag-t { background: #F5F3FF; color: #5B21B6; }

/* ── Data table ── */
.fw-tbl { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.fw-tbl thead th {
  padding: 9px 18px; text-align: left;
  font-size: 0.59rem; font-weight: 700; letter-spacing: 1.8px;
  text-transform: uppercase; color: #9CA3AF;
  border-bottom: 1px solid #E2E2EA;
  background: #FAFAFA; white-space: nowrap;
}
.fw-tbl tbody td { padding: 11px 18px; border-bottom: 1px solid #F3F4F6; vertical-align: middle; }
.fw-tbl tbody tr:last-child td { border-bottom: none; }
.fw-tbl tbody tr:hover td { background: #FAFAFA; }
.col-f { color: #6B7280; font-weight: 500; }
.col-v { color: #0D0D12; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.79rem; }
.col-p { color: #9CA3AF; text-align: center; font-size: 0.74rem; }
.col-n { color: #9CA3AF; font-size: 0.74rem; font-style: italic; }
.col-m { color: #D1D5DB !important; font-style: italic; }
.tr-m td { opacity: 0.5; }
.conf { display: flex; align-items: center; gap: 7px; white-space: nowrap; }
.conf-bar { width: 24px; height: 3px; border-radius: 2px; }
.conf-bar.high   { background: #10B981; }
.conf-bar.medium { background: #F59E0B; }
.conf-bar.low    { background: #EF4444; }
.conf-txt { font-size: 0.67rem; font-weight: 600; }
.conf-txt.high   { color: #059669; }
.conf-txt.medium { color: #D97706; }
.conf-txt.low    { color: #DC2626; }
.conf-txt.na     { color: #D1D5DB; }
.fw-summary-row {
  padding: 11px 18px; background: #F9FAFB;
  border-top: 1px solid #E2E2EA;
  font-size: 0.77rem; color: #6B7280; line-height: 1.5;
}

/* ── Metrics ── */
.fw-metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 28px; }
.fw-metric  {
  background: #FFFFFF; border: 1px solid #E2E2EA;
  border-radius: 10px; padding: 18px 20px;
}
.fw-metric-val { font-size: 1.85rem; font-weight: 800; color: #7B2FE0; line-height: 1; }
.fw-metric-lbl { font-size: 0.62rem; font-weight: 700; color: #9CA3AF;
                 letter-spacing: 1.5px; text-transform: uppercase; margin-top: 5px; }

/* ── Recommendations ── */
.fw-rec {
  display: flex; gap: 0;
  background: #FFFFFF; border: 1px solid #E2E2EA;
  border-radius: 10px; overflow: hidden;
  margin-bottom: 10px;
}
.fw-rec-stripe { width: 4px; flex-shrink: 0; }
.fw-rec-body   { padding: 16px 20px; flex: 1; }
.fw-rec-title  { font-size: 0.87rem; font-weight: 700; color: #0D0D12; margin-bottom: 4px; }
.fw-rec-text   { font-size: 0.78rem; color: #6B7280; line-height: 1.55; margin-bottom: 7px; }
.fw-rec-action { font-size: 0.68rem; font-weight: 700; color: #7B2FE0; text-transform: uppercase; letter-spacing: 0.8px; }

/* ── Chat ── */
.fw-chat { display: flex; flex-direction: column; gap: 18px; }
.fw-msg  { display: flex; gap: 10px; align-items: flex-start; }
.fw-msg.u { flex-direction: row-reverse; }
.fw-av {
  width: 30px; height: 30px; border-radius: 6px; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.62rem; font-weight: 700; font-family: 'Inter', sans-serif;
}
.fw-av.ai   { background: #7B2FE0; color: #FFF; }
.fw-av.user { background: #0D0D12; color: #FFF; }
.fw-bubble {
  max-width: 68%; padding: 11px 15px; border-radius: 9px;
  font-size: 0.84rem; line-height: 1.6;
}
.fw-bubble.ai {
  background: #FFF; border: 1px solid #E2E2EA;
  color: #0D0D12; border-radius: 2px 9px 9px 9px;
}
.fw-bubble.user {
  background: #0D0D12; color: #FFF;
  border-radius: 9px 2px 9px 9px;
}

/* ── Divider ── */
.fw-div { height: 1px; background: #E2E2EA; margin: 24px 0; }

/* ── Login ── */
.fw-login {
  min-height: 100vh; display: flex; align-items: center; justify-content: center;
  background: #0D0D12;
}
.fw-login-box {
  background: #FFFFFF; border-radius: 16px;
  padding: 48px 44px; width: 400px;
  box-shadow: 0 24px 80px rgba(0,0,0,0.4);
}
.fw-login-head { display: flex; align-items: center; gap: 10px; margin-bottom: 36px; }
.fw-login-head img { height: 32px; }
.fw-login-brand { font-size: 1rem; font-weight: 700; color: #0D0D12; }
.fw-login-title { font-size: 1.45rem; font-weight: 800; color: #0D0D12;
                  letter-spacing: -0.4px; margin-bottom: 5px; }
.fw-login-sub   { font-size: 0.8rem; color: #9CA3AF; margin-bottom: 32px; }

/* ── Streamlit overrides ── */
[data-testid="stFileUploader"] > div:first-child {
  background: #FAFAFA !important; border: 1.5px dashed #D1D5DB !important;
  border-radius: 9px !important;
}
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
  background: #FFFFFF !important; border: 1px solid #E2E2EA !important;
  border-radius: 7px !important; color: #0D0D12 !important;
  font-family: 'Inter', sans-serif !important;
}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {
  border-color: #7B2FE0 !important;
  box-shadow: 0 0 0 3px rgba(123,47,224,0.08) !important;
}
[data-testid="stSelectbox"] > div > div {
  background: #FFF !important; border: 1px solid #E2E2EA !important; border-radius: 7px !important;
}
.stButton > button {
  background: #0D0D12 !important; color: #FFF !important; border: none !important;
  border-radius: 7px !important; font-weight: 600 !important; height: 40px !important;
  font-family: 'Inter', sans-serif !important; font-size: 0.84rem !important;
}
.stButton > button:hover { background: #1F1F2E !important; }
.stButton > button:disabled { background: #E5E7EB !important; color: #9CA3AF !important; }
.stDownloadButton > button {
  background: #FFF !important; color: #374151 !important;
  border: 1px solid #E2E2EA !important; border-radius: 7px !important;
  font-weight: 600 !important; height: 38px !important;
}
.stDownloadButton > button:hover { border-color: #7B2FE0 !important; color: #7B2FE0 !important; }
.stFormSubmitButton > button {
  background: #7B2FE0 !important; color: #FFF !important;
  border: none !important; border-radius: 7px !important;
  font-weight: 600 !important; height: 40px !important;
}
[data-testid="stProgressBar"] > div { background: #E2E2EA !important; border-radius: 3px !important; }
[data-testid="stProgressBar"] > div > div { background: #7B2FE0 !important; border-radius: 3px !important; }
.stSpinner > div { border-top-color: #7B2FE0 !important; }
label, [data-testid] label {
  color: #9CA3AF !important; font-size: 0.62rem !important;
  font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 1.5px !important;
}
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 2px; }
</style>
"""

TEMPLATES = {
    "Contract Data":      "Policy number, insurance company, insurance type (KVG/VVG), start date, end date",
    "Personal Data":      "Name, first name, date of birth, address, AHV number, insured number",
    "Premiums & Costs":   "Monthly premium, annual premium, deductible, co-payment, payment frequency",
    "Supplemental Cover": "Products, hospital class (General/Semi-private/Private), benefits, waiting periods",
    "Vehicle Insurance":  "License plate, vehicle type, mileage, liability, comprehensive, premium",
    "All Key Fields":     "Policy number, insured person, date of birth, insurer, insurance type, all premiums, deductible, co-payment, start date, coverage",
}

def _conf(c: str) -> str:
    c = (c or "").lower()
    if c in ("high","medium","low"):
        return f'<div class="conf"><div class="conf-bar {c}"></div><span class="conf-txt {c}">{c.capitalize()}</span></div>'
    return '<span class="conf-txt na">—</span>'

def _result_card(result: dict, filename: str) -> str:
    dt   = result.get("document_type","Unknown")
    ins  = result.get("insurer") or ""
    lang = result.get("document_language","")
    flds = result.get("extracted_fields",[])
    nf   = result.get("not_found",[])
    summ = result.get("summary","")

    tags = ""
    if ins:  tags += f'<span class="fw-tag fw-tag-i">{ins}</span>'
    if lang: tags += f'<span class="fw-tag fw-tag-l">{lang}</span>'
    if dt:   tags += f'<span class="fw-tag fw-tag-t">{dt}</span>'

    rows = "".join(f"""<tr>
      <td class="col-f">{f.get("field","")}</td>
      <td class="col-v">{f.get("value","")}</td>
      <td>{_conf(f.get("confidence",""))}</td>
      <td class="col-p">{f.get("page","")}</td>
      <td class="col-n">{f.get("notes") or ""}</td>
    </tr>""" for f in flds)
    rows += "".join(f"""<tr class="tr-m">
      <td class="col-f">{n}</td><td class="col-m">Not found</td>
      <td><span class="conf-txt na">—</span></td><td></td><td></td>
    </tr>""" for n in nf)

    foot = f'<div class="fw-summary-row">{summ}</div>' if summ else ""
    return f"""
<div class="fw-result">
  <div class="fw-result-head">
    <span class="fw-result-name">{filename}</span>
    <div class="fw-tags">{tags}</div>
  </div>
  <table class="fw-tbl">
    <thead><tr><th>Field</th><th>Value</th><th>Confidence</th><th>Page</th><th>Notes</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  {foot}
</div>"""

# ── Login ─────────────────────────────────────────────────────────────────────
def show_login():
    st.markdown(CSS, unsafe_allow_html=True)
    logo_b64 = _logo_b64()
    logo_tag = f'<img src="data:image/png;base64,{logo_b64}">' if logo_b64 else ""

    st.markdown(f"""
<div class="fw-login">
  <div class="fw-login-box">
    <div class="fw-login-head">{logo_tag}<span class="fw-login-brand">FinWolf OCR</span></div>
    <div class="fw-login-title">Sign in</div>
    <div class="fw-login-sub">Swiss Insurance &amp; Finance Document Intelligence</div>
  </div>
</div>""", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign In", use_container_width=True):
                if _check_credentials(username, password):
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

# ── Topbar ────────────────────────────────────────────────────────────────────
def render_topbar(model_ok: bool, model_name: str, active: int):
    logo_b64 = _logo_b64()
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="height:26px;">' if logo_b64 else ""
    dot_cls = "on" if model_ok else "off"
    nav_items = ["Extract", "Chat", "Reports"]
    nav_html = "".join(
        f'<span class="fw-nav-btn {"active" if i == active else ""}">{label}</span>'
        for i, label in enumerate(nav_items)
    )
    st.markdown(f"""
<div class="fw-topbar">
  <div class="fw-topbar-logo">
    {logo_html}
    <span class="fw-topbar-logo-name">FinWolf</span>
  </div>
  <div class="fw-topbar-logo-sep"></div>
  <div class="fw-nav">{nav_html}</div>
  <div class="fw-topbar-right">
    <div class="fw-model-pill"><span class="fw-dot {dot_cls}"></span>{model_name if model_ok else "No model"}</div>
  </div>
</div>""", unsafe_allow_html=True)

# ── Tab: Extract ──────────────────────────────────────────────────────────────
def tab_extract(model_ok: bool, model_name: str):
    if not model_ok:
        st.warning("No local model detected. Run: ollama serve")
        return

    left, right = st.columns([1, 1.1], gap="large")

    with left:
        st.markdown('<div class="fw-panel"><span class="fw-panel-title">Documents</span>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload files",
            type=["pdf","png","jpg","jpeg","tiff","tif"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            rows = "".join(
                f'<div class="fw-file-row"><div class="fw-file-dot"></div>'
                f'<span class="fw-file-name">{f.name}</span>'
                f'<span class="fw-file-size">{f.size/1024:.0f} KB</span></div>'
                for f in uploaded
            )
            st.markdown(rows, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="fw-panel"><span class="fw-panel-title">Extraction Query</span>', unsafe_allow_html=True)
        chips = "".join(f'<span class="fw-chip">{k}</span>' for k in TEMPLATES)
        st.markdown(f'<div class="fw-chips">{chips}</div>', unsafe_allow_html=True)
        tmpl = st.selectbox("Template", ["Custom"] + list(TEMPLATES.keys()), label_visibility="collapsed")
        default_q = TEMPLATES[tmpl] if tmpl in TEMPLATES else ""
        query = st.text_area("Query", value=default_q, height=96,
                             placeholder="e.g. Policy number, monthly premium, deductible, insured person",
                             label_visibility="collapsed")
        max_p = st.slider("Max pages", 3, 30, 10)
        run = st.button("Run Extraction", disabled=not (uploaded and query.strip()), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="fw-div"></div>', unsafe_allow_html=True)

    if run and uploaded and query.strip():
        st.session_state["results"]   = []
        st.session_state["doc_texts"] = {}
        prog = st.progress(0, text="Starting...")
        for idx, uf in enumerate(uploaded):
            prog.progress(idx / len(uploaded), text=f"Processing {uf.name}...")
            fb = uf.read()
            dt = _extract_text(fb, uf.name)
            st.session_state["doc_texts"][uf.name] = dt
            if not dt.strip():
                st.warning(f"{uf.name} — No text layer found.")
                continue
            with st.spinner(f"Analyzing {uf.name}..."):
                res = _run_extraction(dt, query, uf.name, model_name)
            st.session_state["results"].append((res, uf.name))
        prog.progress(1.0, text="Complete")

    results = st.session_state.get("results", [])
    if results:
        st.markdown('<span class="fw-panel-title">Results</span>', unsafe_allow_html=True)
        for res, fn in results:
            st.markdown(_result_card(res, fn), unsafe_allow_html=True)
        total = sum(len(r.get("extracted_fields",[])) for r,_ in results)
        st.caption(f"{len(results)} document(s) — {total} fields extracted — {model_name}")
        c1, c2, _ = st.columns([1,1,3])
        with c1: st.download_button("Download CSV",  data=export_csv(results),  file_name="finwolf.csv",  mime="text/csv",         use_container_width=True)
        with c2: st.download_button("Download JSON", data=export_json(results), file_name="finwolf.json", mime="application/json", use_container_width=True)
    elif not uploaded:
        st.markdown(
            '<div style="text-align:center;padding:72px 0;">'
            '<div style="width:44px;height:44px;border:1.5px solid #E2E2EA;border-radius:9px;'
            'margin:0 auto 16px;display:flex;align-items:center;justify-content:center;'
            'font-size:1.1rem;color:#9CA3AF;">+</div>'
            '<div style="font-size:0.93rem;font-weight:600;color:#374151;margin-bottom:5px;">No documents</div>'
            '<div style="font-size:0.77rem;color:#9CA3AF;">Upload PDF, PNG, JPG or TIFF files to begin</div>'
            '</div>', unsafe_allow_html=True,
        )

# ── Tab: Chat ─────────────────────────────────────────────────────────────────
def tab_chat(model_ok: bool, model_name: str):
    doc_texts = st.session_state.get("doc_texts", {})
    if not doc_texts:
        st.markdown(
            '<div style="text-align:center;padding:72px 0;">'
            '<div style="font-size:0.93rem;font-weight:600;color:#374151;margin-bottom:5px;">No document loaded</div>'
            '<div style="font-size:0.77rem;color:#9CA3AF;">Extract a document first, then return here to chat.</div>'
            '</div>', unsafe_allow_html=True,
        )
        return
    if not model_ok:
        st.warning("No local model. Run: ollama serve")
        return

    left, right = st.columns([1, 2.2], gap="large")

    with left:
        st.markdown('<div class="fw-panel"><span class="fw-panel-title">Document</span>', unsafe_allow_html=True)
        doc_name = st.selectbox("Document", list(doc_texts.keys()), label_visibility="collapsed")
        if doc_name:
            st.markdown(
                f'<div style="background:#F9F9FB;border:1px solid #E2E2EA;border-radius:7px;'
                f'padding:13px;font-size:0.7rem;color:#6B7280;line-height:1.6;'
                f'font-family:JetBrains Mono,monospace;margin-top:10px;max-height:220px;overflow:hidden;">'
                f'{doc_texts[doc_name][:400]}...</div>',
                unsafe_allow_html=True,
            )
        if st.button("Clear conversation", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="fw-panel"><span class="fw-panel-title">Conversation</span>', unsafe_allow_html=True)
        history = st.session_state.get("chat_history", [])
        if history:
            msgs = "".join(
                f'<div class="fw-msg {"u" if m["role"]=="user" else ""}">'
                f'<div class="fw-av {"user" if m["role"]=="user" else "ai"}">{"You" if m["role"]=="user" else "FW"}</div>'
                f'<div class="fw-bubble {"user" if m["role"]=="user" else "ai"}">{m["content"]}</div>'
                f'</div>'
                for m in history
            )
            st.markdown(f'<div class="fw-chat">{msgs}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="text-align:center;padding:40px 0;color:#9CA3AF;">'
                '<div style="font-size:0.9rem;font-weight:600;color:#374151;margin-bottom:6px;">Start a conversation</div>'
                '<div style="font-size:0.77rem;">Ask anything — premiums, coverage, dates, insured persons.</div>'
                '</div>', unsafe_allow_html=True,
            )
        with st.form("chat_form", clear_on_submit=True):
            q = st.text_input("Message", placeholder="e.g. What is the monthly premium?", label_visibility="collapsed")
            send = st.form_submit_button("Send", use_container_width=True)
        if send and q.strip() and doc_name:
            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []
            st.session_state["chat_history"].append({"role":"user","content":q})
            with st.spinner(""):
                ans = "".join(chat_engine.chat(doc_texts[doc_name], st.session_state["chat_history"][:-1], q, model_name))
            st.session_state["chat_history"].append({"role":"assistant","content":ans})
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ── Tab: Reports ──────────────────────────────────────────────────────────────
def tab_reports():
    results = st.session_state.get("results", [])
    if not results:
        st.markdown(
            '<div style="text-align:center;padding:72px 0;">'
            '<div style="font-size:0.93rem;font-weight:600;color:#374151;margin-bottom:5px;">No data</div>'
            '<div style="font-size:0.77rem;color:#9CA3AF;">Extract documents first, then return here for your report.</div>'
            '</div>', unsafe_allow_html=True,
        )
        return

    report   = report_generator.build_report(results)
    high_pct = round(report["high_conf"] / max(report["total_fields"], 1) * 100)

    st.markdown(f"""
<div class="fw-metrics">
  <div class="fw-metric"><div class="fw-metric-val">{len(results)}</div><div class="fw-metric-lbl">Documents</div></div>
  <div class="fw-metric"><div class="fw-metric-val">{report["total_fields"]}</div><div class="fw-metric-lbl">Fields Extracted</div></div>
  <div class="fw-metric"><div class="fw-metric-val">{high_pct}%</div><div class="fw-metric-lbl">High Confidence</div></div>
  <div class="fw-metric"><div class="fw-metric-val">{len(report["recommendations"])}</div><div class="fw-metric-lbl">Recommendations</div></div>
</div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.2, 1.5, 1], gap="large")
    with c1:
        st.markdown('<span class="fw-panel-title">Premium Breakdown</span>', unsafe_allow_html=True)
        if report["premium_chart"]:
            st.plotly_chart(report["premium_chart"], use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("No premium amounts found.")
    with c2:
        st.markdown('<span class="fw-panel-title">Coverage Analysis</span>', unsafe_allow_html=True)
        if report["coverage_chart"]:
            st.plotly_chart(report["coverage_chart"], use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("No coverage data found.")
    with c3:
        st.markdown('<span class="fw-panel-title">Confidence</span>', unsafe_allow_html=True)
        if report["confidence_chart"]:
            st.plotly_chart(report["confidence_chart"], use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("No data.")

    if report["insurers"] or report["doc_types"]:
        st.markdown('<div class="fw-div"></div>', unsafe_allow_html=True)
        cc1, cc2 = st.columns(2, gap="large")
        with cc1:
            st.markdown('<span class="fw-panel-title">Detected Insurers</span>', unsafe_allow_html=True)
            tags = "".join(f'<span class="fw-tag fw-tag-i" style="margin:3px;">{i}</span>' for i in report["insurers"])
            st.markdown(tags or '<span style="color:#9CA3AF;font-size:0.8rem;">None detected</span>', unsafe_allow_html=True)
        with cc2:
            st.markdown('<span class="fw-panel-title">Document Types</span>', unsafe_allow_html=True)
            tags = "".join(f'<span class="fw-tag fw-tag-t" style="margin:3px;">{d}</span>' for d in report["doc_types"])
            st.markdown(tags or '<span style="color:#9CA3AF;font-size:0.8rem;">None detected</span>', unsafe_allow_html=True)

    st.markdown('<div class="fw-div"></div>', unsafe_allow_html=True)
    st.markdown('<span class="fw-panel-title">Recommendations</span>', unsafe_allow_html=True)
    stripe_colors = {"warning":"#F59E0B","info":"#3B82F6","tip":"#7B2FE0","saving":"#10B981","success":"#10B981"}
    for rec in report["recommendations"]:
        color = stripe_colors.get(rec.get("type","info"), "#7B2FE0")
        st.markdown(f"""
<div class="fw-rec">
  <div class="fw-rec-stripe" style="background:{color};"></div>
  <div class="fw-rec-body">
    <div class="fw-rec-title">{rec['title']}</div>
    <div class="fw-rec-text">{rec['body']}</div>
    <div class="fw-rec-action">{rec['action']}</div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="fw-div"></div>', unsafe_allow_html=True)
    c1, c2, _ = st.columns([1,1,3])
    with c1: st.download_button("Download CSV",  data=export_csv(results),  file_name="finwolf_report.csv",  mime="text/csv",         use_container_width=True)
    with c2: st.download_button("Download JSON", data=export_json(results), file_name="finwolf_report.json", mime="application/json", use_container_width=True)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.get("authenticated"):
        show_login()
        st.stop()

    for key, default in [("results",[]),("doc_texts",{}),("chat_history",[]),("active_tab",0)]:
        if key not in st.session_state:
            st.session_state[key] = default

    st.markdown(CSS, unsafe_allow_html=True)
    model_ok, model_name = chat_engine.is_available()
    active = st.session_state["active_tab"]

    render_topbar(model_ok, model_name, active)

    nav_cols = st.columns([1, 1, 1, 3, 1])
    labels = ["Extract", "Chat", "Reports"]
    for i, (col, label) in enumerate(zip(nav_cols[:3], labels)):
        with col:
            if st.button(label, key=f"nav_{i}", use_container_width=True):
                st.session_state["active_tab"] = i
                st.rerun()
    with nav_cols[4]:
        if st.button("Sign Out", key="signout"):
            for k in ["authenticated","results","doc_texts","chat_history","active_tab"]:
                st.session_state.pop(k, None)
            st.rerun()

    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

    if active == 0:   tab_extract(model_ok, model_name)
    elif active == 1: tab_chat(model_ok, model_name)
    else:             tab_reports()


if __name__ == "__main__":
    main()
