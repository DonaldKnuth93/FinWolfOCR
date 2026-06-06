"""
Report generator — builds Plotly charts and recommendations from extracted fields.
"""
from __future__ import annotations
import re
from typing import Any

SWISS_INSURERS = [
    "AXA","Allianz","Baloise","CSS","Concordia","Generali","Helvetia",
    "Helsana","KPT","Mobiliar","Sanitas","Swiss Life","Sympany","Visana",
    "Zurich","SWICA","Groupe Mutuel","Assura","Atupri","EGK","ÖKK",
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_amount(value: str) -> float | None:
    """Extract a numeric CHF amount from a string."""
    if not value:
        return None
    clean = re.sub(r"[CHF\s'']", "", str(value)).replace(",", ".")
    try:
        return float(re.search(r"\d+\.?\d*", clean).group())
    except Exception:
        return None


def _field_value(fields: list[dict], *keywords: str) -> str | None:
    """Find first field whose name contains any keyword (case-insensitive)."""
    for f in fields:
        name = f.get("field", "").lower()
        if any(kw.lower() in name for kw in keywords):
            return f.get("value", "")
    return None


# ── Charts ─────────────────────────────────────────────────────────────────────

def premium_chart(fields: list[dict]) -> Any | None:
    """Pie chart of premium breakdown."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    monthly = _parse_amount(_field_value(fields, "monatsprämie", "prime mensuelle", "monthly") or "")
    annual  = _parse_amount(_field_value(fields, "jahresprämie", "annual", "annuelle") or "")
    franchise = _parse_amount(_field_value(fields, "franchise") or "")

    values, labels, colors = [], [], []
    if monthly:
        values.append(monthly * 12)
        labels.append("KVG Jahresprämie")
        colors.append("#7B2FE0")
    if annual and annual != monthly * 12 if monthly else True:
        values.append(annual)
        labels.append("VVG Prämie")
        colors.append("#A78BFA")
    if franchise:
        values.append(franchise)
        labels.append("Franchise")
        colors.append("#1A1F36")

    if not values:
        return None

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#F5F3FF", width=3)),
        textinfo="label+percent",
        textfont=dict(family="Inter", size=13),
        hovertemplate="<b>%{label}</b><br>CHF %{value:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=20, l=20, r=20),
        font=dict(family="Inter", color="#1A1F36"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        annotations=[dict(text="Prämien", x=0.5, y=0.5, font_size=16,
                          font_family="Inter", font_color="#1A1F36", showarrow=False)],
    )
    return fig


def coverage_chart(fields: list[dict]) -> Any | None:
    """Horizontal bar showing coverage areas found."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    coverage_items = [
        ("Grundversicherung KVG",  ["kvg", "grundversicherung", "assurance de base"]),
        ("Zusatzversicherung VVG", ["vvg", "zusatz", "complémentaire"]),
        ("Spitalversicherung",     ["spital", "hôpital", "halbprivat", "privat"]),
        ("Unfallversicherung",     ["unfall", "accident", "uvg"]),
        ("Lebensversicherung",     ["leben", "vie", "säule 3a", "pilier 3a"]),
        ("Fahrzeugversicherung",   ["fahrzeug", "mfz", "auto", "véhicule"]),
        ("Hausratversicherung",    ["hausrat", "ménage", "mobilier"]),
        ("Rechtsschutz",           ["rechtsschutz", "protection juridique"]),
    ]

    doc_text_lower = " ".join(f.get("field","").lower() + " " + f.get("value","").lower() for f in fields)
    found = []
    names = []
    for name, keywords in coverage_items:
        score = sum(1 for kw in keywords if kw in doc_text_lower)
        if score > 0:
            found.append(min(score * 33, 100))
            names.append(name)

    if not found:
        return None

    fig = go.Figure(go.Bar(
        x=found, y=names, orientation="h",
        marker=dict(
            color=["#7B2FE0" if v >= 66 else "#A78BFA" if v >= 33 else "#DDD6FE" for v in found],
            line=dict(width=0),
        ),
        text=[f"{v}%" for v in found],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Abdeckung: %{x}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 120], showgrid=False, visible=False),
        yaxis=dict(showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=60),
        font=dict(family="Inter", color="#1A1F36", size=12),
        height=max(200, len(names) * 45),
    )
    return fig


def confidence_chart(fields: list[dict]) -> Any | None:
    """Donut chart showing confidence distribution."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    high   = sum(1 for f in fields if f.get("confidence") == "high")
    medium = sum(1 for f in fields if f.get("confidence") == "medium")
    low    = sum(1 for f in fields if f.get("confidence") == "low")

    if not (high + medium + low):
        return None

    fig = go.Figure(go.Pie(
        labels=["Hoch", "Mittel", "Tief"],
        values=[high, medium, low],
        hole=0.6,
        marker=dict(colors=["#10B981", "#F59E0B", "#EF4444"],
                    line=dict(color="#F5F3FF", width=3)),
        textinfo="label+value",
        textfont=dict(family="Inter", size=12),
    ))
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        font=dict(family="Inter", color="#1A1F36"),
        annotations=[dict(
            text=f"{high+medium+low}<br>Felder",
            x=0.5, y=0.5, font_size=14,
            font_family="Inter", font_color="#1A1F36", showarrow=False,
        )],
        height=220,
    )
    return fig


# ── Recommendations ────────────────────────────────────────────────────────────

def generate_recommendations(fields: list[dict], doc_type: str = "") -> list[dict]:
    """Rule-based recommendations for Swiss insurance optimization."""
    recs = []
    doc_lower = doc_type.lower()
    field_text = " ".join(f.get("field","").lower() + " " + f.get("value","").lower() for f in fields)

    franchise_val = _parse_amount(_field_value(fields, "franchise") or "")
    monthly_val   = _parse_amount(_field_value(fields, "monatsprämie", "prime mensuelle") or "")
    hospital      = _field_value(fields, "spital", "hôpital", "abteilung") or ""
    has_vvg       = any(kw in field_text for kw in ["vvg", "zusatz", "complémentaire"])
    has_dental     = any(kw in field_text for kw in ["zahn", "dentaire", "dental"])
    has_accident  = any(kw in field_text for kw in ["unfall", "accident", "uvg"])

    if franchise_val and franchise_val >= 2500:
        recs.append({
            "type": "warning",
            "icon": "⚠️",
            "title": "Hohe Franchise erkannt",
            "body": f"Ihre Franchise beträgt CHF {franchise_val:,.0f}. "
                    "Bei häufigen Arztbesuchen könnte eine niedrigere Franchise kostengünstiger sein.",
            "action": "Franchise auf CHF 300–1000 reduzieren prüfen",
        })

    if "allgemein" in hospital.lower():
        recs.append({
            "type": "info",
            "icon": "🏥",
            "title": "Spitalabteilung: Allgemein",
            "body": "Sie sind in der allgemeinen Abteilung versichert. "
                    "Ein Upgrade auf Halbprivat bietet ein eigenes Zimmer und freie Arztwahl.",
            "action": "Upgrade auf Halbprivat oder Privat prüfen",
        })

    if not has_vvg and "kvg" in field_text:
        recs.append({
            "type": "info",
            "icon": "➕",
            "title": "Keine Zusatzversicherung gefunden",
            "body": "Das Dokument enthält nur KVG-Grundversicherung. "
                    "Eine VVG-Zusatzversicherung deckt Zahnpflege, alternative Medizin und mehr ab.",
            "action": "Angebot für VVG-Zusatzversicherung einholen",
        })

    if not has_dental:
        recs.append({
            "type": "tip",
            "icon": "🦷",
            "title": "Zahnversicherung nicht erkannt",
            "body": "Zahnbehandlungen sind in der Schweiz teuer und von der KVG nicht gedeckt. "
                    "Eine Zahnzusatzversicherung ab ca. CHF 15/Monat ist empfehlenswert.",
            "action": "Zahnversicherung hinzufügen",
        })

    if monthly_val and monthly_val > 600:
        recs.append({
            "type": "saving",
            "icon": "💰",
            "title": "Prämienoptimierung möglich",
            "body": f"Ihre monatliche Prämie von CHF {monthly_val:,.0f} liegt über dem Schweizer Durchschnitt. "
                    "Ein Prämiienvergleich könnte CHF 50–200/Monat einsparen.",
            "action": "Prämienvergleich auf comparis.ch durchführen",
        })

    if not recs:
        recs.append({
            "type": "success",
            "icon": "✅",
            "title": "Versicherungsschutz vollständig",
            "body": "Basierend auf dem analysierten Dokument ist Ihr Versicherungsschutz umfassend. "
                    "Wir empfehlen eine jährliche Überprüfung Ihrer Police.",
            "action": "Jährlichen Review einplanen",
        })

    return recs


# ── HTML Report Export ─────────────────────────────────────────────────────────

def build_report(results: list[tuple[dict, str]]) -> dict:
    """Aggregate all extracted results into a report data structure."""
    all_fields = []
    doc_types  = []
    insurers   = []

    for result, filename in results:
        all_fields.extend(result.get("extracted_fields", []))
        if result.get("document_type"):
            doc_types.append(result["document_type"])
        if result.get("insurer"):
            insurers.append(result["insurer"])

    return {
        "fields":          all_fields,
        "doc_types":       list(set(doc_types)),
        "insurers":        list(set(insurers)),
        "total_fields":    len(all_fields),
        "high_conf":       sum(1 for f in all_fields if f.get("confidence") == "high"),
        "premium_chart":   premium_chart(all_fields),
        "coverage_chart":  coverage_chart(all_fields),
        "confidence_chart":confidence_chart(all_fields),
        "recommendations": generate_recommendations(all_fields, " ".join(doc_types)),
    }
