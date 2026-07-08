"""
LexAssist - Multi-Agent Legal Research System
Updated UI: clean home page, enriched classifier with sample dataset tabs + audience roles
Run: streamlit run streamlit_app.py
Backend: uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import time
import requests
import pandas as pd
from pathlib import Path
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexAssist",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_BASE = "http://localhost:8000/api"
DATA_DIR = Path(__file__).parent / "data" / "legal_clauses"

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "page"           not in st.session_state: st.session_state.page          = "home"
if "dataset_file"   not in st.session_state: st.session_state.dataset_file  = None
if "clause_prefill" not in st.session_state: st.session_state.clause_prefill = ""

# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE DATASET DEFINITIONS  (Classifier page tabs)
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_DATASETS = [
    {
        "tab":   "📋 Contract Clauses",
        "title": "Contract Clauses Dataset",
        "desc":  "Standard clauses found in commercial contracts — indemnification, termination, warranties, governing law, force majeure and more.",
        "sample_clause": "The Customer shall indemnify, defend, and hold harmless the Company and its officers from any claims, damages, liabilities, costs, and expenses arising out of Customer's use of the Services.",
        "audiences": [
            {
                "persona": "⚖️ Attorney / Legal Counsel",
                "color": "#7c3aed",
                "bg": "#f5f3ff",
                "border": "#c4b5fd",
                "relevance": "Core dataset for reviewing, drafting, and negotiating commercial agreements. Covers every standard clause type used in practice.",
                "use_cases": [
                    "Identify missing or non-standard clauses in a draft contract",
                    "Compare client's clause wording against market-standard language",
                    "Flag unfavourable indemnification or liability caps before signing",
                ],
                "questions": [
                    "Is this termination clause standard or unusually broad?",
                    "What does a typical limitation-of-liability clause look like?",
                    "Are there any missing representations in this agreement?",
                ],
            },
            {
                "persona": "🏢 In-house Legal / Compliance Team",
                "color": "#2563eb",
                "bg": "#eff6ff",
                "border": "#93c5fd",
                "relevance": "Enables rapid clause-level review across high-volume contract pipelines without external counsel for every document.",
                "use_cases": [
                    "Automate first-pass review of vendor agreements",
                    "Check clause consistency across a portfolio of contracts",
                    "Maintain a clause library aligned with company policy",
                ],
                "questions": [
                    "Does this vendor contract have a data protection clause?",
                    "What are the standard payment terms in our industry?",
                    "Which contracts are missing force majeure provisions?",
                ],
            },
            {
                "persona": "👔 CEO / Corporate Leadership",
                "color": "#0d9488",
                "bg": "#f0fdfa",
                "border": "#5eead4",
                "relevance": "High-level summaries of what key clauses mean in plain English — useful before signing major deals.",
                "use_cases": [
                    "Understand what you're committing to before signing",
                    "Get a plain-English risk summary of a contract",
                    "Compare two vendor proposals side by side",
                ],
                "questions": [
                    "What does this indemnification clause actually mean for us?",
                    "What are our obligations if we terminate early?",
                    "Is this contract riskier than our standard agreements?",
                ],
            },
            {
                "persona": "📦 Contract Managers / Procurement",
                "color": "#d97706",
                "bg": "#fffbeb",
                "border": "#fcd34d",
                "relevance": "Speeds up procurement cycles by instantly identifying clause types, flagging deviations from preferred terms.",
                "use_cases": [
                    "Screen supplier contracts against approved clause templates",
                    "Track which suppliers have non-standard payment terms",
                    "Generate clause summaries for non-legal stakeholders",
                ],
                "questions": [
                    "Does this supplier contract match our preferred indemnification language?",
                    "What payment terms are standard across our vendor contracts?",
                    "Which clauses differ from our contract template?",
                ],
            },
        ],
    },
    {
        "tab":   "🛡️ Compliance Policies",
        "title": "Compliance Policies Dataset",
        "desc":  "Regulatory compliance clauses covering data protection, anti-bribery, sanctions, AML, and internal policy obligations.",
        "sample_clause": "The Company shall implement and maintain appropriate technical and organisational measures to ensure a level of security appropriate to the risk posed by the processing of personal data, including as appropriate encryption and pseudonymisation.",
        "audiences": [
            {
                "persona": "🏢 In-house Legal / Compliance Team",
                "color": "#2563eb",
                "bg": "#eff6ff",
                "border": "#93c5fd",
                "relevance": "Essential for mapping compliance obligations across jurisdictions and ensuring policy documents are complete and enforceable.",
                "use_cases": [
                    "Audit internal policies for GDPR, CCPA, or sector-specific compliance",
                    "Identify gaps in data processing agreements",
                    "Create compliance checklists from policy clauses",
                ],
                "questions": [
                    "Does our data processing agreement cover all GDPR obligations?",
                    "Are our anti-bribery clauses consistent with UK Bribery Act requirements?",
                    "What security obligations do our vendors need to meet?",
                ],
            },
            {
                "persona": "⚖️ Attorney / Legal Counsel",
                "color": "#7c3aed",
                "bg": "#f5f3ff",
                "border": "#c4b5fd",
                "relevance": "Research regulatory language and precedents for drafting compliant policies and advising on regulatory exposure.",
                "use_cases": [
                    "Draft compliant data protection addenda",
                    "Advise clients on regulatory risk in existing agreements",
                    "Research sanctions and AML clause language",
                ],
                "questions": [
                    "What does a compliant data retention clause look like?",
                    "What are the standard AML obligations in financial contracts?",
                    "Is this sanctions clause broad enough to cover OFAC requirements?",
                ],
            },
            {
                "persona": "🔬 Researchers / Policy Analysts",
                "color": "#db2777",
                "bg": "#fdf2f8",
                "border": "#f9a8d4",
                "relevance": "Analyse how compliance obligations are drafted across industries and jurisdictions for academic or policy research.",
                "use_cases": [
                    "Compare compliance language across different sectors",
                    "Study how GDPR obligations are implemented in practice",
                    "Research trends in data protection clause drafting",
                ],
                "questions": [
                    "How do data protection clauses differ between EU and US contracts?",
                    "What patterns exist in anti-bribery clause language?",
                    "How have compliance obligations evolved over time?",
                ],
            },
        ],
    },
    {
        "tab":   "👩‍💼 Employment Agreements",
        "title": "Employment Agreements Dataset",
        "desc":  "Clauses from employment contracts — non-compete, non-solicitation, severance, IP assignment, confidentiality, and termination terms.",
        "sample_clause": "During Executive's employment and for a period of two (2) years following the date of termination, Executive will not, directly or indirectly, hire or attempt to hire any employee of the Company or any affiliate of the Company.",
        "audiences": [
            {
                "persona": "⚖️ Attorney / Legal Counsel",
                "color": "#7c3aed",
                "bg": "#f5f3ff",
                "border": "#c4b5fd",
                "relevance": "Research enforceable non-compete and non-solicitation language, draft severance terms, and advise on employment disputes.",
                "use_cases": [
                    "Draft enforceable non-compete clauses by jurisdiction",
                    "Advise executives on severance entitlements",
                    "Review IP assignment clauses in employment agreements",
                ],
                "questions": [
                    "What is the standard non-compete duration for senior executives?",
                    "Is a 2-year non-solicitation clause enforceable in California?",
                    "What IP does an employee typically assign to their employer?",
                ],
            },
            {
                "persona": "🏢 In-house Legal / Compliance Team",
                "color": "#2563eb",
                "bg": "#eff6ff",
                "border": "#93c5fd",
                "relevance": "Standardise employment agreements across the organisation and ensure compliance with labour law obligations.",
                "use_cases": [
                    "Audit all employment contracts for consistent non-compete terms",
                    "Identify contracts missing IP assignment clauses",
                    "Benchmark severance terms against market practice",
                ],
                "questions": [
                    "Are our non-solicitation clauses consistent across all employee levels?",
                    "Do our employment agreements comply with local labour laws?",
                    "What is the market-standard notice period for senior hires?",
                ],
            },
            {
                "persona": "👔 CEO / Corporate Leadership",
                "color": "#0d9488",
                "bg": "#f0fdfa",
                "border": "#5eead4",
                "relevance": "Understand executive agreement terms, severance exposure, and post-employment restrictions before hiring or exiting senior leaders.",
                "use_cases": [
                    "Understand the financial exposure of executive severance packages",
                    "Review post-employment restrictions before making a senior hire",
                    "Compare compensation and benefits terms across executive offers",
                ],
                "questions": [
                    "What severance am I obligated to pay if I terminate this executive?",
                    "Does this executive's non-compete prevent them from joining a competitor?",
                    "What are the IP ownership implications of this employment offer?",
                ],
            },
            {
                "persona": "📦 Contract Managers / Procurement",
                "color": "#d97706",
                "bg": "#fffbeb",
                "border": "#fcd34d",
                "relevance": "Manage contractor and consultant agreements, ensuring scope, IP ownership, and termination terms are clearly defined.",
                "use_cases": [
                    "Review contractor agreements for IP ownership clarity",
                    "Ensure consultant contracts include appropriate confidentiality terms",
                    "Standardise termination clauses across freelancer agreements",
                ],
                "questions": [
                    "Who owns the IP in a contractor-created deliverable?",
                    "What notice period is required to terminate a contractor agreement?",
                    "Does this consulting agreement include a non-solicitation clause?",
                ],
            },
        ],
    },
    {
        "tab":   "📜 Regulatory Documents",
        "title": "Regulatory Documents Dataset",
        "desc":  "Clauses from regulatory filings, government contracts, financial regulations, and sector-specific compliance documents.",
        "sample_clause": "The Regulated Entity shall maintain at all times a minimum capital adequacy ratio of not less than eight per cent (8%) of its risk-weighted assets, calculated in accordance with the applicable prudential standards issued by the Authority.",
        "audiences": [
            {
                "persona": "⚖️ Attorney / Legal Counsel",
                "color": "#7c3aed",
                "bg": "#f5f3ff",
                "border": "#c4b5fd",
                "relevance": "Research regulatory obligations, draft compliant submissions, and advise clients on regulatory risk across sectors.",
                "use_cases": [
                    "Research precedent regulatory language for financial filings",
                    "Draft responses to regulatory investigations",
                    "Advise on capital adequacy and prudential requirements",
                ],
                "questions": [
                    "What are the standard capital requirements in financial services regulations?",
                    "How is 'material adverse change' defined in regulatory contexts?",
                    "What disclosure obligations apply to publicly traded companies?",
                ],
            },
            {
                "persona": "🏢 In-house Legal / Compliance Team",
                "color": "#2563eb",
                "bg": "#eff6ff",
                "border": "#93c5fd",
                "relevance": "Monitor regulatory changes, ensure internal policies align with current regulatory requirements, and prepare for audits.",
                "use_cases": [
                    "Map regulatory obligations to internal policy controls",
                    "Prepare for regulatory audits with clause-level compliance checks",
                    "Monitor changes in applicable financial regulations",
                ],
                "questions": [
                    "Do our internal policies reflect current regulatory requirements?",
                    "What regulatory reporting obligations apply to our business?",
                    "Are we in compliance with applicable capital adequacy requirements?",
                ],
            },
            {
                "persona": "🔬 Researchers / Policy Analysts",
                "color": "#db2777",
                "bg": "#fdf2f8",
                "border": "#f9a8d4",
                "relevance": "Analyse regulatory trends, compare obligations across jurisdictions, and study the evolution of regulatory language.",
                "use_cases": [
                    "Compare financial regulations across different jurisdictions",
                    "Study how regulatory language has evolved post-financial crisis",
                    "Analyse the gap between regulatory intent and enacted language",
                ],
                "questions": [
                    "How do prudential standards differ between EU and US regulators?",
                    "What patterns exist in regulatory enforcement language?",
                    "How have AML obligations changed over the past decade?",
                ],
            },
            {
                "persona": "👔 CEO / Corporate Leadership",
                "color": "#0d9488",
                "bg": "#f0fdfa",
                "border": "#5eead4",
                "relevance": "Understand regulatory obligations that directly affect business strategy, capital allocation, and risk exposure.",
                "use_cases": [
                    "Understand capital requirements before entering a new regulated market",
                    "Assess regulatory risk when considering a merger or acquisition",
                    "Get plain-English summaries of regulatory filings",
                ],
                "questions": [
                    "What regulatory approvals do we need to enter this market?",
                    "What are the penalties for breaching these capital requirements?",
                    "How does this regulation affect our business model?",
                ],
            },
        ],
    },
    {
        "tab":   "🎓 Student Policy Resources",
        "title": "International Student Policy Resources",
        "desc":  "Policy clauses relevant to international students — visa conditions, enrolment obligations, fee refund policies, housing agreements, and institutional compliance requirements.",
        "sample_clause": "International students must maintain full-time enrolment status as defined by the institution and comply with all visa conditions stipulated by the relevant immigration authority. Failure to maintain the required course load may result in notification to the immigration authority and potential visa cancellation.",
        "audiences": [
            {
                "persona": "🎓 International Students",
                "color": "#7c3aed",
                "bg": "#f5f3ff",
                "border": "#c4b5fd",
                "relevance": "Understand visa obligations, enrolment requirements, fee structures, and housing terms in plain English.",
                "use_cases": [
                    "Understand what maintaining 'full-time status' means for your visa",
                    "Know your rights if a course is cancelled or fee is disputed",
                    "Review housing agreement terms before signing",
                ],
                "questions": [
                    "What happens to my visa if I drop below full-time enrolment?",
                    "Am I entitled to a tuition refund if I withdraw early?",
                    "What are my obligations under the student housing agreement?",
                ],
            },
            {
                "persona": "🏢 In-house Legal / Compliance Team",
                "color": "#2563eb",
                "bg": "#eff6ff",
                "border": "#93c5fd",
                "relevance": "Ensure institutional policies comply with immigration law, accreditation standards, and student protection regulations.",
                "use_cases": [
                    "Audit student enrolment agreements for regulatory compliance",
                    "Ensure fee refund policies meet consumer protection requirements",
                    "Review visa reporting obligations for sponsored students",
                ],
                "questions": [
                    "Does our enrolment agreement comply with visa sponsor obligations?",
                    "Are our fee refund policies compliant with consumer protection law?",
                    "What reporting obligations do we have when a student withdraws?",
                ],
            },
            {
                "persona": "🔬 Researchers / Policy Analysts",
                "color": "#db2777",
                "bg": "#fdf2f8",
                "border": "#f9a8d4",
                "relevance": "Study student policy language across institutions, analyse visa obligation clauses, and research international student welfare frameworks.",
                "use_cases": [
                    "Compare student policy clauses across different universities",
                    "Analyse visa obligation language across host countries",
                    "Research best practices in international student welfare policies",
                ],
                "questions": [
                    "How do enrolment obligations differ across institutions?",
                    "What visa compliance clauses are most commonly included in student agreements?",
                    "How do refund policies compare across higher education institutions?",
                ],
            },
            {
                "persona": "⚖️ Attorney / Legal Counsel",
                "color": "#0d9488",
                "bg": "#f0fdfa",
                "border": "#5eead4",
                "relevance": "Advise students or institutions on disputes arising from enrolment agreements, visa breaches, or fee disputes.",
                "use_cases": [
                    "Advise a student on rights following a visa condition breach notice",
                    "Review institutional refund policy for enforceability",
                    "Represent a student in a housing agreement dispute",
                ],
                "questions": [
                    "Is this enrolment termination clause enforceable?",
                    "What recourse does a student have if the institution cancels their course?",
                    "Does this housing agreement comply with residential tenancy law?",
                ],
            },
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def go(page: str, **kwargs):
    st.session_state.page = page
    for k, v in kwargs.items():
        st.session_state[k] = v
    st.rerun()


def api_get(endpoint: str):
    try:
        r = requests.get(API_BASE + endpoint, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_post(endpoint: str, payload: dict, timeout: int = 120):
    try:
        r = requests.post(API_BASE + endpoint, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Run: uvicorn app.main:app --reload --port 8000"}
    except Exception as e:
        return {"error": str(e)}


def backend_alive() -> bool:
    try:
        return requests.get("http://localhost:8000/health", timeout=3).status_code == 200
    except Exception:
        return False


def parse_risk(raw: str) -> dict:
    try:
        return json.loads(raw.strip().strip("```json").strip("```").strip())
    except Exception:
        return {"risk_level": "unknown", "risk_factors": [], "recommendation": raw, "flagged_phrases": []}


def get_csv_files() -> list:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.csv"))


def load_csv(filepath: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        try:
            return pd.read_csv(filepath, encoding="latin-1", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()


def fmt_domain(stem: str) -> str:
    return stem.replace("-", " ").replace("_", " ").title()


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&display=swap');

*, html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif !important; }
.block-container { padding: 2rem 3rem 3rem !important; max-width: 1400px; }

#MainMenu, footer, header, section[data-testid="stSidebar"] { display: none !important; }
div[data-testid="stToolbar"] { display: none !important; }

/* ── top nav ── */
.topnav { display:flex; align-items:center; justify-content:space-between;
          padding:0.8rem 0 1.5rem; border-bottom:2px solid #f1f5f9; margin-bottom:2rem; }
.topnav .logo { font-size:1.5rem; font-weight:900;
    background:linear-gradient(135deg,#7c3aed,#4f46e5,#0ea5e9);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; letter-spacing:-0.5px; }
.status-pill { display:inline-flex; align-items:center; gap:6px;
               padding:6px 16px; border-radius:999px; font-size:0.78rem; font-weight:700; }
.status-pill.online  { background:#dcfce7; color:#15803d; border:1px solid #86efac; }
.status-pill.offline { background:#fee2e2; color:#b91c1c; border:1px solid #fca5a5; }

/* ── hero ── */
.hero-wrap { text-align:center; padding:3rem 2rem 2rem; }
.hero-wrap h1 { font-size:3.2rem; font-weight:900; line-height:1.1; letter-spacing:-1.5px;
    background:linear-gradient(135deg,#7c3aed 0%,#4f46e5 40%,#0ea5e9 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; margin-bottom:1rem; }
.hero-wrap p { font-size:1.1rem; color:#64748b; max-width:600px;
               margin:0 auto 2rem; line-height:1.7; }

/* ── stat strip ── */
.stat-strip { display:flex; gap:1.5rem; justify-content:center;
              margin-bottom:3rem; flex-wrap:wrap; }
.stat-chip { background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
             padding:0.7rem 1.4rem; text-align:center; }
.stat-chip .sv { font-size:1.4rem; font-weight:800; color:#1e1b4b; }
.stat-chip .sl { font-size:0.72rem; color:#94a3b8; font-weight:600;
                 text-transform:uppercase; letter-spacing:0.08em; }

/* ── section cards ── */
.section-card { border-radius:24px; padding:2rem 1.8rem; border:2px solid transparent;
    position:relative; overflow:hidden; text-align:center; transition:all 0.25s ease; }
.section-card:hover { transform:translateY(-4px); box-shadow:0 20px 50px rgba(0,0,0,0.10); }
.section-card.purple { background:linear-gradient(135deg,#f5f3ff,#ede9fe); border-color:#c4b5fd; }
.section-card.purple:hover { border-color:#7c3aed; box-shadow:0 20px 50px #7c3aed22; }
.section-card.blue   { background:linear-gradient(135deg,#eff6ff,#dbeafe); border-color:#93c5fd; }
.section-card.blue:hover   { border-color:#2563eb; box-shadow:0 20px 50px #2563eb22; }
.section-card.rose   { background:linear-gradient(135deg,#fff1f2,#ffe4e6); border-color:#fca5a5; }
.section-card.rose:hover   { border-color:#e11d48; box-shadow:0 20px 50px #e11d4822; }
.section-card.green  { background:linear-gradient(135deg,#f0fdf4,#dcfce7); border-color:#86efac; }
.section-card.green:hover  { border-color:#16a34a; box-shadow:0 20px 50px #16a34a22; }
.section-card.amber  { background:linear-gradient(135deg,#fffbeb,#fef3c7); border-color:#fcd34d; }
.section-card.amber:hover  { border-color:#d97706; box-shadow:0 20px 50px #d9770622; }
.section-card .card-icon  { font-size:2.5rem; margin-bottom:0.8rem; display:block; }
.section-card .card-title { font-size:1.1rem; font-weight:800; color:#1e1b4b; margin-bottom:0.4rem; }
.section-card .card-desc  { font-size:0.82rem; color:#64748b; line-height:1.5; }
.section-card .card-arrow { display:inline-flex; align-items:center; justify-content:center;
    width:32px; height:32px; border-radius:50%; background:rgba(255,255,255,0.7);
    font-size:1rem; margin-top:1rem; font-weight:700; color:#1e1b4b; }

/* ── page title bar ── */
.page-title-bar { display:flex; align-items:center; gap:14px;
                  margin-bottom:1.8rem; padding-bottom:1rem; border-bottom:2px solid #f1f5f9; }
.page-title-bar .ptb-icon { width:48px; height:48px; border-radius:14px;
    display:flex; align-items:center; justify-content:center; font-size:1.5rem; }
.page-title-bar h2 { font-size:1.6rem; font-weight:800; color:#1e1b4b; margin:0; }
.page-title-bar p  { font-size:0.85rem; color:#64748b; margin:2px 0 0; }

/* ── result cards ── */
.rcard { border-radius:16px; padding:1.4rem 1.6rem; margin-bottom:1rem; border:1.5px solid; }
.rcard.purple { background:linear-gradient(135deg,#f5f3ff,#ede9fe); border-color:#c4b5fd; }
.rcard.blue   { background:linear-gradient(135deg,#eff6ff,#dbeafe); border-color:#93c5fd; }
.rcard.green  { background:linear-gradient(135deg,#f0fdf4,#dcfce7); border-color:#86efac; }
.rcard.amber  { background:linear-gradient(135deg,#fffbeb,#fef3c7); border-color:#fcd34d; }
.rcard.rose   { background:linear-gradient(135deg,#fff1f2,#ffe4e6); border-color:#fca5a5; }
.rcard .rt { font-size:0.72rem; font-weight:700; text-transform:uppercase;
             letter-spacing:0.1em; color:#6b7280; margin-bottom:0.5rem; }
.rcard .rv { font-size:1.2rem; font-weight:800; color:#1e1b4b; }
.rcard .rb { font-size:0.9rem; color:#374151; line-height:1.6; }

/* ── risk badge ── */
.rbadge { display:inline-flex; align-items:center; gap:5px;
          padding:5px 14px; border-radius:999px; font-size:0.78rem; font-weight:700; }
.rbadge.low    { background:#dcfce7; color:#15803d; border:1px solid #86efac; }
.rbadge.medium { background:#fef9c3; color:#a16207; border:1px solid #fde047; }
.rbadge.high   { background:#fee2e2; color:#b91c1c; border:1px solid #fca5a5; }

/* ── clause box ── */
.clause-box { background:#fafafa; border:1px solid #e5e7eb; border-left:4px solid #7c3aed;
    border-radius:0 10px 10px 0; padding:0.9rem 1.1rem; font-size:0.84rem;
    color:#374151; white-space:pre-wrap; word-break:break-word; margin-bottom:0.8rem; line-height:1.6; }

/* ── dataset header ── */
.dataset-header { background:linear-gradient(135deg,#7c3aed,#4f46e5);
    border-radius:16px; padding:1.5rem 1.8rem; margin-bottom:1.5rem; color:white; }
.dataset-header h3 { font-size:1.3rem; font-weight:800; margin:0 0 0.3rem; }
.dataset-header p  { font-size:0.85rem; opacity:0.85; margin:0; }

/* ── audience card ── */
.audience-card { border-radius:16px; padding:1.3rem 1.5rem; margin-bottom:1rem; border:1.5px solid; }
.audience-card .ac-title { font-size:1rem; font-weight:800; color:#1e1b4b; margin-bottom:0.8rem; }
.audience-card .ac-label { font-size:0.7rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.1em; margin-bottom:0.4rem; }
.audience-card ul { margin:0 0 0 1rem; padding:0; }
.audience-card li { font-size:0.84rem; color:#374151; margin-bottom:3px; line-height:1.5; }
.q-pill { display:inline-block; background:#f1f5f9; border:1px solid #e2e8f0;
    color:#475569; border-radius:8px; padding:5px 10px; font-size:0.78rem;
    font-weight:500; margin:3px; line-height:1.4; }

/* ── sample clause box ── */
.sample-clause { background:#f8fafc; border:2px solid #e2e8f0; border-left:5px solid #7c3aed;
    border-radius:0 14px 14px 0; padding:1.1rem 1.3rem; font-size:0.86rem;
    color:#374151; line-height:1.7; margin-bottom:1.2rem; font-style:italic; }

/* ── step progress ── */
.step-row { display:flex; gap:0.5rem; align-items:center; margin-bottom:1rem; }
.step { display:flex; align-items:center; gap:6px; background:#f3f4f6;
    border-radius:999px; padding:5px 14px; font-size:0.78rem; font-weight:600; color:#6b7280; }
.step.active { background:linear-gradient(135deg,#7c3aed,#4f46e5); color:#fff; }
.step.done   { background:#dcfce7; color:#15803d; }
.step-arrow  { color:#d1d5db; font-size:0.8rem; }

/* ── buttons ── */
.stButton > button { border-radius:999px !important; font-weight:700 !important; transition:all 0.2s !important; }
.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#7c3aed,#4f46e5) !important;
    color:white !important; border:none !important;
    box-shadow:0 4px 15px #7c3aed33 !important; padding:0.6rem 2rem !important; }
.stButton > button[kind="primary"]:hover { transform:translateY(-2px) !important; box-shadow:0 8px 25px #7c3aed44 !important; }
.stButton > button[kind="secondary"] { background:white !important; color:#7c3aed !important; border:2px solid #c4b5fd !important; }
.stButton > button[kind="secondary"]:hover { background:#f5f3ff !important; transform:translateY(-1px) !important; }

div[data-testid="stProgress"] > div > div > div {
    background:linear-gradient(90deg,#7c3aed,#4f46e5,#0ea5e9) !important; }
div[data-testid="stAlert"] { border-radius:14px !important; }
.stTextArea textarea { border-radius:14px !important; border:2px solid #e2e8f0 !important; font-size:0.92rem !important; }
.stTextArea textarea:focus { border-color:#7c3aed !important; box-shadow:0 0 0 3px #7c3aed18 !important; }
.stSelectbox > div > div { border-radius:12px !important; }
div[data-testid="stTabs"] button { border-radius:10px 10px 0 0 !important; font-weight:600 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TOP NAV BAR
# ─────────────────────────────────────────────────────────────────────────────
alive = backend_alive()
sc    = "online" if alive else "offline"
sd    = "🟢"     if alive else "🔴"
st_   = "Backend Online" if alive else "Backend Offline"

nav1, nav2, nav3 = st.columns([2, 6, 2])
with nav1:
    st.markdown('<div class="topnav"><span class="logo">⚖️ LexAssist</span></div>', unsafe_allow_html=True)
with nav3:
    st.markdown(
        '<div class="topnav" style="justify-content:flex-end">'
        '<span class="status-pill ' + sc + '">' + sd + " " + st_ + '</span></div>',
        unsafe_allow_html=True,
    )

# ── Back button (all non-home pages) ─────────────────────────────────────────
if st.session_state.page != "home":
    if st.button("← Back to Home", key="back_home"):
        go("home")


# =============================================================================
# HOME PAGE
# =============================================================================
if st.session_state.page == "home":

    # Hero
    st.markdown("""
    <div class="hero-wrap">
        <h1>Legal Research,<br>Simplified.</h1>
        <p>Search, classify and analyse 150,000+ legal clauses across 395 domains
           using AI-powered multi-agent research.</p>
    </div>
    """, unsafe_allow_html=True)

    # Stat strip
    stats   = api_get("/stats") if alive else {}
    v_count = stats.get("total_vectors", 0) if stats and "error" not in stats else 0
    d_count = stats.get("total_domains", 0) if stats and "error" not in stats else 0
    st.markdown(
        '<div class="stat-strip">'
        '<div class="stat-chip"><div class="sv">' + "{:,}".format(v_count) + '</div><div class="sl">Clauses Indexed</div></div>'
        '<div class="stat-chip"><div class="sv">' + str(d_count)           + '</div><div class="sl">Legal Domains</div></div>'
        '<div class="stat-chip"><div class="sv">4</div><div class="sl">Active Agents</div></div>'
        '<div class="stat-chip"><div class="sv">395</div><div class="sl">CSV Datasets</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Section cards — 5 cards only, no browse dataset section below
    section_defs = [
        ("research",   "purple", "🔍", "Legal Research",    "Ask any legal question. Retrieve, classify, risk-analyse and summarise clauses in seconds."),
        ("classifier", "blue",   "🏷️", "Clause Classifier", "Paste any legal clause and instantly identify its type from 395 categories."),
        ("risk",       "rose",   "⚠️", "Risk Analyzer",     "Detect risky language patterns, flagged phrases and get recommendations."),
        ("dashboard",  "green",  "📊", "Dashboard",         "System stats, vector store info and pipeline architecture overview."),
        ("dataset",    "amber",  "📂", "Dataset Browser",   "Browse all 395 legal clause datasets. View and search full clause collections."),
    ]

    card_cols = st.columns(len(section_defs))
    for col, (pg, color, icon, title, desc) in zip(card_cols, section_defs):
        with col:
            st.markdown(
                '<div class="section-card ' + color + '">'
                '<span class="card-icon">' + icon + '</span>'
                '<div class="card-title">' + title + '</div>'
                '<div class="card-desc">'  + desc  + '</div>'
                '<div class="card-arrow">→</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            if st.button("Open " + title, key="card_" + pg, use_container_width=True, type="primary"):
                go(pg)


# =============================================================================
# RESEARCH PAGE
# =============================================================================
elif st.session_state.page == "research":

    st.markdown(
        '<div class="page-title-bar">'
        '<div class="ptb-icon" style="background:linear-gradient(135deg,#7c3aed,#4f46e5)">'
        '<span style="color:white">🔍</span></div>'
        '<div><h2>Legal Research</h2>'
        '<p>Full multi-agent pipeline: retrieve → classify → risk → summarise</p></div></div>',
        unsafe_allow_html=True,
    )

    domains_data = api_get("/domains") if alive else {}
    all_domains  = ["(all domains)"] + (domains_data.get("domains", []) if domains_data and "error" not in domains_data else [])

    col_q, col_d = st.columns([3, 1])
    with col_q:
        question = st.text_area("Research question",
            placeholder="e.g. What are typical indemnification clauses in bank account agreements?", height=120)
    with col_d:
        domain_sel = st.selectbox("Domain filter", all_domains)
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Analyse", type="primary", use_container_width=True, disabled=not alive)

    with st.expander("💡 Example queries"):
        examples = [
            "What are typical indemnification clauses in bank account agreements?",
            "Find termination clauses that allow unilateral cancellation",
            "What does a force majeure clause look like in employment contracts?",
            "Show me confidentiality clauses with broad IP assignment language",
            "What payment terms are common in service agreements?",
        ]
        ec = st.columns(2)
        for i, ex in enumerate(examples):
            if ec[i % 2].button("↗ " + ex, key="ex_" + str(i)):
                st.session_state["clause_prefill"] = ex
                st.rerun()

    if st.session_state.get("clause_prefill") and not question:
        question = st.session_state.pop("clause_prefill")

    if run_btn and question.strip():
        domain_filter = None if domain_sel == "(all domains)" else domain_sel
        ph = st.empty()
        prog = st.progress(0)

        def show_steps(active: int, done: int):
            labels = ["Routing", "Retrieving", "Classifying", "Risk Check", "Summarising"]
            html = '<div class="step-row">'
            for i, lbl in enumerate(labels):
                cls  = "done" if i < done else ("active" if i == active else "")
                icon = "✓"   if i < done else ("●" if i == active else str(i+1))
                html += '<div class="step ' + cls + '">' + icon + " " + lbl + "</div>"
                if i < len(labels)-1: html += '<span class="step-arrow">›</span>'
            html += "</div>"
            ph.markdown(html, unsafe_allow_html=True)

        show_steps(0,0); prog.progress(5,  "Routing…");     time.sleep(0.2)
        show_steps(1,1); prog.progress(20, "Retrieving…")
        result = api_post("/query", {"question": question.strip(), "domain_filter": domain_filter}, timeout=180)
        show_steps(2,2); prog.progress(55, "Classifying…"); time.sleep(0.2)
        show_steps(3,3); prog.progress(75, "Risk check…");  time.sleep(0.2)
        show_steps(4,4); prog.progress(95, "Summarising…"); time.sleep(0.2)
        prog.progress(100, "Done!"); time.sleep(0.3)
        prog.empty(); ph.empty()

        if result and "error" not in result:
            st.markdown("#### 📋 Research Summary")
            final = result.get("final_summary", "")
            if final:
                st.markdown('<div class="rcard purple"><div class="rb">' + final + "</div></div>", unsafe_allow_html=True)
            else:
                st.info("No summary generated.")

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                cls_label = result.get("classification", "unknown")
                cmap = {"indemnification":"#7c3aed","termination":"#f97316","confidentiality":"#8b5cf6",
                        "liability":"#ec4899","payment":"#10b981","governing_law":"#0ea5e9",
                        "force_majeure":"#f59e0b","warranty":"#3b82f6","dispute_resolution":"#f43f5e",
                        "intellectual_property":"#a855f7"}
                fg = cmap.get(cls_label.lower(), "#6366f1")
                st.markdown(
                    '<div class="rcard blue"><div class="rt">🏷️ Clause Type</div>'
                    '<div class="rv" style="color:' + fg + '">' + cls_label.upper() + "</div></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                risk    = parse_risk(result.get("risk_analysis", "{}"))
                level   = risk.get("risk_level", "unknown")
                factors = risk.get("risk_factors", [])
                rec     = risk.get("recommendation", "")
                rcls    = {"low":"green","medium":"amber","high":"rose"}.get(level,"amber")
                rbcls   = level if level in ("low","medium","high") else "medium"
                icons   = {"low":"✅","medium":"⚠️","high":"🚨"}
                f_html  = "".join("<li>" + f + "</li>" for f in factors) if factors else "<li>None identified</li>"
                rec_html = "<p style='font-size:0.82rem;color:#6b7280;margin-top:0.5rem'><strong>Recommendation:</strong> " + rec + "</p>" if rec else ""
                st.markdown(
                    '<div class="rcard ' + rcls + '">'
                    '<div class="rt">⚠️ Risk &nbsp;<span class="rbadge ' + rbcls + '">'
                    + icons.get(level,"❓") + " " + level.upper() + " RISK</span></div>"
                    '<ul style="font-size:0.86rem;color:#374151;margin:0.5rem 0 0 1rem;padding:0">'
                    + f_html + "</ul>" + rec_html + "</div>",
                    unsafe_allow_html=True,
                )
            with st.expander("📄 View Retrieved Clauses"):
                retrieved = result.get("retrieved_clauses","")
                if retrieved:
                    for chunk in retrieved.split("---"):
                        c = chunk.strip()
                        if c: st.markdown('<div class="clause-box">' + c + "</div>", unsafe_allow_html=True)
                else:
                    st.info("No clauses retrieved.")
        else:
            err = result.get("error","Unknown error") if result else "No response."
            st.error("**Error:** " + err)
    elif run_btn:
        st.warning("Please enter a research question.")


# =============================================================================
# CLASSIFIER PAGE  ← completely redesigned
# =============================================================================
elif st.session_state.page == "classifier":

    st.markdown(
        '<div class="page-title-bar">'
        '<div class="ptb-icon" style="background:linear-gradient(135deg,#2563eb,#0ea5e9)">'
        '<span style="color:white">🏷️</span></div>'
        '<div><h2>Clause Classifier</h2>'
        '<p>Paste any legal clause and identify its type from 395 categories</p></div></div>',
        unsafe_allow_html=True,
    )

    # ── Classifier input ──────────────────────────────────────────────────────
    clause_input = st.text_area(
        "Paste your clause here",
        value=st.session_state.get("clause_prefill", ""),
        placeholder="Paste any legal clause text to classify…",
        height=160,
    )
    if st.session_state.get("clause_prefill"):
        st.session_state.clause_prefill = ""

    classify_btn = st.button("🏷️ Classify Clause", type="primary", disabled=not alive)

    if classify_btn:
        if clause_input.strip():
            with st.spinner("Classifying…"):
                result = api_post("/classify", {"clause_text": clause_input.strip()})
            if result and "error" not in result:
                label = result.get("classification", "unknown")
                cmap = {
                    "indemnification":"#7c3aed","termination":"#f97316","confidentiality":"#8b5cf6",
                    "liability":"#ec4899","payment":"#10b981","governing_law":"#0ea5e9",
                    "force_majeure":"#f59e0b","warranty":"#3b82f6","dispute_resolution":"#f43f5e",
                    "intellectual_property":"#a855f7","non_solicitation":"#14b8a6",
                    "severance":"#6366f1","assignment":"#8b5cf6","amendment":"#f59e0b",
                    "non_compete":"#0d9488","data_protection":"#2563eb","representations":"#9333ea",
                }
                fg = cmap.get(label.lower(), "#6366f1")
                st.markdown(
                    '<div style="text-align:center;background:linear-gradient(135deg,#f5f3ff,#ede9fe);'
                    'border:2px solid #c4b5fd;border-radius:20px;padding:2.5rem;margin-bottom:1.5rem">'
                    '<div style="font-size:0.75rem;font-weight:700;color:#6b7280;'
                    'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem">Detected Clause Type</div>'
                    '<div style="font-size:2.8rem;font-weight:900;color:' + fg + ';letter-spacing:-1px">'
                    + label.upper().replace("_"," ") + '</div>'
                    '<div style="margin-top:1rem;font-size:0.82rem;color:#6b7280">'
                    'Matched from 150,000+ indexed clauses across 395 legal domains</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.error(result.get("error","Error") if result else "No response")
        else:
            st.warning("Please paste a clause first.")

    st.markdown("---")

    # ── Sample dataset tabs ───────────────────────────────────────────────────
    st.markdown("### 📚 Sample Datasets & Audience Guide")
    st.caption("Explore what each dataset contains, who it is designed for, and what questions you can answer with it.")

    tab_labels = [ds["tab"] for ds in SAMPLE_DATASETS]
    tabs       = st.tabs(tab_labels)

    for tab, ds in zip(tabs, SAMPLE_DATASETS):
        with tab:

            # Dataset title + description
            st.markdown(
                '<div style="background:linear-gradient(135deg,#7c3aed,#4f46e5);'
                'border-radius:16px;padding:1.4rem 1.8rem;margin-bottom:1.2rem;color:white">'
                '<div style="font-size:1.1rem;font-weight:800;margin-bottom:0.3rem">' + ds["title"] + '</div>'
                '<div style="font-size:0.85rem;opacity:0.9">' + ds["desc"] + '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            # Sample clause with load button
            st.markdown("**Sample Clause:**")
            st.markdown(
                '<div class="sample-clause">' + ds["sample_clause"] + '</div>',
                unsafe_allow_html=True,
            )
            if st.button("↗ Load this sample clause", key="load_" + ds["tab"], type="secondary"):
                st.session_state.clause_prefill = ds["sample_clause"]
                st.rerun()

            st.markdown("---")
            st.markdown("#### 👥 Who is this dataset for?")

            # Audience cards — 2 per row
            audiences = ds["audiences"]
            for i in range(0, len(audiences), 2):
                row_auds = audiences[i:i+2]
                cols     = st.columns(len(row_auds))
                for col, aud in zip(cols, row_auds):
                    with col:
                        # Use cases as bullet list
                        uc_html = "".join("<li>" + u + "</li>" for u in aud["use_cases"])
                        # Questions as pills
                        q_html  = "".join('<span class="q-pill">"' + q + '"</span>' for q in aud["questions"])

                        st.markdown(
                            '<div class="audience-card" style="background:' + aud["bg"] + ';border-color:' + aud["border"] + '">'

                            # Persona title
                            '<div class="ac-title" style="color:' + aud["color"] + '">'
                            + aud["persona"] + '</div>'

                            # Why relevant
                            '<div class="ac-label" style="color:' + aud["color"] + '">Why Relevant</div>'
                            '<p style="font-size:0.84rem;color:#374151;margin:0 0 0.8rem;line-height:1.5">'
                            + aud["relevance"] + '</p>'

                            # Use cases
                            '<div class="ac-label" style="color:' + aud["color"] + '">Example Use Cases</div>'
                            '<ul>' + uc_html + '</ul>'

                            # Sample questions
                            '<div class="ac-label" style="color:' + aud["color"] + ';margin-top:0.8rem">'
                            'Questions You Can Ask</div>'
                            '<div style="margin-top:4px">' + q_html + '</div>'

                            '</div>',
                            unsafe_allow_html=True,
                        )


# =============================================================================
# RISK PAGE
# =============================================================================
elif st.session_state.page == "risk":

    st.markdown(
        '<div class="page-title-bar">'
        '<div class="ptb-icon" style="background:linear-gradient(135deg,#e11d48,#f97316)">'
        '<span style="color:white">⚠️</span></div>'
        '<div><h2>Risk Analyzer</h2>'
        '<p>Detect risky language, flagged phrases and get plain-English recommendations</p></div></div>',
        unsafe_allow_html=True,
    )

    risk_input = st.text_area("Paste clause to analyze", placeholder="Paste any legal clause…", height=200)
    risk_btn   = st.button("⚠️ Analyse Risk", type="primary", disabled=not alive)

    if risk_btn:
        if risk_input.strip():
            with st.spinner("Analysing…"):
                result = api_post("/risk", {"clause_text": risk_input.strip()})
            if result and "error" not in result:
                risk    = parse_risk(result.get("risk_analysis","{}"))
                level   = risk.get("risk_level","unknown")
                factors = risk.get("risk_factors",[])
                rec     = risk.get("recommendation","")
                flagged = risk.get("flagged_phrases",[])

                level_cfg = {
                    "low":    ("#15803d","#f0fdf4","#86efac","🟢","Low Risk — Clause appears standard."),
                    "medium": ("#a16207","#fefce8","#fde047","🟡","Medium Risk — Review carefully before signing."),
                    "high":   ("#b91c1c","#fff1f2","#fca5a5","🔴","High Risk — Seek legal advice before proceeding."),
                }
                fg,bg,bdr,icon,msg = level_cfg.get(level,("#6b7280","#f9fafb","#d1d5db","⚪","Risk level unknown."))
                st.markdown(
                    '<div style="background:' + bg + ';border:2px solid ' + bdr + ';'
                    'border-radius:16px;padding:1.4rem 1.8rem;margin-bottom:1.2rem;'
                    'display:flex;align-items:center;gap:16px">'
                    '<div style="font-size:2.5rem">' + icon + '</div>'
                    '<div><div style="font-size:1.5rem;font-weight:900;color:' + fg + '">' + level.upper() + " RISK</div>"
                    '<div style="font-size:0.88rem;color:' + fg + ';opacity:0.8">' + msg + '</div></div></div>',
                    unsafe_allow_html=True,
                )
                r1,r2 = st.columns(2)
                with r1:
                    if factors:
                        st.markdown("**⚠️ Risk Factors**")
                        for f in factors:
                            st.markdown(
                                '<div style="background:#fff7ed;border-left:4px solid #f97316;'
                                'border-radius:0 8px 8px 0;padding:8px 12px;margin-bottom:6px;'
                                'font-size:0.87rem;color:#431407">⚡ ' + f + '</div>',
                                unsafe_allow_html=True,
                            )
                with r2:
                    if flagged:
                        st.markdown("**🔎 Flagged Phrases**")
                        for phrase in flagged:
                            st.markdown(
                                '<div style="background:#fff1f2;border-left:4px solid #dc2626;'
                                'border-radius:0 8px 8px 0;padding:8px 12px;margin-bottom:6px;'
                                'font-size:0.85rem;color:#7f1d1d;font-style:italic">"' + phrase + '"</div>',
                                unsafe_allow_html=True,
                            )
                if rec:
                    st.markdown(
                        '<div style="background:linear-gradient(135deg,#eff6ff,#dbeafe);'
                        'border:1px solid #93c5fd;border-radius:12px;padding:1rem 1.3rem;margin-top:1rem">'
                        '<div style="font-size:0.72rem;font-weight:700;color:#1d4ed8;'
                        'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px">💡 Recommendation</div>'
                        '<div style="font-size:0.92rem;color:#1e3a5f">' + rec + '</div></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.error(result.get("error","Error") if result else "No response")
        else:
            st.warning("Please paste a clause first.")


# =============================================================================
# DASHBOARD PAGE
# =============================================================================
elif st.session_state.page == "dashboard":

    st.markdown(
        '<div class="page-title-bar">'
        '<div class="ptb-icon" style="background:linear-gradient(135deg,#059669,#0d9488)">'
        '<span style="color:white">📊</span></div>'
        '<div><h2>Dashboard</h2><p>System overview and pipeline architecture</p></div></div>',
        unsafe_allow_html=True,
    )

    if alive:
        stats = api_get("/stats")
        if stats and "error" not in stats:
            s1,s2,s3,s4 = st.columns(4)
            v_fmt = "{:,}".format(stats.get("total_vectors",0))
            d_fmt = str(stats.get("total_domains",0))
            for col,bg,icon,val,lbl in [
                (s1,"linear-gradient(135deg,#7c3aed,#4f46e5)","📦",v_fmt,"Clauses Indexed"),
                (s2,"linear-gradient(135deg,#0ea5e9,#06b6d4)","🗂️",d_fmt,"Legal Domains"),
                (s3,"linear-gradient(135deg,#ec4899,#f43f5e)","🤖","4","Active Agents"),
                (s4,"linear-gradient(135deg,#10b981,#059669)","✅","Online","System Status"),
            ]:
                col.markdown(
                    '<div style="background:' + bg + ';border-radius:16px;padding:1.3rem;text-align:center;color:white">'
                    '<div style="font-size:1.6rem">' + icon + '</div>'
                    '<div style="font-size:1.8rem;font-weight:800">' + val + '</div>'
                    '<div style="font-size:0.72rem;opacity:0.85;text-transform:uppercase;letter-spacing:0.08em;margin-top:2px">' + lbl + '</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("---")
        st.markdown("#### 🏛️ Agent Pipeline")
        p1,p2,p3,p4,p5 = st.columns(5)
        for col,color,ico,title,desc in [
            (p1,"#7c3aed","🧭","Supervisor",  "Routes queries to the right agent path"),
            (p2,"#4f46e5","🔎","Retriever",   "Searches 150K+ clauses by meaning"),
            (p3,"#0ea5e9","🏷️","Classifier",  "Labels clause type using AI"),
            (p4,"#f97316","⚠️","Risk Agent",  "Detects risky language patterns"),
            (p5,"#10b981","📋","Summariser",  "Generates plain-English summary"),
        ]:
            col.markdown(
                '<div style="background:' + color + '12;border:1.5px solid ' + color + '44;'
                'border-radius:14px;padding:1.1rem;text-align:center">'
                '<div style="font-size:1.8rem">' + ico + '</div>'
                '<div style="font-size:0.88rem;font-weight:700;color:' + color + ';margin:6px 0 4px">' + title + '</div>'
                '<div style="font-size:0.72rem;color:#6b7280">' + desc + '</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.warning("Start the backend to view dashboard statistics.")
        st.code("uvicorn app.main:app --reload --port 8000", language="bash")


# =============================================================================
# DATASET BROWSER PAGE
# =============================================================================
elif st.session_state.page == "dataset":

    csv_files = get_csv_files()

    if st.session_state.dataset_file:
        fpath = Path(st.session_state.dataset_file)
        st.markdown(
            '<div class="dataset-header"><h3>📂 ' + fmt_domain(fpath.stem) + '</h3>'
            '<p>' + fpath.name + ' · Full clause dataset</p></div>',
            unsafe_allow_html=True,
        )
        if st.button("← All Datasets", key="back_ds"):
            st.session_state.dataset_file = None
            st.rerun()

        df = load_csv(fpath)
        if df.empty:
            st.error("Could not load file or file is empty.")
        else:
            m1,m2,m3 = st.columns(3)
            m1.metric("Total Clauses", len(df))
            m2.metric("Columns", len(df.columns))
            if "clause_type" in df.columns:
                m3.metric("Clause Types", df["clause_type"].nunique())
            st.markdown("---")

            search_ds = st.text_input("🔍 Search within dataset", placeholder="Filter by keyword…", label_visibility="collapsed")

            text_col = None
            for c in ["clause_text","text","clause","content","description","sentence"]:
                if c in df.columns: text_col = c; break
            if not text_col:
                for c in df.columns:
                    if df[c].dtype == object: text_col = c; break

            df_show = df[df[text_col].astype(str).str.contains(search_ds, case=False, na=False)] if (text_col and search_ds) else df
            st.caption("Showing " + str(len(df_show)) + " of " + str(len(df)) + " clauses")

            if text_col:
                type_col = "clause_type" if "clause_type" in df_show.columns else None
                colors   = ["#7c3aed","#2563eb","#059669","#d97706","#db2777","#0d9488"]
                for i, (_, row) in enumerate(df_show.iterrows()):
                    clause_text = str(row[text_col])
                    clause_type = str(row[type_col]) if type_col else ""
                    c = colors[i % len(colors)]
                    with st.expander("Clause " + str(i+1) + (" — " + clause_type if clause_type else ""), expanded=(i==0)):
                        st.markdown(
                            '<div style="background:#fafafa;border-left:4px solid ' + c + ';'
                            'border-radius:0 10px 10px 0;padding:1rem 1.2rem;'
                            'font-size:0.88rem;color:#374151;line-height:1.7">' + clause_text + '</div>',
                            unsafe_allow_html=True,
                        )
                        if st.button("🏷️ Classify this clause", key="cls_direct_" + str(i)):
                            st.session_state.page           = "classifier"
                            st.session_state.clause_prefill = clause_text
                            st.rerun()
            else:
                st.dataframe(df_show, use_container_width=True)

    else:
        st.markdown(
            '<div class="page-title-bar">'
            '<div class="ptb-icon" style="background:linear-gradient(135deg,#d97706,#f59e0b)">'
            '<span style="color:white">📂</span></div>'
            '<div><h2>Dataset Browser</h2>'
            '<p>All 395 legal clause datasets — click any domain to view its full clause collection</p></div></div>',
            unsafe_allow_html=True,
        )

        search_all = st.text_input("🔍 Search domains", placeholder="Type to filter…", label_visibility="collapsed")
        filtered   = [f for f in csv_files if search_all.lower() in f.stem.lower()] if search_all else csv_files
        st.caption(str(len(filtered)) + " domains found")

        cols_per_row = 5
        rows = [filtered[i:i+cols_per_row] for i in range(0, len(filtered), cols_per_row)]
        for row in rows:
            btn_cols = st.columns(len(row))
            for col, csvf in zip(btn_cols, row):
                with col:
                    if st.button("📄 " + fmt_domain(csvf.stem), key="ds_btn_" + csvf.stem,
                                 use_container_width=True, type="secondary"):
                        st.session_state.dataset_file = str(csvf)
                        st.rerun()
