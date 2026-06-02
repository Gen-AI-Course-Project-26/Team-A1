import streamlit as st
import os
import json
import numpy as np
import requests
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN     = os.getenv("HF_TOKEN")

VECTOR_DIR   = "data/vectorstore"
MODEL        = "llama-3.3-70b-versatile"
HF_MODEL_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

DOC_NAMES = {
    "constitution": "Constitution of India",
    "ipc":          "Indian Penal Code",
    "crpc":         "Code of Criminal Procedure",
    "bnss":         "BNSS 2023",
}

DOC_TAGS = {
    "constitution": "CONST",
    "ipc":          "IPC",
    "crpc":         "CrPC",
    "bnss":         "BNSS",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexRAG — Indian Legal Intelligence",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0a0a !important;
    color: #e8e8e8 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] > .main {
    background-color: #0a0a0a !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0f0f0f !important;
    border-right: 1px solid #1e1e1e !important;
}
[data-testid="stSidebar"] * { color: #c8c8c8 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 13px !important; }
[data-testid="stSidebar"] .stButton button {
    background: #141414 !important;
    border: 1px solid #222 !important;
    color: #aaa !important;
    font-size: 12px !important;
    text-align: left !important;
    border-radius: 4px !important;
    padding: 6px 10px !important;
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stButton button:hover {
    border-color: #c9a84c !important;
    color: #c9a84c !important;
    background: #1a1a14 !important;
}

/* ── Main text input ── */
[data-testid="stTextInput"] input {
    background: #111 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px !important;
    color: #f0f0f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    padding: 14px 16px !important;
    transition: border-color 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: #c9a84c !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.1) !important;
}
[data-testid="stTextInput"] input::placeholder { color: #444 !important; }
[data-testid="stTextInput"] label { color: #666 !important; font-size: 11px !important; }

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: #c9a84c !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    background: #e0bb5a !important;
    transform: translateY(-1px) !important;
}

/* ── Toggle ── */
[data-testid="stToggle"] label { color: #888 !important; font-size: 13px !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #c9a84c !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #111 !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 4px !important;
    margin-bottom: 8px !important;
}
[data-testid="stExpander"] summary {
    color: #c8c8c8 !important;
    font-size: 13px !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stExpander"] summary:hover { color: #c9a84c !important; }

/* ── Info / warning boxes ── */
[data-testid="stInfo"] {
    background: #111 !important;
    border: 1px solid #c9a84c44 !important;
    color: #c9a84c !important;
    border-radius: 4px !important;
}
[data-testid="stWarning"] {
    background: #111 !important;
    border: 1px solid #555 !important;
    color: #888 !important;
    border-radius: 4px !important;
}

/* ── Divider ── */
hr { border-color: #1e1e1e !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #222; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #333; }
</style>
""", unsafe_allow_html=True)

# ── Backend ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_vectorstore():
    matrix   = np.load(os.path.join(VECTOR_DIR, "embeddings.npy"))
    metadata = json.load(open(os.path.join(VECTOR_DIR, "metadata.json"), encoding="utf-8"))
    return matrix, metadata

def get_embedding(text):
    headers  = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload  = {"inputs": [text], "options": {"wait_for_model": True}}
    response = requests.post(HF_MODEL_URL, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        raise Exception(f"Embedding error: {response.text}")
    result = response.json()
    if isinstance(result, list) and isinstance(result[0], list):
        return np.array(result[0], dtype=np.float32)
    raise Exception("Unexpected embedding response format")

def search(query, matrix, metadata, top_k=5, filter_doc=None):
    query_vec   = get_embedding(query)
    query_norm  = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    scores      = matrix_norm @ query_norm
    top_idx     = np.argsort(scores)[::-1]
    results = []
    for i in top_idx:
        if filter_doc and metadata[i]["doc_key"] != filter_doc:
            continue
        results.append({**metadata[i], "relevance": round(float(scores[i]), 3)})
        if len(results) >= top_k:
            break
    return results

def build_prompt(query, sections, simple_language=False):
    context = ""
    for i, sec in enumerate(sections, 1):
        context += f"\n[Source {i}] {DOC_NAMES.get(sec['doc_key'], sec['doc_key'].upper())}"
        context += f" — Section/Article {sec['section_number']}\n"
        context += f"{sec['text'][:800]}\n"
        context += "-" * 40 + "\n"
    tone = (
        "Explain in simple plain English anyone can understand. Avoid legal jargon. Use short sentences."
        if simple_language else
        "Provide a precise legal answer. Reference specific sections and articles directly."
    )
    return f"""You are LexRAG, an expert Indian legal assistant.
Answer using ONLY the legal sections provided below.
Do NOT use any knowledge outside the provided sources.
Always cite which Section/Article your answer comes from.
If the sections do not contain enough information, say so clearly.

{tone}

--- LEGAL SOURCES ---
{context}
--- END OF SOURCES ---

User Question: {query}

Answer:"""

def call_groq(prompt):
    client   = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.1,
    )
    return response.choices[0].message.content

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 48px 0 32px 0; border-bottom: 1px solid #1e1e1e; margin-bottom: 40px;">
    <div style="display:flex; align-items:baseline; gap:16px; margin-bottom:8px;">
        <span style="font-family:'Playfair Display',serif; font-size:42px; font-weight:700;
                     color:#f0f0f0; letter-spacing:-0.02em;">LexRAG</span>
        <span style="font-family:'DM Mono',monospace; font-size:11px; color:#c9a84c;
                     letter-spacing:0.15em; text-transform:uppercase;
                     border:1px solid #c9a84c44; padding:3px 8px; border-radius:2px;">
            BETA
        </span>
    </div>
    <p style="font-family:'DM Sans',sans-serif; font-size:15px; color:#555;
              letter-spacing:0.04em; text-transform:uppercase; font-weight:300;">
        Indian Legal Intelligence System &nbsp;·&nbsp;
        Constitution &nbsp;·&nbsp; IPC &nbsp;·&nbsp; CrPC &nbsp;·&nbsp; BNSS 2023
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:24px 0 16px 0; border-bottom:1px solid #1e1e1e; margin-bottom:20px;">
        <span style="font-family:'DM Mono',monospace; font-size:10px;
                     color:#c9a84c; letter-spacing:0.2em; text-transform:uppercase;">
            ⚖ LexRAG
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:10px; color:#444; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:8px;">Mode</p>', unsafe_allow_html=True)
    simple_mode = st.toggle("Plain English Mode", value=False)

    st.markdown('<div style="border-top:1px solid #1e1e1e; margin:20px 0;"></div>', unsafe_allow_html=True)

    st.markdown('<p style="font-size:10px; color:#444; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:8px;">Source Filter</p>', unsafe_allow_html=True)
    filter_doc = st.radio(
        "", 
        options=["All", "Constitution", "IPC", "CrPC", "BNSS 2023"],
        index=0,
        label_visibility="collapsed",
    )
    doc_filter_map = {
        "All": None, "Constitution": "constitution",
        "IPC": "ipc", "CrPC": "crpc", "BNSS 2023": "bnss",
    }
    selected_doc = doc_filter_map[filter_doc]

    st.markdown('<div style="border-top:1px solid #1e1e1e; margin:20px 0;"></div>', unsafe_allow_html=True)

    st.markdown('<p style="font-size:10px; color:#444; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:12px;">Quick Search</p>', unsafe_allow_html=True)
    samples = [
        "Punishment for murder",
        "Fundamental rights",
        "Arrest without warrant",
        "Bail procedure",
        "Right to life Article 21",
        "Culpable homicide",
        "Rights of arrested person",
        "Death penalty provisions",
    ]
    for s in samples:
        if st.button(s, use_container_width=True, key=f"btn_{s}"):
            st.session_state["sample_query"] = s

    st.markdown('<div style="border-top:1px solid #1e1e1e; margin:20px 0;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'DM Mono',monospace; font-size:11px; color:#333; line-height:2;">
        <div>CONST &nbsp; 411 articles</div>
        <div>IPC &nbsp;&nbsp;&nbsp; 534 sections</div>
        <div>CrPC &nbsp;&nbsp; 525 sections</div>
        <div>BNSS &nbsp;&nbsp; 329 sections</div>
        <div style="color:#555; margin-top:8px; padding-top:8px; border-top:1px solid #1e1e1e;">
            TOTAL &nbsp; 1,799 indexed
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
matrix, metadata = load_vectorstore()

# ── Search bar ────────────────────────────────────────────────────────────────
default_query = st.session_state.get("sample_query", "")

col_input, col_btn = st.columns([6, 1])
with col_input:
    query = st.text_input(
        "Query",
        value=default_query,
        placeholder="Ask a legal question — e.g. What is the punishment for theft under IPC?",
        label_visibility="collapsed",
    )
with col_btn:
    search_btn = st.button("Search", type="primary", use_container_width=True)

if simple_mode:
    st.markdown('<p style="font-size:12px; color:#c9a84c; margin-top:4px;">● Plain English Mode active</p>', unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:32px;'></div>", unsafe_allow_html=True)

# ── Results ───────────────────────────────────────────────────────────────────
if search_btn and query.strip():
    st.session_state["sample_query"] = ""

    with st.spinner("Searching legal database..."):
        try:
            sections = search(query, matrix, metadata, top_k=5, filter_doc=selected_doc)
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    with st.spinner("Generating answer..."):
        try:
            prompt = build_prompt(query, sections, simple_mode)
            answer = call_groq(prompt)
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.stop()

    # Answer block
    st.markdown("""
    <p style="font-family:'DM Mono',monospace; font-size:10px; color:#c9a84c;
              letter-spacing:0.2em; text-transform:uppercase; margin-bottom:12px;">
        ● Analysis
    </p>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#111; border:1px solid #1e1e1e; border-left: 3px solid #c9a84c;
                padding:28px 32px; border-radius:4px; margin-bottom:40px;">
        <p style="font-family:'DM Sans',sans-serif; font-size:15px; color:#e0e0e0;
                  line-height:1.8; white-space:pre-wrap;">{answer}</p>
    </div>
    """, unsafe_allow_html=True)

    # Sources
    st.markdown("""
    <p style="font-family:'DM Mono',monospace; font-size:10px; color:#444;
              letter-spacing:0.2em; text-transform:uppercase; margin-bottom:16px;">
        Sources Retrieved
    </p>
    """, unsafe_allow_html=True)

    for sec in sections:
        tag      = DOC_TAGS.get(sec["doc_key"], "DOC")
        doc_name = DOC_NAMES.get(sec["doc_key"], sec["doc_key"].upper())
        pct      = int(sec["relevance"] * 100)
        bar_w    = pct

        with st.expander(f"[{tag}] § {sec['section_number']}  —  {doc_name}  ·  {pct}% match"):
            st.markdown(f"""
            <div style="margin-bottom:12px;">
                <div style="background:#1a1a1a; border-radius:2px; height:2px; margin-bottom:16px;">
                    <div style="background:#c9a84c; height:2px; width:{bar_w}%; border-radius:2px;"></div>
                </div>
                <p style="font-family:'DM Mono',monospace; font-size:11px;
                          color:#c9a84c; margin-bottom:12px; letter-spacing:0.05em;">
                    {sec['section_title'][:100]}
                </p>
                <p style="font-family:'DM Sans',sans-serif; font-size:13px;
                          color:#888; line-height:1.75; white-space:pre-wrap;">
                    {sec['text'][:800]}
                </p>
            </div>
            """, unsafe_allow_html=True)

elif search_btn:
    st.markdown('<p style="color:#444; font-size:13px;">Enter a legal question to search.</p>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid #141414; margin-top:80px; padding-top:24px;">
    <p style="font-family:'DM Mono',monospace; font-size:10px; color:#2a2a2a;
              letter-spacing:0.15em; text-align:center;">
        LEXRAG &nbsp;·&nbsp; INDIAN LEGAL INTELLIGENCE &nbsp;·&nbsp;
        POWERED BY LLAMA 3.3 &amp; SENTENCE TRANSFORMERS
    </p>
</div>
""", unsafe_allow_html=True)