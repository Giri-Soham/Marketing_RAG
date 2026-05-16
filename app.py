import streamlit as st
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from retriever import load_index, build_bm25, retrieve
from re_ranker import load_reranker, rerank
from rag_pipeline import ask, EMBED_MODEL, RERANK_MODEL, INDEX_PATH, CHUNKS_PATH

load_dotenv()

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MarketingRAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=DM+Mono:wght@400;500&display=swap');

/* ── reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #0a0c10 !important;
    color: #e2e8f0 !important;
}

/* ── hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #0f1117 !important;
    border-right: 1px solid #1e2433 !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}

/* ── chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 0 !important;
    border: none !important;
}
[data-testid="stChatMessage"] p {
    font-size: 14px !important;
    line-height: 1.75 !important;
    color: #e2e8f0 !important;
}

/* ── chat input ── */
[data-testid="stChatInput"] {
    background-color: #0f1117 !important;
    border-top: 1px solid #1e2433 !important;
    padding: 16px 24px !important;
}
[data-testid="stChatInput"] textarea {
    background-color: #161c2a !important;
    border: 1px solid #2d3748 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    caret-color: #818cf8 !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,0.12) !important;
}
[data-testid="stChatInput"] button {
    background: #818cf8 !important;
    border-radius: 8px !important;
    color: white !important;
}

/* ── buttons in sidebar ── */
.stButton > button {
    width: 100% !important;
    background: transparent !important;
    border: 1px solid #1e2433 !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    padding: 7px 12px !important;
    text-align: left !important;
    transition: all 0.15s ease !important;
    margin-bottom: 4px !important;
}
.stButton > button:hover {
    background: #161c2a !important;
    border-color: #818cf8 !important;
    color: #e2e8f0 !important;
}

/* ── clear button ── */
.clear-btn > button {
    background: transparent !important;
    border: 1px solid #2d3748 !important;
    color: #64748b !important;
    font-size: 11px !important;
    padding: 6px 12px !important;
    border-radius: 6px !important;
    text-align: center !important;
}
.clear-btn > button:hover {
    background: #1e2433 !important;
    border-color: #f87171 !important;
    color: #f87171 !important;
}

/* ── divider ── */
hr {
    border: none !important;
    border-top: 1px solid #1e2433 !important;
    margin: 14px 0 !important;
}

/* ── expander ── */
[data-testid="stExpander"] {
    background: #161c2a !important;
    border: 1px solid #2d3748 !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    color: #94a3b8 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 10px 14px !important;
    cursor: pointer !important;
    list-style: none !important;
    user-select: none !important;
}
[data-testid="stExpander"] summary:hover { color: #e2e8f0 !important; }
[data-testid="stExpander"] summary::marker,
[data-testid="stExpander"] summary::-webkit-details-marker { display: none !important; }
[data-testid="stExpander"] details[open] summary {
    border-bottom: 1px solid #2d3748 !important;
}

/* ── welcome card buttons ── */
.welcome-btn > button {
    background: #0f1117 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    padding: 10px 16px !important;
    text-align: center !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
.welcome-btn > button:hover {
    background: #161c2a !important;
    border-color: #818cf8 !important;
    color: #e2e8f0 !important;
    transform: translateY(-1px) !important;
}

/* ── scrollbar ── */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #4a5568; }

/* ── spinner ── */
.stSpinner > div { border-top-color: #818cf8 !important; }

/* ── main chat area padding ── */
.main-chat { padding: 24px 40px; max-width: 860px; margin: 0 auto; }

</style>
""", unsafe_allow_html=True)

# ── SESSION STATE INIT ────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ── LOAD PIPELINE (cached — runs once only) ───────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_pipeline():
    from groq import Groq
    client        = Groq(api_key=os.getenv("GROQ_API_KEY"))
    index, chunks = load_index(INDEX_PATH, CHUNKS_PATH)
    embed_model   = SentenceTransformer(EMBED_MODEL)
    bm25          = build_bm25(chunks)
    rerank_model  = load_reranker(RERANK_MODEL)
    return client, index, chunks, embed_model, bm25, rerank_model

# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_confidence(chunks):
    if not chunks:
        return "Low", "#f87171"
    score = chunks[0].get("rerank_score", 0)
    if score >= 5.0:
        return "High", "#4ade80"
    elif score >= 1.0:
        return "Medium", "#fbbf24"
    return "Low", "#f87171"

def source_colour(source):
    if "Smart Insights" in source or "PDF" in source.lower():
        return "#818cf8"
    elif "Neil Patel" in source or "patel" in source.lower():
        return "#34d399"
    return "#fb923c"

def source_type(source):
    if "Smart Insights" in source:
        return "PDF"
    elif "Neil Patel" in source:
        return "WEB"
    return "CSV"

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:

    # brand header — always visible, never collapses
    st.markdown("""
    <div style="padding:28px 20px 20px; border-bottom:1px solid #1e2433; margin-bottom:12px;">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
            <div style="width:32px; height:32px; border-radius:9px; background:#818cf8;
                        display:flex; align-items:center; justify-content:center;
                        font-size:16px; flex-shrink:0;">📊</div>
            <div>
                <div style="font-family:'Syne',sans-serif; font-size:15px; font-weight:600;
                            color:#f1f5f9; letter-spacing:-0.01em;">MarketingRAG</div>
                <div style="font-size:10px; color:#475569; letter-spacing:0.05em;">AI-powered insight engine</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # collapsible: data sources
    with st.expander("Data Sources", expanded=True):
        st.markdown("""
        <div style="display:flex; flex-direction:column; gap:9px; padding:4px 0;">
            <div style="display:flex; align-items:center; justify-content:space-between;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <div style="width:8px; height:8px; border-radius:2px; background:#818cf8; flex-shrink:0;"></div>
                    <span style="font-size:12px; color:#94a3b8;">Smart Insights PDF</span>
                </div>
                <span style="font-family:'DM Mono',monospace; font-size:11px; color:#475569;
                             background:#161c2a; padding:2px 8px; border-radius:4px;
                             border:1px solid #1e2433;">83</span>
            </div>
            <div style="display:flex; align-items:center; justify-content:space-between;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <div style="width:8px; height:8px; border-radius:2px; background:#34d399; flex-shrink:0;"></div>
                    <span style="font-size:12px; color:#94a3b8;">Neil Patel Articles</span>
                </div>
                <span style="font-family:'DM Mono',monospace; font-size:11px; color:#475569;
                             background:#161c2a; padding:2px 8px; border-radius:4px;
                             border:1px solid #1e2433;">292</span>
            </div>
            <div style="display:flex; align-items:center; justify-content:space-between;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <div style="width:8px; height:8px; border-radius:2px; background:#fb923c; flex-shrink:0;"></div>
                    <span style="font-size:12px; color:#94a3b8;">Campaign CSV</span>
                </div>
                <span style="font-family:'DM Mono',monospace; font-size:11px; color:#475569;
                             background:#161c2a; padding:2px 8px; border-radius:4px;
                             border:1px solid #1e2433;">340</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # collapsible: pipeline
    with st.expander("Pipeline", expanded=True):
        st.markdown("""
        <div style="display:flex; flex-direction:column; gap:8px; padding:4px 0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:12px; color:#64748b;">Embeddings</span>
                <span style="font-family:'DM Mono',monospace; font-size:11px; color:#818cf8;">MiniLM-L6</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:12px; color:#64748b;">Re-ranker</span>
                <span style="font-family:'DM Mono',monospace; font-size:11px; color:#818cf8;">ms-marco</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:12px; color:#64748b;">LLM</span>
                <span style="font-family:'DM Mono',monospace; font-size:11px; color:#818cf8;">Llama-3.3-70B</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:12px; color:#64748b;">Vectors</span>
                <span style="font-family:'DM Mono',monospace; font-size:11px; color:#818cf8;">720</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # suggested questions
    st.markdown("""
    <div style="padding:0 4px 10px;">
        <div style="font-size:10px; color:#475569; letter-spacing:0.08em; font-weight:500;
                    margin-bottom:10px;">TRY ASKING</div>
    </div>
    """, unsafe_allow_html=True)

    suggestions = [
        "Which channel has the highest ROI?",
        "What are the pillars of digital marketing?",
        "Best channel by conversion rate?",
        "How do influencer campaigns perform?",
        "What is content marketing?",
    ]

    for s in suggestions:
        if st.button(s, key=f"sug_{s}"):
            st.session_state.pending_query = s

    st.markdown("<hr>", unsafe_allow_html=True)

    # chat history
    st.markdown("""
    <div style="padding:0 20px 10px;">
        <div style="font-size:10px; color:#475569; letter-spacing:0.08em; font-weight:500;
                    margin-bottom:10px;">CHAT HISTORY</div>
    </div>
    """, unsafe_allow_html=True)

    user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
    if user_msgs:
        for i, msg in enumerate(reversed(user_msgs[-8:])):
            q = msg["content"]
            display = q[:36] + "…" if len(q) > 36 else q
            st.markdown(f"""
            <div style="padding:7px 20px; border-left:2px solid transparent; cursor:default;
                        transition:all 0.15s;" onmouseover="this.style.borderLeftColor='#818cf8';
                        this.style.background='rgba(129,140,248,0.06)'"
                        onmouseout="this.style.borderLeftColor='transparent';
                        this.style.background='transparent'">
                <span style="font-family:'DM Mono',monospace; font-size:10px;
                             color:#334155;">{len(user_msgs)-i:02d}</span>
                <span style="font-size:12px; color:#64748b; margin-left:8px;
                             line-height:1.4;">{display}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding:0 20px;">
            <span style="font-size:12px; color:#334155; font-style:italic;">No history yet.</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # clear chat
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("🗑  Clear conversation", key="clear_chat"):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ── LOAD PIPELINE ─────────────────────────────────────────────────────────────

with st.spinner("Loading models — first run takes ~20 seconds..."):
    client, index, chunks, embed_model, bm25, rerank_model = load_pipeline()

# ── MAIN AREA ─────────────────────────────────────────────────────────────────

# top header bar
st.markdown("""
<div style="background:#0a0c10; border-bottom:1px solid #1e2433;
            padding:14px 40px; display:flex; align-items:center;
            justify-content:space-between; position:sticky; top:0; z-index:100;">
    <div>
        <span style="font-family:'Syne',sans-serif; font-size:17px; font-weight:600;
                     color:#f1f5f9; letter-spacing:-0.02em;">
            Marketing Intelligence Q&amp;A
        </span>
        <span style="font-size:12px; color:#334155; margin-left:14px;">
            Vector search · BM25 · Cross-encoder re-ranking · GPT-3.5
        </span>
    </div>
    <div style="display:flex; align-items:center; gap:16px;">
        <span style="font-family:'DM Mono',monospace; font-size:11px;
                     color:#334155; background:#0f1117; padding:4px 10px;
                     border-radius:6px; border:1px solid #1e2433;">
            720 vectors indexed
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# welcome screen if no messages
if not st.session_state.messages:
    st.markdown("""
    <div style="display:flex; flex-direction:column; align-items:center;
                text-align:center; padding:60px 20px 32px;">
        <div style="font-size:48px; margin-bottom:20px;">📊</div>
        <div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:600;
                    color:#f1f5f9; margin-bottom:12px; letter-spacing:-0.03em;
                    max-width:520px; line-height:1.3;">
            Ask your marketing data anything
        </div>
        <div style="font-size:14px; color:#64748b; max-width:440px;
                    line-height:1.7; margin-bottom:36px;">
            Powered by 720 indexed chunks across a research report, web articles,
            and 200,000 campaign records. Every answer is grounded and cited.
        </div>
    </div>
    """, unsafe_allow_html=True)

    welcome_questions = [
        "Which channel has the highest ROI?",
        "What are digital marketing pillars?",
        "Best channel by conversion rate?",
        "How do influencer campaigns perform?",
        "What is content marketing?",
        "What is the ROI of email marketing?",
    ]
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        cols = st.columns(2)
        for i, q in enumerate(welcome_questions):
            with cols[i % 2]:
                st.markdown('<div class="welcome-btn">', unsafe_allow_html=True)
                if st.button(q, key=f"welcome_{i}"):
                    st.session_state.pending_query = q
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

# ── RENDER CHAT HISTORY ───────────────────────────────────────────────────────
# Key fix: render all previous messages FIRST using st.chat_message
# This prevents duplicate boxes on rerun because Streamlit only
# re-renders what changes — the existing messages are stable.

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(f"""
            <div style="background:#1e1b4b; border:1px solid #3730a3; border-radius:12px;
                        padding:14px 18px; max-width:75%; margin-left:auto;
                        font-size:14px; color:#e2e8f0; line-height:1.6;">
                {msg['content']}
            </div>
            """, unsafe_allow_html=True)

    elif msg["role"] == "assistant":
        result = msg["result"]
        conf_label, conf_color = get_confidence(result["chunks"])
        latency = result["latency"]
        answer  = result["answer"]

        with st.chat_message("assistant", avatar="📊"):

            # answer card
            st.markdown(f"""
            <div style="background:#0f1117; border:1px solid #1e2433;
                        border-radius:2px 14px 14px 14px; padding:18px 22px;
                        max-width:85%; margin-bottom:10px;">
                <div style="font-size:14px; color:#e2e8f0; line-height:1.8;
                            margin-bottom:16px; white-space:pre-wrap;">{answer}</div>

                <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;
                            padding-top:12px; border-top:1px solid #1e2433;">

                    <span style="display:inline-flex; align-items:center; gap:5px;
                                 background:rgba(74,222,128,0.08); border:1px solid rgba(74,222,128,0.2);
                                 border-radius:20px; padding:4px 12px; font-size:11px;
                                 color:{conf_color}; font-weight:500; letter-spacing:0.02em;">
                        ● {conf_label} confidence
                    </span>

                    <span style="display:inline-flex; align-items:center; gap:5px;
                                 background:#161c2a; border:1px solid #1e2433;
                                 border-radius:20px; padding:4px 12px;
                                 font-family:'DM Mono',monospace; font-size:11px; color:#475569;">
                        ⚡ {latency:.2f}s
                    </span>

                </div>
            </div>
            """, unsafe_allow_html=True)


            # raw chunks expander
            if result["chunks"]:
                with st.expander("View retrieved chunks", expanded=False):
                    for i, chunk in enumerate(result["chunks"], 1):
                        col = source_colour(chunk["source"])
                        st.markdown(f"""
                        <div style="background:#161c2a; border-left:3px solid {col};
                                    border-radius:0 8px 8px 0; padding:12px 14px;
                                    margin-bottom:8px;">
                            <div style="font-size:10px; color:#475569; font-family:'DM Mono',monospace;
                                        margin-bottom:6px; display:flex; justify-content:space-between;">
                                <span>{chunk['source']}</span>
                                <span>score: {chunk.get('rerank_score',0):.3f}</span>
                            </div>
                            <div style="font-size:12px; color:#94a3b8; line-height:1.6;">
                                {chunk['text'][:400]}{"..." if len(chunk['text']) > 400 else ""}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

# ── CHAT INPUT ────────────────────────────────────────────────────────────────
# st.chat_input is the CORRECT widget for chat apps in Streamlit.
# Unlike st.text_input + st.button, it doesn't cause duplicate reruns.
# It returns a value only when the user submits — never on keystroke.

query = st.chat_input("Ask a marketing question...")

# handle suggestion clicks from sidebar
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None

# ── PROCESS QUERY ─────────────────────────────────────────────────────────────

if query and query.strip():
    clean_query = query.strip()

    # 1. append user message immediately
    st.session_state.messages.append({
        "role": "user",
        "content": clean_query
    })

    # 2. show user bubble
    with st.chat_message("user", avatar="🧑"):
        st.markdown(f"""
        <div style="background:#1e1b4b; border:1px solid #3730a3; border-radius:12px;
                    padding:14px 18px; max-width:75%; margin-left:auto;
                    font-size:14px; color:#e2e8f0; line-height:1.6;">
            {clean_query}
        </div>
        """, unsafe_allow_html=True)

    # 3. run pipeline and show assistant response
    with st.chat_message("assistant", avatar="📊"):
        with st.spinner("Retrieving from 720 vectors..."):
            result = ask(
                clean_query,
                index, chunks,
                embed_model, bm25,
                rerank_model, client
            )

        conf_label, conf_color = get_confidence(result["chunks"])
        latency = result["latency"]
        answer  = result["answer"]

        # answer card
        st.markdown(f"""
        <div style="background:#0f1117; border:1px solid #1e2433;
                    border-radius:2px 14px 14px 14px; padding:18px 22px;
                    max-width:85%; margin-bottom:10px;">
            <div style="font-size:14px; color:#e2e8f0; line-height:1.8;
                        margin-bottom:16px; white-space:pre-wrap;">{answer}</div>
            <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;
                        padding-top:12px; border-top:1px solid #1e2433;">
                <span style="display:inline-flex; align-items:center; gap:5px;
                             background:rgba(74,222,128,0.08); border:1px solid rgba(74,222,128,0.2);
                             border-radius:20px; padding:4px 12px; font-size:11px;
                             color:{conf_color}; font-weight:500; letter-spacing:0.02em;">
                    ● {conf_label} confidence
                </span>
                <span style="display:inline-flex; align-items:center; gap:5px;
                             background:#161c2a; border:1px solid #1e2433;
                             border-radius:20px; padding:4px 12px;
                             font-family:'DM Mono',monospace; font-size:11px; color:#475569;">
                    ⚡ {latency:.2f}s
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)


        # raw chunks expander
        if result["chunks"]:
            with st.expander("View retrieved chunks", expanded=False):
                for i, chunk in enumerate(result["chunks"], 1):
                    col = source_colour(chunk["source"])
                    st.markdown(f"""
                    <div style="background:#161c2a; border-left:3px solid {col};
                                border-radius:0 8px 8px 0; padding:12px 14px;
                                margin-bottom:8px;">
                        <div style="font-size:10px; color:#475569; font-family:'DM Mono',monospace;
                                    margin-bottom:6px; display:flex; justify-content:space-between;">
                            <span>{chunk['source']}</span>
                            <span>score: {chunk.get('rerank_score',0):.3f}</span>
                        </div>
                        <div style="font-size:12px; color:#94a3b8; line-height:1.6;">
                            {chunk['text'][:400]}{"..." if len(chunk['text']) > 400 else ""}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # 4. save assistant result to session state
    st.session_state.messages.append({
        "role": "assistant",
        "result": result
    })

# ── FOOTER ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; padding:12px; border-top:1px solid #1e2433;
            margin-top:20px;">
    <span style="font-size:11px; color:#1e2433;">
        Answers grounded in indexed sources only · May not reflect real-time data
    </span>
</div>
""", unsafe_allow_html=True)
