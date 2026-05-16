import os
import time
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from retriever import load_index, build_bm25, retrieve
from re_ranker import load_reranker, rerank

# load environment variables from .env
load_dotenv()

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

EMBED_MODEL   = "all-MiniLM-L6-v2"
RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL     = "llama-3.3-70b-versatile"
TOP_K_RETRIEVE = 20               # candidates sent to re-ranker
TOP_K_RERANK   = 5                # final chunks sent to LLM
INDEX_PATH     = "index.faiss"
CHUNKS_PATH    = "all_chunks.pkl"

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a marketing analytics assistant with access to a 
knowledge base containing marketing research reports, articles, and campaign 
performance data.

Your job is to answer the user's question using ONLY the context provided below. 

Rules you must follow:
1. Base your answer strictly on the provided context — do not use outside knowledge.
2. Always cite your sources by referencing the [PDF Source N], [Web Source N], or [CSV Source N] labels inline.
3. If the context does not contain enough information to answer the question, 
   say clearly: "I don't have enough information in my knowledge base to answer this."
4. Be concise and direct. Lead with the answer, then provide supporting detail.
5. When quoting numbers or statistics, always mention which source they came from.
"""

# ── PROMPT BUILDER ────────────────────────────────────────────────────────────

def _source_type(source: str) -> str:
    if "Smart Insights" in source or source.lower().endswith(".pdf"):
        return "PDF"
    elif "Neil Patel" in source or "patel" in source.lower():
        return "Web"
    return "CSV"


def build_prompt(query, reranked_chunks):
    context_parts = []
    for i, chunk in enumerate(reranked_chunks, start=1):
        stype = _source_type(chunk["source"])
        context_parts.append(
            f"[{stype} Source {i}] {chunk['source']}\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    return prompt


# ── LLM CALL ─────────────────────────────────────────────────────────────────

def call_llm(system_prompt, user_prompt, client, model=LLM_MODEL):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.8,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


# ── CITATION FORMATTER ────────────────────────────────────────────────────────

def format_sources(reranked_chunks):
    """
    Build a clean source list to display below the answer.
    Shows the source label and name for every chunk used as context.
    """
    lines = ["\n--- Sources used ---"]
    seen = set()
    for i, chunk in enumerate(reranked_chunks, start=1):
        source = chunk["source"]
        if source not in seen:
            lines.append(f"[Source {i}] {source}")
            seen.add(source)
    return "\n".join(lines)


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def ask(query, index, chunks, embed_model, bm25, rerank_model, client):
    """
    Full end-to-end RAG pipeline:
      1. Retrieve top-20 candidates (vector + BM25 + sentence window + RRF)
      2. Re-rank to top-5 using cross-encoder
      3. Build prompt with context + question
      4. Call OpenAI LLM
      5. Return answer + sources + timing

    This function is what the Streamlit frontend will call in Phase 4.
    """
    start = time.time()

    # step 1: retrieve
    candidates = retrieve(query, index, chunks, embed_model, bm25, top_k=TOP_K_RETRIEVE)

    # step 2: re-rank
    final_chunks = rerank(query, candidates, rerank_model, top_k=TOP_K_RERANK)

    # step 3: build prompt
    user_prompt = build_prompt(query, final_chunks)

    # step 4: call LLM
    answer = call_llm(SYSTEM_PROMPT, user_prompt, client)

    # step 5: format sources
    sources = format_sources(final_chunks)

    elapsed = time.time() - start

    return {
        "answer":   answer,
        "sources":  sources,
        "chunks":   final_chunks,
        "latency":  elapsed,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not found in .env file")
        exit(1)

    print("Loading models and index...")
    client = Groq(api_key=api_key)
    index, chunks = load_index(INDEX_PATH, CHUNKS_PATH)
    embed_model  = SentenceTransformer(EMBED_MODEL)
    bm25         = build_bm25(chunks)
    rerank_model = load_reranker(RERANK_MODEL)
    print("All models loaded.\n")

    # test with all three original smoke test queries
    test_queries = [
        "What is the ROI of email marketing campaigns?",
        "What are the key pillars of digital marketing strategy?",
        "Which marketing channel has the highest conversion rate?",
    ]

    for query in test_queries:
        print("=" * 65)
        print(f"QUESTION: {query}")
        print("=" * 65)

        result = ask(query, index, chunks, embed_model, bm25, rerank_model, client)

        print(f"\nANSWER:\n{result['answer']}")
        print(result["sources"])
        print(f"\nLatency: {result['latency']:.2f}s")
        print()
