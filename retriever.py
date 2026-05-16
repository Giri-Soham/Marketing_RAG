import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ── LOAD INDEX & CHUNKS ───────────────────────────────────────────────────────

def load_index(index_path="index.faiss", chunks_path="all_chunks.pkl"):
    """
    Load the FAISS index and chunks list from disk.
    Both files are always loaded together — FAISS stores vectors,
    chunks stores the text and source labels.
    """
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded index: {index.ntotal} vectors, {len(chunks)} chunks")
    return index, chunks


# ── 1. VECTOR SEARCH ──────────────────────────────────────────────────────────

def vector_search(query, index, chunks, model, top_k=10):
    """
    Embed the query and find the top_k most similar chunks by vector distance.

    This is exactly what test_query.py did in Phase 2 — we're just
    wrapping it in a clean function so retriever.py can call it.

    Returns a list of dicts with text, source, score, and index position.
    The index position is critical — sentence window retrieval uses it
    to grab neighbouring chunks.
    """
    query_vector = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "text":   chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "score":  float(1 - dist / 2),
            "index":  int(idx),          # position in the chunks list
            "method": "vector"
        })
    return results


# ── 2. BM25 SEARCH ────────────────────────────────────────────────────────────

def build_bm25(chunks):
    """
    Build a BM25 index from all chunks.

    BM25Okapi expects a list of tokenised documents — each document
    is a list of lowercase words. We do a simple whitespace split here.
    For production you'd use a proper tokeniser (spaCy, NLTK) but
    this is sufficient for our use case.

    Why build it fresh each time?
      BM25 is cheap to build (milliseconds for 705 chunks) and
      lives only in memory — no need to save to disk like FAISS.
    """
    tokenised = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenised)
    return bm25


def bm25_search(query, bm25, chunks, top_k=10):
    """
    Score every chunk against the query using BM25 and return top_k.

    BM25 scoring rewards:
      - Chunks that contain the query words many times (term frequency)
      - Chunks that contain rare words from the query (inverse document frequency)
      - Shorter chunks (length normalisation — a short chunk mentioning
        "conversion rate" twice is more focused than a long chunk mentioning
        it twice)

    This is why BM25 catches exact keyword matches that vector search misses.
    """
    tokenised_query = query.lower().split()
    scores = bm25.get_scores(tokenised_query)

    # get the indices of the top_k highest scores
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # skip chunks with zero relevance
            results.append({
                "text":   chunks[idx]["text"],
                "source": chunks[idx]["source"],
                "score":  float(scores[idx]),
                "index":  int(idx),
                "method": "bm25"
            })
    return results


# ── 3. SENTENCE WINDOW RETRIEVAL ──────────────────────────────────────────────

def expand_with_window(results, chunks, window=1):
    """
    For each retrieved chunk, also grab the chunks immediately before
    and after it in the original list (if they come from the same source).

    Why? A single 500-char chunk often lacks the context needed to
    answer a question fully. The answer might be split across two chunks.
    By including neighbours, we give the LLM more to work with.

    The same-source check is important — we don't want to grab a chunk
    from the Neil Patel web scrape as "context" for a PDF chunk.

    window=1 means 1 chunk before and 1 chunk after (±1).
    """
    expanded = list(results)  # start with original results
    seen_indices = {r["index"] for r in results}

    for result in results:
        idx = result["index"]
        source = result["source"]

        for offset in range(-window, window + 1):
            neighbour_idx = idx + offset
            if offset == 0:
                continue  # skip the chunk itself
            if neighbour_idx < 0 or neighbour_idx >= len(chunks):
                continue  # out of bounds
            if neighbour_idx in seen_indices:
                continue  # already included
            if chunks[neighbour_idx]["source"] != source:
                continue  # different source — don't cross boundaries

            expanded.append({
                "text":   chunks[neighbour_idx]["text"],
                "source": chunks[neighbour_idx]["source"],
                "score":  result["score"] * 0.85,  # slight score penalty for neighbours
                "index":  neighbour_idx,
                "method": "window"
            })
            seen_indices.add(neighbour_idx)

    return expanded


# ── 4. RECIPROCAL RANK FUSION ─────────────────────────────────────────────────

def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    """
    Merge results from vector search and BM25 into a single ranked list.

    RRF formula: score(chunk) = sum of 1/(k + rank) across all result lists
    where rank is the chunk's position in each list (1-indexed).

    Why RRF instead of just combining scores directly?
      - Vector scores and BM25 scores live on completely different scales
        (0-1 vs 0-∞). You can't just add them.
      - RRF only uses rank position, which is scale-independent.
      - A chunk ranked #1 in vector search AND #1 in BM25 gets
        a very high RRF score. A chunk only in one list gets a lower score.

    k=60 is the standard value from the original RRF paper (Cormack 2009).
    """
    scores = {}  # index → fused score
    sources = {}  # index → chunk data

    for rank, result in enumerate(vector_results, start=1):
        idx = result["index"]
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)
        sources[idx] = result

    for rank, result in enumerate(bm25_results, start=1):
        idx = result["index"]
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)
        if idx not in sources:
            sources[idx] = result

    # sort by fused score descending
    fused = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)

    return [
        {**sources[idx], "score": scores[idx], "method": "fused"}
        for idx in fused
    ]


# ── 5. MAIN RETRIEVE FUNCTION ─────────────────────────────────────────────────

def retrieve(query, index, chunks, model, bm25, top_k=20):
    """
    Full retrieval pipeline — called by rag_pipeline.py at query time.

    Steps:
      1. Vector search → top 10 semantic matches
      2. BM25 search   → top 10 keyword matches
      3. Sentence window expansion → add neighbouring chunks
      4. RRF fusion → merge into one ranked list
      5. Return top_k candidates for the re-ranker

    The re-ranker (reranker.py) will then pick the best 5 from these.
    """
    # step 1 & 2: parallel retrieval
    vec_results = vector_search(query, index, chunks, model, top_k=10)
    bm25_results = bm25_search(query, bm25, chunks, top_k=10)
    
    # step 3: expand with sentence window
    vec_results = expand_with_window(vec_results, chunks, window=1)
    
    # step 4: fuse results
    fused = reciprocal_rank_fusion(vec_results, bm25_results)

    # return top_k for re-ranking
    return fused[:top_k]


# ── SMOKE TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading index and model...")
    index, chunks = load_index()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    bm25  = build_bm25(chunks)

    query = "Which marketing channel has the highest conversion rate?"
    print(f"\nQuery: {query}")
    print("=" * 60)

    results = retrieve(query, index, chunks, model, bm25, top_k=10)

    print(f"Retrieved {len(results)} candidates\n")
    for i, r in enumerate(results[:5]):
        print(f"[{i+1}] Method: {r['method']}  |  Score: {r['score']:.4f}")
        print(f"    Source: {r['source']}")
        print(f"    Text  : {r['text'][:150]}")
        print()
