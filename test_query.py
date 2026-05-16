import numpy as np
from sentence_transformers import SentenceTransformer
from retriever import load_index


# ── SEARCH ────────────────────────────────────────────────────────────────────

def search(query, index, chunks, model, top_k=5):
    """
    Embed the query using the same model used during indexing,
    then ask FAISS for the top_k most similar vectors.

    CRITICAL: you must use the same embedding model at query time
    as you used at index time. Mixing models produces garbage results
    because the vector spaces are incompatible.

    Returns a list of the top_k most relevant chunks with their
    similarity scores and source labels.
    """
    # embed the query — produces a (1, 384) array
    query_vector = model.encode([query], convert_to_numpy=True)

    # normalise to unit length — same as we did during indexing
    faiss.normalize_L2(query_vector)

    # search the index — returns distances and indices of top_k matches
    distances, indices = index.search(query_vector, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue  # FAISS returns -1 if fewer than top_k results exist
        results.append({
            "text":   chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "score":  float(1 - dist)  # convert L2 distance to similarity score
        })
    return results


# ── DISPLAY ───────────────────────────────────────────────────────────────────

def display_results(query, results):
    print(f"Query: {query}")
    print("=" * 60)
    for i, result in enumerate(results):
        print(f"\n[Result {i + 1}]")
        print(f"Source : {result['source']}")
        print(f"Score  : {result['score']:.4f}")
        print(f"Text   : {result['text'][:300]}")
        print("-" * 60)


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # load index and chunks
    index, chunks = load_index("index.faiss", "all_chunks.pkl")

    # load the same embedding model used during indexing
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded\n")

    # test queries — one per data source to verify all three are being retrieved
    test_queries = [
        "What is the ROI of email marketing campaigns?",
        "What are the key pillars of digital marketing strategy?",
        "Which marketing channel has the highest conversion rate?",
    ]

    for query in test_queries:
        results = search(query, index, chunks, model, top_k=3)
        display_results(query, results)
        print()

