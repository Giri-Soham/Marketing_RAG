from sentence_transformers import CrossEncoder

# ── LOAD MODEL ────────────────────────────────────────────────────────────────

def load_reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """
    Load the cross-encoder re-ranking model.

    What is a cross-encoder?
      A bi-encoder (like all-MiniLM-L6-v2) encodes query and document
      SEPARATELY into vectors, then compares them.

      A cross-encoder encodes query and document TOGETHER in one pass.
      It can see the full relationship between them — which words in the
      chunk respond to which words in the query.

      This makes it much more accurate, but also much slower — you can't
      pre-compute embeddings because the query changes every time.
      That's why we only re-rank the top 10-20 candidates, not all 715.

    ms-marco-MiniLM-L-6-v2:
      Trained on the MS MARCO dataset — 500k+ real search queries paired
      with relevant passages. This is the standard re-ranking model used
      in production RAG systems. Free, CPU-compatible, ~80MB download.
    """
    print(f"Loading re-ranker model: {model_name}")
    model = CrossEncoder(model_name)
    print("Re-ranker loaded\n")
    return model


# ── RERANK ────────────────────────────────────────────────────────────────────

def rerank(query, candidates, model, top_k=5):
    """
    Score each candidate chunk against the query using the cross-encoder,
    then return the top_k highest-scoring chunks.

    How it works:
      1. Build pairs: [(query, chunk1_text), (query, chunk2_text), ...]
      2. Feed all pairs through the cross-encoder in one batch
      3. Model outputs a relevance score for each pair
      4. Sort by score descending, return top_k

    The scores are raw logits (unbounded floats, e.g. -5 to +10).
    Higher = more relevant. We don't need to normalise them —
    we only care about relative ranking, not absolute values.
    """
    if not candidates:
        return []

    # build (query, chunk_text) pairs for the cross-encoder
    pairs = [(query, candidate["text"]) for candidate in candidates]

    # score all pairs in one batch — cross-encoder handles batching internally
    scores = model.predict(pairs, show_progress_bar=False)

    # attach scores back to candidates
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    # sort by rerank score descending and return top_k
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


# ── SMOKE TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import faiss
    import pickle
    from sentence_transformers import SentenceTransformer
    from retriever import load_index, build_bm25, retrieve

    # load everything
    index, chunks = load_index()
    embed_model   = SentenceTransformer("all-MiniLM-L6-v2")
    bm25          = build_bm25(chunks)
    rerank_model  = load_reranker()

    # test with the previously weak query
    query = "Which marketing channel has the highest conversion rate?"
    print(f"Query: {query}")
    print("=" * 60)

    # retrieve 20 candidates
    candidates = retrieve(query, index, chunks, embed_model, bm25, top_k=20)
    print(f"Candidates from retriever: {len(candidates)}")

    # re-rank to top 5
    final = rerank(query, candidates, rerank_model, top_k=5)

    print(f"\nFinal top 5 after re-ranking:\n")
    for i, result in enumerate(final):
        print(f"[{i+1}] Rerank score: {result['rerank_score']:.4f}")
        print(f"    Source: {result['source']}")
        print(f"    Text  : {result['text'][:200]}")
        print()
