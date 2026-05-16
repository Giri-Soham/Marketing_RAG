import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── 1. MERGE ──────────────────────────────────────────────────────────────────

def load_all_chunks(chunk_files):
    """
    Load all three .pkl files and merge into one flat list.

    Each .pkl is a list of dicts: {"text": "...", "source": "..."}
    We simply concatenate them — order doesn't matter for FAISS.
    """
    all_chunks = []

    for path in chunk_files:
        if not os.path.exists(path):
            print(f"WARNING: {path} not found — skipping")
            continue

        with open(path, "rb") as f:
            chunks = pickle.load(f)

        print(f"  Loaded {len(chunks):>4} chunks from {path}")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks merged: {len(all_chunks)}")
    return all_chunks


# ── 2. EMBED ──────────────────────────────────────────────────────────────────

def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    """
    Convert every chunk's text into a 384-dimensional vector.

    Why all-MiniLM-L6-v2?
      - Free, runs on CPU (no GPU needed)
      - 384-dim output — small enough for fast search, rich enough for accuracy
      - Trained on 1B+ sentence pairs — strong semantic understanding
      - Used widely in production RAG systems

    SentenceTransformer.encode() handles batching internally.
    batch_size=64 means it processes 64 chunks at a time — memory efficient.
    show_progress_bar=True prints a progress bar so you know it's working.

    Output shape: (705, 384) — one 384-dim vector per chunk.
    """
    print(f"\nLoading embedding model: {model_name}")
    print("(First run downloads ~90MB — subsequent runs use cache)\n")

    model = SentenceTransformer(model_name)

    # extract just the text strings — model doesn't need the source labels
    texts = [chunk["text"] for chunk in chunks]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True   # FAISS needs numpy arrays, not tensors
    )

    print(f"\nEmbedding complete. Shape: {embeddings.shape}")
    # embeddings.shape will print something like (705, 384)
    # 720 = number of chunks, 384 = vector dimensions
    return embeddings


# ── 3. BUILD & SAVE FAISS INDEX ───────────────────────────────────────────────

def build_and_save_index(embeddings, chunks, index_path, chunks_path):
    """
    Build a FAISS index from the embeddings and save everything to disk.

    What is FAISS?
      FAISS (Facebook AI Similarity Search) is a library for fast nearest-neighbour
      search over dense vectors. Given a query vector, it finds the K most similar
      vectors in milliseconds — even across millions of entries.

    IndexFlatL2:
      The simplest FAISS index — does exact L2 (Euclidean) distance search.
      No approximation, guaranteed to find the true nearest neighbours.
      Good choice for <100k vectors (our 705 chunks is tiny by comparison).

    Why save chunks separately?
      FAISS only stores vectors — not the original text or source labels.
      When we retrieve chunk index 42, we need to look up chunks[42] to get
      the actual text and source. So we save both files and always load them together.
    """
    dimension = embeddings.shape[1]  # 384
    print(f"\nBuilding FAISS index (dimension={dimension})...")

    # normalise vectors to unit length — makes L2 distance equivalent to cosine similarity
    faiss.normalize_L2(embeddings)

    # create the index
    index = faiss.IndexFlatL2(dimension)

    # add all vectors to the index
    index.add(embeddings)
    print(f"Index built. Total vectors stored: {index.ntotal}")

    # save the FAISS index to disk
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to: {index_path}")

    # save the chunks list to disk (needed to retrieve text + source after search)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved all chunks to: {chunks_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # input: the three pkl files from Steps 2, 3, 4
    chunk_files = [
        "chunks_pdf.pkl",
        "chunks_web.pkl",
        "chunks_csv.pkl",
    ]

    # output: FAISS index + merged chunks list
    index_path  = "index.faiss"
    chunks_path = "all_chunks.pkl"

    print("=" * 50)
    print("Step 1: Merging chunks")
    print("=" * 50)
    all_chunks = load_all_chunks(chunk_files)

    print("\n" + "=" * 50)
    print("Step 2: Embedding chunks")
    print("=" * 50)
    embeddings = embed_chunks(all_chunks)

    print("\n" + "=" * 50)
    print("Step 3: Building and saving FAISS index")
    print("=" * 50)
    build_and_save_index(embeddings, all_chunks, index_path, chunks_path)

    print("\n" + "=" * 50)
    print("Phase 2 Steps 5 complete!")
    print(f"  index.faiss   — your searchable vector database")
    print(f"  all_chunks.pkl — text + source labels for retrieval")
    print("=" * 50)
