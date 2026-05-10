import fitz  # PyMuPDF
import os
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 1. LOAD ───────────────────────────────────────────────────────────────────

def load_pdf(path):
    """
    Extract text from each page of a text-based PDF using PyMuPDF.
    Each page becomes one item in the list, tagged with its source label.
    """
    doc = fitz.open(path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        if text.strip():  # skip blank or image-only pages
            pages.append({
                "text": text,
                "source": f"Smart Insights: Future of Digital Marketing — page {page_num + 1}"
            })

    print(f"Loaded {len(pages)} pages from PDF")
    return pages

# ── 2. CHUNK ──────────────────────────────────────────────────────────────────

def chunk_pages(pages):
    """
    Split each page's text into overlapping chunks of ~500 characters.

    Why chunk?
      - LLMs have context limits and can't take a full report at once
      - Smaller chunks give more precise similarity matches during retrieval
      - Overlap (50 chars) ensures sentences aren't cut off at boundaries
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # max characters per chunk
        chunk_overlap=50,     # overlap between consecutive chunks
        separators=["\n\n", "\n", ".", " "]  # split at natural breaks first
    )

    chunks = []

    for page in pages:
        splits = splitter.split_text(page["text"])

        for split in splits:
            # each chunk carries its source label for citation later
            chunks.append({
                "text": split,
                "source": page["source"]
            })

    print(f"Created {len(chunks)} chunks from PDF")
    return chunks

# ── 3. SAVE ───────────────────────────────────────────────────────────────────

def save_chunks(chunks, output_path):
    """
    Serialize the chunks list to disk using pickle.
    This is equivalent to df.to_csv() — saves Python objects to a binary file.
    build_index.py will load this file later to build the FAISS index.
    """
    with open(output_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved chunks to {output_path}")

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pdf_path    = "data/imh_report_2024.pdf"
    output_path = "chunks_pdf.pkl"

    if not os.path.exists(pdf_path):
        print(f"ERROR: Could not find {pdf_path}")
        print("Download the Smart Insights PDF and save it to data/imh_report_2024.pdf")
    else:
        pages  = load_pdf(pdf_path)
        chunks = chunk_pages(pages)
        save_chunks(chunks, output_path)

        if chunks:
            print("\nDone! Preview of first 3 chunks:\n")
            for i, chunk in enumerate(chunks[:3]):
                print(f"--- Chunk {i + 1} ---")
                print(f"Source : {chunk['source']}")
                print(f"Text   : {chunk['text'][:200]}")
                print()
        else:
            print("\nERROR: 0 chunks — paste the verify check output here and we'll debug.")
