import requests
import pickle
import time
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

# These are the Neil Patel pages we'll scrape.
# Each is a focused marketing stats article — clean, text-heavy, no login needed.
URLS = [
    "https://neilpatel.com/what-is-digital-marketing/",
    "https://neilpatel.com/what-is-content-marketing/",
    "https://neilpatel.com/what-is-seo/",
]

# Pretend to be a browser so the server doesn't block us.
# Many websites reject requests that don't have a User-Agent header.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ── 1. LOAD ───────────────────────────────────────────────────────────────────

def scrape_page(url):
    """
    Fetch one URL and extract clean text from it.

    Steps:
      1. requests.get() fetches the raw HTML (like opening a webpage)
      2. BeautifulSoup parses the HTML into a tree we can navigate
      3. We remove script/style tags — these are code, not content
      4. get_text() extracts all remaining visible text
      5. We clean up excessive whitespace
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()  # raises an error if status is 4xx or 5xx

        # parse the raw HTML — "html.parser" is Python's built-in parser
        soup = BeautifulSoup(response.text, "html.parser")

        # remove script and style tags entirely — they contain no useful content
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()  # decompose = remove the tag and its contents

        # extract all visible text from what's left
        text = soup.get_text(separator="\n")

        # clean up: collapse multiple blank lines into one
        lines = [line.strip() for line in text.splitlines()]
        clean_text = "\n".join(line for line in lines if line)

        print(f"  Scraped {len(clean_text)} characters from {url}")
        return clean_text

    except Exception as e:
        print(f"  ERROR scraping {url}: {e}")
        return ""  # return empty string so we can continue with other URLs


def load_web(urls):
    """
    Scrape all URLs and return a list of page dicts.
    We add a 1-second delay between requests — this is called being 'polite'.
    Hammering a server with rapid requests can get your IP blocked.
    """
    pages = []

    for url in urls:
        print(f"Scraping: {url}")
        text = scrape_page(url)

        if text:
            pages.append({
                "text": text,
                # extract a readable name from the URL for the source label
                "source": f"Neil Patel: {url.split('/')[-2].replace('-', ' ').title()}"
            })

        time.sleep(1)  # wait 1 second between requests — polite scraping

    print(f"\nLoaded {len(pages)} pages from web")
    return pages

# ── 2. CHUNK ──────────────────────────────────────────────────────────────────

def chunk_pages(pages):
    """
    Same chunking logic as ingest_pdf.py.
    The Load step changed, but Chunk and Save are identical — 
    this is intentional. RAG treats all text the same once it's clean.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = []

    for page in pages:
        splits = splitter.split_text(page["text"])

        for split in splits:
            chunks.append({
                "text": split,
                "source": page["source"]
            })

    print(f"Created {len(chunks)} chunks from web pages")
    return chunks

# ── 3. SAVE ───────────────────────────────────────────────────────────────────

def save_chunks(chunks, output_path):
    """Save chunks to disk as a pickle file."""
    with open(output_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved chunks to {output_path}")

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    output_path = "chunks_web.pkl"

    pages  = load_web(URLS)
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
        print("\nERROR: 0 chunks — paste the output here and we'll debug.")
