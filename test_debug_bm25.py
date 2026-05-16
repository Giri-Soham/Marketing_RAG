import pickle
from rank_bm25 import BM25Okapi
import numpy as np

with open("all_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

tokenised = [chunk["text"].lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenised)

query = "Which marketing channel has the highest conversion rate?"
scores = bm25.get_scores(query.lower().split())
top_indices = np.argsort(scores)[::-1][:5]

print("Top 5 BM25 results:\n")
for idx in top_indices:
    print(f"Score : {scores[idx]:.4f}")
    print(f"Source: {chunks[idx]['source']}")
    print(f"Text  : {chunks[idx]['text'][:150]}")
    print()

