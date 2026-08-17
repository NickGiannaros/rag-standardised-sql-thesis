"""
4_build_memory.py
What this does: turns the texts from step 3 into embeddings
(all-MiniLM-L6-v2) and builds the two FAISS search indices -- Memory A (raw
SQL) and Memory B (Standardised SQL) -- so we can look up similar examples
later.
Input: texts_cache.pkl.
Output: memory_A.index, memory_B.index, memory_metadata.pkl (not committed --
regenerate from steps 1-4).
"""
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

with open("texts_cache.pkl", "rb") as f:
    cache = pickle.load(f)

texts_A = cache["texts_A"]
texts_B = cache["texts_B"]
metadata = cache["metadata"]

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding Memory A (raw SQL)...")
embeddings_A = model.encode(texts_A, show_progress_bar=True, normalize_embeddings=True)

print("Embedding Memory B (standardised SQL)...")
embeddings_B = model.encode(texts_B, show_progress_bar=True, normalize_embeddings=True)

dim = embeddings_A.shape[1]

index_A = faiss.IndexFlatIP(dim)
index_A.add(np.array(embeddings_A).astype("float32"))

index_B = faiss.IndexFlatIP(dim)
index_B.add(np.array(embeddings_B).astype("float32"))

faiss.write_index(index_A, "memory_A.index")
faiss.write_index(index_B, "memory_B.index")

with open("memory_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("Done.")
print("Memory A size:", index_A.ntotal)
print("Memory B size:", index_B.ntotal)