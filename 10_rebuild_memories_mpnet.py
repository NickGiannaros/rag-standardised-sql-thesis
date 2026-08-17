"""
10_rebuild_memories_mpnet.py
What this does: a robustness check -- rebuilds Memory A, full Memory B, and
compact Memory B using a second, different embedding model
(all-mpnet-base-v2, bigger than the all-MiniLM-L6-v2 used everywhere else),
to see whether the compact-fix improvement holds up or was specific to one
model.
Input: memory_metadata.pkl.
Output: memory_A_mpnet.index, memory_B_full_mpnet.index,
memory_B_compact_mpnet.index.
"""
import sys, os, pickle, re
sys.path.insert(0, os.getcwd())
from eval import process_sql
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def strip_none_clauses(sql):
    lines = sql.split("\n")
    kept = [l for l in lines if not re.match(
        r"^\s*(GROUP BY|ORDER BY|HAVING|LIMIT|INTERSECT|EXCEPT|UNION|WHERE)\s+NONE\s*$", l)]
    return "\n".join(kept).strip()

with open("memory_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print("Entries:", len(metadata))

schema_cache = {}
def get_schema_str(db_id):
    if db_id not in schema_cache:
        db_path = os.path.join("database", db_id, db_id + ".sqlite")
        schema_cache[db_id] = str(process_sql.get_schema(db_path))
    return schema_cache[db_id]

texts_A, texts_B_full, texts_B_compact = [], [], []
for i, m in enumerate(metadata):
    schema_str = get_schema_str(m["db_id"])
    texts_A.append("Question: " + m["question"] + " Schema: " + schema_str + " SQL: " + m["raw_sql"])
    texts_B_full.append("Question: " + m["question"] + " Schema: " + schema_str + " SQL: " + m["standardised_sql"])
    compact_b = strip_none_clauses(m["standardised_sql"])
    texts_B_compact.append("Question: " + m["question"] + " Schema: " + schema_str + " SQL: " + compact_b)
    if (i + 1) % 1000 == 0:
        print("Built texts:", i + 1)

print("Loading embedding model (all-mpnet-base-v2)...")
model = SentenceTransformer("all-mpnet-base-v2")

def build_index(texts, name):
    print("Embedding", name, "...")
    emb = np.array(model.encode(texts, normalize_embeddings=True, show_progress_bar=True)).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, name + ".index")
    print("Saved", name + ".index")

build_index(texts_A, "memory_A_mpnet")
build_index(texts_B_full, "memory_B_full_mpnet")
build_index(texts_B_compact, "memory_B_compact_mpnet")

print("Done.")
