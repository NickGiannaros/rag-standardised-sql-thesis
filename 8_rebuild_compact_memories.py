"""
8_rebuild_compact_memories.py
What this does: the actual fix. Standardised SQL writes out every clause
even when it's empty ("GROUP BY NONE ORDER BY NONE HAVING NONE ..."), and on
simple queries that boilerplate ends up dominating the embedded text,
drowning out what the query is actually about. This script rebuilds Memory B
with those empty "CLAUSE NONE" lines stripped -- but only from the text used
for embedding/search. The LLM still sees the full, correct Standardised SQL
in the final prompt; nothing changes about what it's shown. Memory A is
rebuilt too, with identical text-construction code, so the comparison stays
fair.
Input: memory_metadata.pkl.
Output: memory_A_v2.index, memory_B_compact.index (boilerplate stripped from
embedding text only).
"""
import sys, os, re, pickle
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

texts_A, texts_B = [], []
for i, m in enumerate(metadata):
    schema_str = get_schema_str(m["db_id"])
    texts_A.append("Question: " + m["question"] + " Schema: " + schema_str + " SQL: " + m["raw_sql"])
    compact_b = strip_none_clauses(m["standardised_sql"])
    texts_B.append("Question: " + m["question"] + " Schema: " + schema_str + " SQL: " + compact_b)
    if (i + 1) % 1000 == 0:
        print("Built texts:", i + 1)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding Memory A (v2, consistent format)...")
emb_A = np.array(model.encode(texts_A, normalize_embeddings=True, show_progress_bar=True)).astype("float32")
index_A2 = faiss.IndexFlatIP(emb_A.shape[1])
index_A2.add(emb_A)
faiss.write_index(index_A2, "memory_A_v2.index")

print("Embedding Memory B (compact, NONE stripped)...")
emb_B = np.array(model.encode(texts_B, normalize_embeddings=True, show_progress_bar=True)).astype("float32")
index_B2 = faiss.IndexFlatIP(emb_B.shape[1])
index_B2.add(emb_B)
faiss.write_index(index_B2, "memory_B_compact.index")

print("Done. memory_A_v2.index and memory_B_compact.index saved.")
