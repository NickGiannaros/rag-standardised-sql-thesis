"""
8b_measure_retrieval_quality_compact.py
What this does: reruns the same skeleton-matching check as
7_measure_retrieval_quality.py, but on the fixed (compact) indices. Result:
Memory A is unchanged (20.8% / 32.0%), Memory B improves from 19.0% to 20.4%
top-1 and from 31.2% to 32.5% top-3 -- now ahead of Memory A on top-3.
Input: memory_metadata.pkl, memory_A_v2.index, memory_B_compact.index, full
Spider dev set.
Output: printed top-1 / top-3 match rates for both memories.
"""
import sys, os, pickle, re
sys.path.insert(0, os.getcwd())
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
import faiss

AGG_FUNCS = ["count", "avg", "sum", "max", "min"]

def extract_skeleton(sql):
    s = sql.lower()
    for clause in ["where", "group by", "having", "order by", "limit", "intersect", "except", "union"]:
        s = re.sub(r"\b" + re.escape(clause) + r"\s+none\b", "", s)
    return (
        bool(re.search(r"\bwhere\b", s)),
        bool(re.search(r"\bgroup by\b", s)),
        bool(re.search(r"\bhaving\b", s)),
        bool(re.search(r"\border by\b", s)),
        bool(re.search(r"\blimit\b", s)),
        bool(re.search(r"\bjoin\b", s)),
        bool(re.search(r"\bdistinct\b", s)),
        frozenset(f for f in AGG_FUNCS if re.search(r"\b" + f + r"\s*\(", s)),
    )

with open("memory_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

index_A = faiss.read_index("memory_A_v2.index")
index_B = faiss.read_index("memory_B_compact.index")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading full Spider dev set...")
spider = load_dataset("xlangai/spider")
dev_full = list(spider["validation"])
print("Total dev questions:", len(dev_full))

k = 3
top1_A = top1_B = any3_A = any3_B = 0
total = 0

for q in dev_full:
    db_id = q["db_id"]
    db_path = os.path.join("database", db_id, db_id + ".sqlite")
    try:
        from eval import process_sql
        schema_str = str(process_sql.get_schema(db_path))
    except Exception:
        continue

    gold_skel = extract_skeleton(q["query"])
    query_text = "Question: " + q["question"] + " Schema: " + schema_str
    query_emb = model.encode([query_text], normalize_embeddings=True)
    query_emb = np.array(query_emb).astype("float32")

    _, idx_A = index_A.search(query_emb, k)
    _, idx_B = index_B.search(query_emb, k)
    examples_A = [metadata[i] for i in idx_A[0]]
    examples_B = [metadata[i] for i in idx_B[0]]

    skels_A = [extract_skeleton(m["raw_sql"]) for m in examples_A]
    skels_B = [extract_skeleton(m["standardised_sql"]) for m in examples_B]

    if skels_A and skels_A[0] == gold_skel:
        top1_A += 1
    if skels_B and skels_B[0] == gold_skel:
        top1_B += 1
    if any(s == gold_skel for s in skels_A):
        any3_A += 1
    if any(s == gold_skel for s in skels_B):
        any3_B += 1

    total += 1
    if total % 100 == 0:
        print("Processed:", total)

print("\nTotal questions evaluated:", total)
print("Memory A (v2)      top-1 skeleton match: {}/{} ({:.1f}%)".format(top1_A, total, 100*top1_A/total))
print("Memory B (compact) top-1 skeleton match: {}/{} ({:.1f}%)".format(top1_B, total, 100*top1_B/total))
print("Memory A (v2)      match in top-3:       {}/{} ({:.1f}%)".format(any3_A, total, 100*any3_A/total))
print("Memory B (compact) match in top-3:       {}/{} ({:.1f}%)".format(any3_B, total, 100*any3_B/total))
