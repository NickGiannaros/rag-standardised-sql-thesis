"""
7_measure_retrieval_quality.py
What this does: checks retrieval quality on its own, with no LLM involved.
For each dev question, we look at whether the top-1 (or any of the top-3)
retrieved example has the same structural "skeleton" as the gold query --
same WHERE/GROUP BY/HAVING/ORDER BY/LIMIT/JOIN/DISTINCT/aggregations present
or absent. Empty "CLAUSE NONE" boilerplate is stripped first so it doesn't
count as a real clause. This is the "before" measurement, run before the
boilerplate-dilution fix -- it's what first showed Memory A slightly ahead
of Memory B, the opposite of what we expected.
Input: memory_metadata.pkl, memory_A.index, memory_B.index, full Spider dev set.
Output: printed top-1 / top-3 match rates for Memory A and Memory B.
"""
import sys, os, json, pickle, re
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

index_A = faiss.read_index("memory_A.index")
index_B = faiss.read_index("memory_B.index")

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

print(f"\nTotal questions: {total}")
print(f"Memory A -- top-1 skeleton match: {top1_A}/{total} ({100*top1_A/total:.1f}%)")
print(f"Memory B -- top-1 skeleton match: {top1_B}/{total} ({100*top1_B/total:.1f}%)")
print(f"Memory A -- match in top-3: {any3_A}/{total} ({100*any3_A/total:.1f}%)")
print(f"Memory B -- match in top-3: {any3_B}/{total} ({100*any3_B/total:.1f}%)")
