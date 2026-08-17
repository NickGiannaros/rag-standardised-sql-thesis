"""
10b_measure_retrieval_quality_mpnet.py
What this does: the same skeleton-matching check as
7_measure_retrieval_quality.py / 8b_measure_retrieval_quality_compact.py,
run this time with all-mpnet-base-v2 for all three memory versions at once.
Result: A 31.6% top-1, full B 30.7%, compact B 32.9% -- the fix holds up
(and does slightly better) on a second embedding model.
Input: memory_metadata.pkl, memory_A_mpnet.index, memory_B_full_mpnet.index,
memory_B_compact_mpnet.index, full Spider dev set.
Output: printed top-1 / top-3 match rates for all three memories.
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

index_A = faiss.read_index("memory_A_mpnet.index")
index_B_full = faiss.read_index("memory_B_full_mpnet.index")
index_B_compact = faiss.read_index("memory_B_compact_mpnet.index")

print("Loading embedding model (all-mpnet-base-v2)...")
model = SentenceTransformer("all-mpnet-base-v2")

print("Loading full Spider dev set...")
spider = load_dataset("xlangai/spider")
dev_full = list(spider["validation"])
print("Total dev questions:", len(dev_full))

k = 3
top1_A = top1_Bfull = top1_Bcompact = 0
any3_A = any3_Bfull = any3_Bcompact = 0
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
    _, idx_Bfull = index_B_full.search(query_emb, k)
    _, idx_Bcompact = index_B_compact.search(query_emb, k)

    skels_A = [extract_skeleton(metadata[i]["raw_sql"]) for i in idx_A[0]]
    skels_Bfull = [extract_skeleton(metadata[i]["standardised_sql"]) for i in idx_Bfull[0]]
    skels_Bcompact = [extract_skeleton(metadata[i]["standardised_sql"]) for i in idx_Bcompact[0]]

    if skels_A and skels_A[0] == gold_skel:
        top1_A += 1
    if skels_Bfull and skels_Bfull[0] == gold_skel:
        top1_Bfull += 1
    if skels_Bcompact and skels_Bcompact[0] == gold_skel:
        top1_Bcompact += 1
    if any(s == gold_skel for s in skels_A):
        any3_A += 1
    if any(s == gold_skel for s in skels_Bfull):
        any3_Bfull += 1
    if any(s == gold_skel for s in skels_Bcompact):
        any3_Bcompact += 1

    total += 1
    if total % 100 == 0:
        print("Processed:", total)

print("\nTotal questions evaluated:", total)
print("Memory A (mpnet)          top-1: {}/{} ({:.1f}%)".format(top1_A, total, 100*top1_A/total))
print("Memory B full (mpnet)     top-1: {}/{} ({:.1f}%)".format(top1_Bfull, total, 100*top1_Bfull/total))
print("Memory B compact (mpnet)  top-1: {}/{} ({:.1f}%)".format(top1_Bcompact, total, 100*top1_Bcompact/total))
print("Memory A (mpnet)          top-3: {}/{} ({:.1f}%)".format(any3_A, total, 100*any3_A/total))
print("Memory B full (mpnet)     top-3: {}/{} ({:.1f}%)".format(any3_Bfull, total, 100*any3_Bfull/total))
print("Memory B compact (mpnet)  top-3: {}/{} ({:.1f}%)".format(any3_Bcompact, total, 100*any3_Bcompact/total))
