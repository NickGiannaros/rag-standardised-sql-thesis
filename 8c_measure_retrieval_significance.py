"""
8c_measure_retrieval_significance.py
What this does: checks whether the compact-memory improvement in
8b_measure_retrieval_quality_compact.py is statistically real, using a
paired McNemar test on the same 1034 questions (old Memory B vs compact
Memory B). Result: 180 both-correct, 16 old-only, 31 new-only, 807
both-wrong -> p = 0.0400 -- yes, statistically significant.
Input: memory_metadata.pkl, memory_B.index, memory_B_compact.index, full
Spider dev set. Requires: pip install statsmodels.
Output: printed contingency table + McNemar statistic and p-value.
"""
import sys, os, pickle, re
sys.path.insert(0, os.getcwd())
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from statsmodels.stats.contingency_tables import mcnemar
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

index_B_old = faiss.read_index("memory_B.index")
index_B_new = faiss.read_index("memory_B_compact.index")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading full Spider dev set...")
spider = load_dataset("xlangai/spider")
dev_full = list(spider["validation"])
print("Total dev questions:", len(dev_full))

both_correct = old_only = new_only = both_wrong = 0
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

    _, idx_old = index_B_old.search(query_emb, 1)
    _, idx_new = index_B_new.search(query_emb, 1)

    old_correct = extract_skeleton(metadata[idx_old[0][0]]["standardised_sql"]) == gold_skel
    new_correct = extract_skeleton(metadata[idx_new[0][0]]["standardised_sql"]) == gold_skel

    if old_correct and new_correct:
        both_correct += 1
    elif old_correct and not new_correct:
        old_only += 1
    elif new_correct and not old_correct:
        new_only += 1
    else:
        both_wrong += 1

    total += 1
    if total % 100 == 0:
        print("Processed:", total)

print("\nTotal questions evaluated:", total)
print("Both correct:", both_correct)
print("Old-only correct (old right, new wrong):", old_only)
print("New-only correct (new right, old wrong):", new_only)
print("Both wrong:", both_wrong)

table = [[both_correct, old_only], [new_only, both_wrong]]
result = mcnemar(table, exact=True)
print("\nMcNemar (old Memory B vs compact Memory B) statistic: {:.4f}  p-value: {:.4f}".format(result.statistic, result.pvalue))
