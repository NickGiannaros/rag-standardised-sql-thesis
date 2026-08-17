"""
10c_measure_retrieval_significance_mpnet.py
What this does: the same paired McNemar significance test as
8c_measure_retrieval_significance.py, but for the mpnet robustness check.
Result: 267 both-correct, 50 full-only, 73 compact-only, 644 both-wrong ->
p = 0.0469 -- significant, confirms the effect isn't specific to
all-MiniLM-L6-v2.
Input: memory_metadata.pkl, memory_B_full_mpnet.index,
memory_B_compact_mpnet.index, full Spider dev set. Requires: pip install
statsmodels.
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

index_B_full = faiss.read_index("memory_B_full_mpnet.index")
index_B_compact = faiss.read_index("memory_B_compact_mpnet.index")

print("Loading embedding model (all-mpnet-base-v2)...")
model = SentenceTransformer("all-mpnet-base-v2")

print("Loading full Spider dev set...")
spider = load_dataset("xlangai/spider")
dev_full = list(spider["validation"])
print("Total dev questions:", len(dev_full))

both_correct = full_only = compact_only = both_wrong = 0
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

    _, idx_full = index_B_full.search(query_emb, 1)
    _, idx_compact = index_B_compact.search(query_emb, 1)

    full_correct = extract_skeleton(metadata[idx_full[0][0]]["standardised_sql"]) == gold_skel
    compact_correct = extract_skeleton(metadata[idx_compact[0][0]]["standardised_sql"]) == gold_skel

    if full_correct and compact_correct:
        both_correct += 1
    elif full_correct and not compact_correct:
        full_only += 1
    elif compact_correct and not full_correct:
        compact_only += 1
    else:
        both_wrong += 1

    total += 1
    if total % 100 == 0:
        print("Processed:", total)

print("\nTotal questions evaluated:", total)
print("Both correct:", both_correct)
print("Full-only correct:", full_only)
print("Compact-only correct:", compact_only)
print("Both wrong:", both_wrong)

table = [[both_correct, full_only], [compact_only, both_wrong]]
result = mcnemar(table, exact=True)
print("\nMcNemar (B full vs B compact, mpnet) statistic: {:.4f}  p-value: {:.4f}".format(result.statistic, result.pvalue))
