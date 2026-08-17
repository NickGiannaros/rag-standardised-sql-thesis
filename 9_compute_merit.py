"""
9_compute_merit.py
What this does: a simplified version of the "Merit" metric, from the paper
"Retrieval-Augmented NL2SQL Generation with Data-Centric Query Capsules for
Enterprise Applications" -- gives partial credit for "close but not exact"
answers instead of a strict right/wrong. Structural Score = how similar the
predicted and gold SQL trees are (sqlglot + zss, normalized tree-edit
distance); Semantic Score = cosine similarity between their sentence
embeddings (reusing the same all-MiniLM-L6-v2 model rather than adding
BERTScore); F1 = harmonic mean of the two; Merit = mean F1(condition) - mean
F1(Baseline). Simplified vs. the original paper: scored per whole query
instead of matching 8 separate SQL components. Doesn't need new API calls --
just reuses the SQL we already generated.
Input: generation_results.json, generation_results_multi.json,
generation_results_remaining.json, generation_results_compact_B.json.
Requires: pip install sqlglot zss.
Output: printed average F1 and Merit per condition.
"""
import json, re, sys, os
sys.path.insert(0, os.getcwd())
from eval import process_sql
import sqlglot
from sqlglot import exp
import zss
from sentence_transformers import SentenceTransformer
import numpy as np

def clean_markdown(sql):
    sql = re.sub(r"```sql\s*", "", sql)
    sql = re.sub(r"```", "", sql)
    return sql.strip()

def clean_memory_B(sql):
    sql = clean_markdown(sql)
    if "NONE" in sql:
        try:
            cleaned = process_sql.SimpleSQL_to_SQL(sql)
            if cleaned and cleaned != "SELECT ERROR":
                return cleaned
        except Exception:
            pass
    return sql

def to_zss_node(node):
    z = zss.Node(type(node).__name__)
    if isinstance(node, exp.Expression):
        for k, v in node.args.items():
            if isinstance(v, exp.Expression):
                z.addkid(to_zss_node(v))
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, exp.Expression):
                        z.addkid(to_zss_node(item))
    return z

def tree_size(node):
    return 1 + sum(tree_size(c) for c in node.children)

def structural_score(sql1, sql2):
    try:
        t1 = sqlglot.parse_one(sql1, read="sqlite")
        t2 = sqlglot.parse_one(sql2, read="sqlite")
        n1 = to_zss_node(t1)
        n2 = to_zss_node(t2)
        dist = zss.simple_distance(n1, n2)
        max_size = max(tree_size(n1), tree_size(n2), 1)
        return max(0.0, 1.0 - dist / max_size)
    except Exception:
        return 0.0

def f1(struct, sem, eps=1e-9):
    if struct + sem < eps:
        return 0.0
    return 2 * struct * sem / (struct + sem + eps)

all_raw = []
with open("generation_results.json") as f:
    single = json.load(f)
    for r in single:
        r["db_id"] = "concert_singer"
    all_raw.extend(single)
with open("generation_results_multi.json") as f:
    all_raw.extend(json.load(f))
with open("generation_results_remaining.json") as f:
    all_raw.extend(json.load(f))

old_by_key = {(r["db_id"], r["question"]): r for r in all_raw}

with open("generation_results_compact_B.json") as f:
    new_B = json.load(f)
new_by_key = {(r["db_id"], r["question"]): r for r in new_B}

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

f1_baseline, f1_A, f1_Bold, f1_Bnew = [], [], [], []
total = 0

keys = [k for k in old_by_key if k in new_by_key]
print("Total matched questions:", len(keys))

for i, key in enumerate(keys):
    r = old_by_key[key]
    gold_sql = r["gold_sql"]
    baseline_sql = clean_markdown(r["sql_baseline"])
    A_sql = clean_markdown(r["sql_memory_A"])
    Bold_sql = clean_memory_B(r["sql_memory_B"])
    Bnew_sql = clean_memory_B(new_by_key[key]["sql_memory_B_compact"])

    sqls = [gold_sql, baseline_sql, A_sql, Bold_sql, Bnew_sql]
    embs = model.encode(sqls, normalize_embeddings=True)

    sem_baseline = max(0.0, float(np.dot(embs[0], embs[1])))
    sem_A = max(0.0, float(np.dot(embs[0], embs[2])))
    sem_Bold = max(0.0, float(np.dot(embs[0], embs[3])))
    sem_Bnew = max(0.0, float(np.dot(embs[0], embs[4])))

    struct_baseline = structural_score(gold_sql, baseline_sql)
    struct_A = structural_score(gold_sql, A_sql)
    struct_Bold = structural_score(gold_sql, Bold_sql)
    struct_Bnew = structural_score(gold_sql, Bnew_sql)

    f1_baseline.append(f1(struct_baseline, sem_baseline))
    f1_A.append(f1(struct_A, sem_A))
    f1_Bold.append(f1(struct_Bold, sem_Bold))
    f1_Bnew.append(f1(struct_Bnew, sem_Bnew))

    total += 1
    if total % 100 == 0:
        print("Processed:", total)

avg_baseline = sum(f1_baseline) / total
avg_A = sum(f1_A) / total
avg_Bold = sum(f1_Bold) / total
avg_Bnew = sum(f1_Bnew) / total

print("\nTotal questions evaluated:", total)
print("Avg F1 -- Baseline:        {:.4f}".format(avg_baseline))
print("Avg F1 -- Memory A:        {:.4f}".format(avg_A))
print("Avg F1 -- Memory B (old):  {:.4f}".format(avg_Bold))
print("Avg F1 -- Memory B (new):  {:.4f}".format(avg_Bnew))

print("\nMerit (F1 with memory - F1 baseline):")
print("Merit A:        {:+.4f}".format(avg_A - avg_baseline))
print("Merit B (old):   {:+.4f}".format(avg_Bold - avg_baseline))
print("Merit B (new):   {:+.4f}".format(avg_Bnew - avg_baseline))
