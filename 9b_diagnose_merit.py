"""
9b_diagnose_merit.py
What this does: digs into the Merit result to check if it's a fair signal or
a side effect of writing style. Restricts the F1 comparison to only the
questions where a condition was already execution-correct -- if Memory B
still scores lower there, it's not just because it gets more wrong answers,
its correct answers must also be phrased differently. Result: the gap
roughly halves but doesn't disappear -- about half looks like a real
quality difference, half looks like a style effect (Memory B's correct
answers use fully-qualified table.column names more often, which reads less
like Spider's natural gold-answer style than Memory A's correct answers do).
Input: generation_results.json, generation_results_multi.json,
generation_results_remaining.json, generation_results_compact_B.json.
Output: printed avg F1 restricted to exec-correct rows, per condition.
"""
import json, re, sys, os, sqlite3
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

def execute_query(db_path, sql):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return sorted(rows)
    except Exception as e:
        return "ERROR: " + str(e)

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

keys = [k for k in old_by_key if k in new_by_key]
print("Total matched questions:", len(keys))

rows = []
total = 0
for key in keys:
    r = old_by_key[key]
    db_path = os.path.join("database", r["db_id"], r["db_id"] + ".sqlite")
    gold_sql = r["gold_sql"]
    gold_result = execute_query(db_path, gold_sql)

    baseline_sql = clean_markdown(r["sql_baseline"])
    A_sql = clean_markdown(r["sql_memory_A"])
    Bold_sql = clean_memory_B(r["sql_memory_B"])
    Bnew_sql = clean_memory_B(new_by_key[key]["sql_memory_B_compact"])

    baseline_ok = execute_query(db_path, baseline_sql) == gold_result
    A_ok = execute_query(db_path, A_sql) == gold_result
    Bold_ok = execute_query(db_path, Bold_sql) == gold_result
    Bnew_ok = execute_query(db_path, Bnew_sql) == gold_result

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

    rows.append({
        "baseline_ok": baseline_ok, "A_ok": A_ok, "Bold_ok": Bold_ok, "Bnew_ok": Bnew_ok,
        "f1_baseline": f1(struct_baseline, sem_baseline),
        "f1_A": f1(struct_A, sem_A),
        "f1_Bold": f1(struct_Bold, sem_Bold),
        "f1_Bnew": f1(struct_Bnew, sem_Bnew),
    })

    total += 1
    if total % 100 == 0:
        print("Processed:", total)

def report(label, ok_key, f1_key):
    correct_rows = [r for r in rows if r[ok_key]]
    n = len(correct_rows)
    avg_f1_correct = sum(r[f1_key] for r in correct_rows) / n if n else 0
    avg_f1_all = sum(r[f1_key] for r in rows) / len(rows)
    print(f"{label}: exec-correct count={n} | avg F1 on exec-correct only={avg_f1_correct:.4f} | avg F1 overall={avg_f1_all:.4f}")

print("\n--- Diagnostic: avg F1 restricted to exec-correct answers only ---")
report("Baseline", "baseline_ok", "f1_baseline")
report("Memory A", "A_ok", "f1_A")
report("Memory B (old)", "Bold_ok", "f1_Bold")
report("Memory B (new/compact)", "Bnew_ok", "f1_Bnew")
