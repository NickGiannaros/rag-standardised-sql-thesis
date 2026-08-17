"""
6c_evaluate_multi_db.py
What this does: execution accuracy for the 4-database pilot extension (160
questions), broken down per database, plus a McNemar test on Baseline vs
Memory B.
Input: generation_results_multi.json.
Output: printed per-database and total summary + McNemar p-value.
"""
import json, sqlite3, re, sys, os
sys.path.insert(0, os.getcwd())
from eval import process_sql

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

with open("generation_results_multi.json") as f:
    results = json.load(f)

rows = []
for r in results:
    db_path = os.path.join("database", r["db_id"], r["db_id"] + ".sqlite")
    gold_result = execute_query(db_path, r["gold_sql"])
    baseline_sql = clean_markdown(r["sql_baseline"])
    A_sql = clean_markdown(r["sql_memory_A"])
    B_sql = clean_memory_B(r["sql_memory_B"])
    baseline_result = execute_query(db_path, baseline_sql)
    A_result = execute_query(db_path, A_sql)
    B_result = execute_query(db_path, B_sql)
    rows.append({
        "db_id": r["db_id"], "question": r["question"],
        "baseline_ok": baseline_result == gold_result,
        "A_ok": A_result == gold_result,
        "B_ok": B_result == gold_result,
    })

total = len(rows)
print(f"TOTAL — Baseline: {sum(r['baseline_ok'] for r in rows)}/{total}")
print(f"TOTAL — Memory A: {sum(r['A_ok'] for r in rows)}/{total}")
print(f"TOTAL — Memory B: {sum(r['B_ok'] for r in rows)}/{total}")

print("\nPer database:")
for db in sorted(set(r["db_id"] for r in rows)):
    sub = [r for r in rows if r["db_id"] == db]
    n = len(sub)
    print(f"{db}: Baseline {sum(r['baseline_ok'] for r in sub)}/{n} | A {sum(r['A_ok'] for r in sub)}/{n} | B {sum(r['B_ok'] for r in sub)}/{n}")

baseline_right_B_wrong = sum(1 for r in rows if r["baseline_ok"] and not r["B_ok"])
B_right_baseline_wrong = sum(1 for r in rows if not r["baseline_ok"] and r["B_ok"])
n = baseline_right_B_wrong + B_right_baseline_wrong
print(f"\nMcNemar — Baseline correct/B wrong: {baseline_right_B_wrong} | B correct/Baseline wrong: {B_right_baseline_wrong}")
try:
    from scipy.stats import binomtest
    if n > 0:
        p = binomtest(min(baseline_right_B_wrong, B_right_baseline_wrong), n, 0.5).pvalue
        print(f"p-value: {p:.4f}")
except ImportError:
    print("(pip install scipy)")