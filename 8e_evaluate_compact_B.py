"""
8e_evaluate_compact_B.py
What this does: compares execution accuracy of old Memory B vs compact
Memory B (with Baseline/A for context) across all 1034 questions. Result:
Baseline 737, Memory A 731, Memory B old 733, Memory B compact 732 --
McNemar p = 1.0000. The retrieval-quality improvement doesn't carry over to
execution accuracy.
Input: generation_results.json, generation_results_multi.json,
generation_results_remaining.json, generation_results_compact_B.json.
Output: printed summary, per-database table, three McNemar p-values.
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

rows = []
for key, r in old_by_key.items():
    if key not in new_by_key:
        continue
    db_path = os.path.join("database", r["db_id"], r["db_id"] + ".sqlite")
    gold_result = execute_query(db_path, r["gold_sql"])
    baseline_sql = clean_markdown(r["sql_baseline"])
    A_sql = clean_markdown(r["sql_memory_A"])
    B_old_sql = clean_memory_B(r["sql_memory_B"])
    B_new_sql = clean_memory_B(new_by_key[key]["sql_memory_B_compact"])

    baseline_result = execute_query(db_path, baseline_sql)
    A_result = execute_query(db_path, A_sql)
    B_old_result = execute_query(db_path, B_old_sql)
    B_new_result = execute_query(db_path, B_new_sql)

    rows.append({
        "db_id": r["db_id"], "question": r["question"],
        "baseline_ok": baseline_result == gold_result,
        "A_ok": A_result == gold_result,
        "B_old_ok": B_old_result == gold_result,
        "B_new_ok": B_new_result == gold_result,
    })

total = len(rows)
print("Total matched questions:", total)
print(f"TOTAL -- Baseline: {sum(r['baseline_ok'] for r in rows)}/{total}")
print(f"TOTAL -- Memory A: {sum(r['A_ok'] for r in rows)}/{total}")
print(f"TOTAL -- Memory B (old): {sum(r['B_old_ok'] for r in rows)}/{total}")
print(f"TOTAL -- Memory B (compact): {sum(r['B_new_ok'] for r in rows)}/{total}")

print("\nPer database:")
for db in sorted(set(r["db_id"] for r in rows)):
    sub = [r for r in rows if r["db_id"] == db]
    n = len(sub)
    print(f"{db}: Baseline {sum(r['baseline_ok'] for r in sub)}/{n} | A {sum(r['A_ok'] for r in sub)}/{n} | B_old {sum(r['B_old_ok'] for r in sub)}/{n} | B_new {sum(r['B_new_ok'] for r in sub)}/{n}")

def mcnemar_report(label, key1, key2):
    only1 = sum(1 for r in rows if r[key1] and not r[key2])
    only2 = sum(1 for r in rows if r[key2] and not r[key1])
    both = sum(1 for r in rows if r[key1] and r[key2])
    neither = sum(1 for r in rows if not r[key1] and not r[key2])
    from statsmodels.stats.contingency_tables import mcnemar
    table = [[both, only1], [only2, neither]]
    result = mcnemar(table, exact=True)
    print(f"\nMcNemar ({label}) -- only-{key1}: {only1} | only-{key2}: {only2} | p-value: {result.pvalue:.4f}")

mcnemar_report("Baseline vs Memory B compact", "baseline_ok", "B_new_ok")
mcnemar_report("Memory A vs Memory B compact", "A_ok", "B_new_ok")
mcnemar_report("Memory B old vs Memory B compact", "B_old_ok", "B_new_ok")
