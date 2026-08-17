"""
6b_error_analysis.py
What this does: a closer look at where Memory B goes wrong on the single-db
pilot -- prints gold vs. produced SQL/result for every failure, flags cases
where only Memory B failed (Baseline and A were both fine), and runs a
McNemar test to check if Baseline vs Memory B differs by more than chance.
Input: generation_results.json.
Output: printed failure detail + McNemar p-value.
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

db_path = "database/concert_singer/concert_singer.sqlite"

with open("generation_results.json") as f:
    results = json.load(f)

rows = []
for r in results:
    gold_result = execute_query(db_path, r["gold_sql"])
    baseline_sql = clean_markdown(r["sql_baseline"])
    A_sql = clean_markdown(r["sql_memory_A"])
    B_sql = clean_memory_B(r["sql_memory_B"])
    baseline_result = execute_query(db_path, baseline_sql)
    A_result = execute_query(db_path, A_sql)
    B_result = execute_query(db_path, B_sql)
    rows.append({
        "question": r["question"], "gold_sql": r["gold_sql"], "gold_result": gold_result,
        "baseline_sql": baseline_sql, "baseline_ok": baseline_result == gold_result,
        "A_sql": A_sql, "A_ok": A_result == gold_result,
        "B_sql": B_sql, "B_ok": B_result == gold_result, "B_result": B_result,
    })

total = len(rows)
print(f"Baseline: {sum(r['baseline_ok'] for r in rows)}/{total}")
print(f"Memory A: {sum(r['A_ok'] for r in rows)}/{total}")
print(f"Memory B: {sum(r['B_ok'] for r in rows)}/{total}")

print("\n=== MEMORY B FAILURES (detail) ===")
b_failures = [r for r in rows if not r["B_ok"]]
for r in b_failures:
    print("="*60)
    print("Question:", r["question"])
    print("Gold SQL:", r["gold_sql"], "| Gold result:", r["gold_result"])
    print("Memory B SQL:", r["B_sql"])
    print("Memory B result:", r["B_result"])
    print("Baseline also wrong here?", not r["baseline_ok"])
    print("Memory A also wrong here?", not r["A_ok"])

only_B_fails = [r for r in b_failures if r["baseline_ok"] and r["A_ok"]]
fails_everywhere = [r for r in b_failures if not r["baseline_ok"] and not r["A_ok"]]
print(f"\n=== OVERLAP ===")
print(f"Only Memory B fails (Baseline+A were correct): {len(only_B_fails)}")
print(f"Fails everywhere (hard question, all 3 wrong): {len(fails_everywhere)}")

baseline_right_B_wrong = sum(1 for r in rows if r["baseline_ok"] and not r["B_ok"])
B_right_baseline_wrong = sum(1 for r in rows if not r["baseline_ok"] and r["B_ok"])
n = baseline_right_B_wrong + B_right_baseline_wrong
print(f"\n=== McNemar (Baseline vs Memory B) ===")
print(f"Baseline correct, B wrong: {baseline_right_B_wrong}")
print(f"B correct, Baseline wrong: {B_right_baseline_wrong}")
try:
    from scipy.stats import binomtest
    if n > 0:
        p = binomtest(min(baseline_right_B_wrong, B_right_baseline_wrong), n, 0.5).pvalue
        print(f"Exact p-value: {p:.4f}")
    else:
        print("No discordant pairs.")
except ImportError:
    print("(χρειάζεται scipy: pip install scipy)")