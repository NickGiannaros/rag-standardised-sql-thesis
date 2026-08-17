"""
6_evaluate_single_db.py
What this does: checks execution accuracy for the single-db pilot -- runs
the gold SQL and each condition's SQL against the real SQLite database and
compares results. Memory B's output first goes through Xander's
SimpleSQL_to_SQL() cleanup, because the model sometimes copies invalid
Standardised-SQL boilerplate (like "HAVING NONE") straight into its answer.
Input: generation_results.json.
Output: printed summary (how many Baseline/A/B got the right answer).
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

correct_baseline = 0
correct_A = 0
correct_B = 0
total = len(results)

for r in results:
    gold_result = execute_query(db_path, r["gold_sql"])

    baseline_sql = clean_markdown(r["sql_baseline"])
    A_sql = clean_markdown(r["sql_memory_A"])
    B_sql = clean_memory_B(r["sql_memory_B"])

    baseline_result = execute_query(db_path, baseline_sql)
    A_result = execute_query(db_path, A_sql)
    B_result = execute_query(db_path, B_sql)

    if baseline_result == gold_result: correct_baseline += 1
    if A_result == gold_result: correct_A += 1
    if B_result == gold_result: correct_B += 1

print("SUMMARY (after SimpleSQL_to_SQL cleanup on Memory B)")
print("Baseline:", str(correct_baseline) + "/" + str(total))
print("Memory A:", str(correct_A) + "/" + str(total))
print("Memory B:", str(correct_B) + "/" + str(total))